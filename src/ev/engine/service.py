"""EV GUI engine 命令调度、生命周期与后台任务。"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import threading
import uuid
from dataclasses import asdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import numpy as np

from ..asr.adapters import SpeakerEmbeddingAdapter
from ..audio.capture import AudioCapture
from ..audio.devices import list_input_devices, resolve_device
from ..audio.environment import EnvironmentMonitor, EnvEvent
from ..config import Settings
from ..model_catalog import get_catalog, get_definition
from ..model_download import DownloadCancelled, ModelDownloader
from ..model_registry import ModelRegistry, migrate_from_legacy
from ..models import require_models, verify_models
from ..pipeline.runtime import transcribe_forever
from ..speaker.profile import VoiceProfileManager
from ..speaker.verification import build_profile, normalize_embedding
from ..store.audio import archive_wav, read_wav
from ..store.db import Store
from ..store.environment import EnvironmentLog
from .protocol import EngineRequest, ProtocolWriter


LOGGER = logging.getLogger(__name__)


def _resample_np(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    if orig_rate == target_rate:
        return audio
    if audio.size == 0:
        return audio
    duration = len(audio) / orig_rate
    target_length = max(1, int(duration * target_rate))
    indices = np.linspace(0, len(audio) - 1, target_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class EngineService:
    def __init__(self, settings: Settings, output: TextIO = sys.stdout):
        self.settings = settings
        self.writer = ProtocolWriter(output)
        self.state = "stopped"
        self.device_selector: str | None = None
        self._running = True
        self._listen_thread: threading.Thread | None = None
        self._listen_stop = threading.Event()
        self._download_thread: threading.Thread | None = None
        self._download_cancel = threading.Event()
        self._voice_auto_learn = settings.voice_learning.auto_learn_enabled
        self._segment_worker = None  # type: ignore[assignment]
        self._last_listen_params: dict | None = None
        self._env_log: EnvironmentLog | None = None
        self._env_monitor: EnvironmentMonitor | None = None

        self._registry = ModelRegistry(
            models_root=settings.models_dir,
            emit=lambda kind, payload: self.writer.emit(kind, payload),
        )
        self._ensure_registry_populated()
        # Voice enrollment mode (reuses main pipeline logic)
        self._enroll_thread: threading.Thread | None = None
        self._enroll_stop = threading.Event()
        self._enroll_capture: AudioCapture | None = None
        self._enroll_frames: list[np.ndarray] = []
        self._enroll_device: str | None = None

    def serve(self, input_stream: TextIO = sys.stdin) -> int:
        self.settings.ensure_dirs()
        self._migrate_sample_audio()
        LOGGER.info("engine service started")
        self._emit_state()
        for line in input_stream:
            if not self._running:
                break
            if not line.strip():
                continue
            try:
                request = EngineRequest.parse(line)
                LOGGER.info("engine command: %s", request.command)
                self.handle(request)
            except Exception as exc:
                self.writer.error(str(exc), code="invalid_request")
        LOGGER.info("engine input closed")
        self.shutdown()
        return 0

    def handle(self, request: EngineRequest) -> None:
        handlers = {
            "get_status": self._get_status,
            "list_devices": self._list_devices,
            "verify_models": self._verify_models,
            "download_models": self._download_models,
            "cancel_download": self._cancel_download,
            "list_available_models": self._list_available_models,
            "list_installed_models": self._list_installed_models,
            "install_model": self._install_model,
            "uninstall_model": self._uninstall_model,
            "set_active_model": self._set_active_model,
            "reload_registry": self._reload_registry,
            "start_listening": self._start_listening,
            "stop_listening": self._stop_listening,
            "set_device": self._set_device,
            "set_thresholds": self._set_thresholds,
            "list_voice_samples": self._list_voice_samples,
            "delete_voice_sample": self._delete_voice_sample,
            "promote_voice_sample": self._promote_voice_sample,
            "list_pending_voice_samples": self._list_pending_voice_samples,
            "confirm_voice_sample": self._confirm_voice_sample,
            "reject_voice_sample": self._reject_voice_sample,
            "reset_voice_profile": self._reset_voice_profile,
            "learn_voice_samples": self._learn_voice_samples,
            "set_voice_learning": self._set_voice_learning,
            "capture_manual_sample": self._capture_manual_sample,
            "start_voice_enrollment": self._start_voice_enrollment,
            "stop_voice_enrollment": self._stop_voice_enrollment,
            "list_segments": self._list_segments,
            "delete_segment": self._delete_segment,
            "delete_all_segments": self._delete_all_segments,
            "delete_quality_rejected_segments": self._delete_quality_rejected_segments,
            "submit_manual_query": self._submit_manual_query,
            "delete_query": self._delete_query,
            "delete_all_queries": self._delete_all_queries,
            "list_lexicon": self._list_lexicon,
            "add_lexicon_word": self._add_lexicon_word,
            "update_lexicon_word": self._update_lexicon_word,
            "delete_lexicon_word": self._delete_lexicon_word,
            "clear_auto_lexicon": self._clear_auto_lexicon,
            "correct_segment": self._correct_segment,
            "list_corrections": self._list_corrections,
            "learn_corrections": self._learn_corrections,
            "shutdown": self._shutdown_command,
        }
        handler = handlers.get(request.command)
        if handler is None:
            self.writer.error(
                f"未知命令: {request.command}", request.request_id, "unknown_command"
            )
            return
        try:
            handler(request)
        except Exception as exc:
            self.writer.error(str(exc), request.request_id)

    def _get_status(self, request: EngineRequest) -> None:
        self._emit_state(request.request_id)
        funasr_ready = importlib.util.find_spec("funasr") is not None
        torch_ready = importlib.util.find_spec("torch") is not None
        self.writer.emit(
            "runtime_status",
            {
                "ready": funasr_ready and torch_ready,
                "funasr": funasr_ready,
                "torch": torch_ready,
            },
            request.request_id,
        )
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        self.writer.emit("profile_status", profile, request.request_id)

    def _profile_status(self, store: Store) -> dict:
        profile = store.profile_status(
            max_core_samples=self.settings.speaker.max_core_samples,
            max_cache_samples=self.settings.speaker.max_cache_samples,
            max_centroids=self.settings.speaker.max_centroids,
        )
        profile["auto_learn"] = self._voice_auto_learn
        return profile

    def _list_devices(self, request: EngineRequest) -> None:
        devices = [asdict(device) for device in list_input_devices()]
        self.writer.emit("device_list", {"devices": devices}, request.request_id)

    def _verify_models(self, request: EngineRequest) -> None:
        """使用 Registry 校验所有模型状态。"""
        statuses = self._registry.get_all_slot_status()
        self.writer.emit(
            "model_status",
            {
                "status": "ready" if self._registry.are_all_slots_ready() else "missing",
                "models": statuses,
                "all_ready": self._registry.are_all_slots_ready(),
            },
            request.request_id,
        )

    def _download_models(self, request: EngineRequest) -> None:
        """批量下载旧4个模型（向后兼容）。"""
        if self._download_thread and self._download_thread.is_alive():
            raise RuntimeError("模型下载已在进行")
        self._download_cancel = threading.Event()

        def run() -> None:
            try:
                downloader = ModelDownloader(
                    self.settings.models,
                    lambda kind, payload: self.writer.emit(kind, payload, request.request_id),
                    self._download_cancel,
                )
                downloader.download_all()
            except DownloadCancelled as exc:
                self.writer.emit("model_status", {"status": "cancelled", "message": str(exc)}, request.request_id)
            except Exception as exc:
                self.writer.error(str(exc), request.request_id, "model_download_failed")

        self._download_thread = threading.Thread(target=run, name="ev-model-download", daemon=True)
        self._download_thread.start()
        self._ack(request)

    def _cancel_download(self, request: EngineRequest) -> None:
        self._download_cancel.set()
        self._registry.cancel_download()
        self._ack(request)

    # ── 新 Registry 管理端点 ──────────────────────────────────────

    def _list_available_models(self, request: EngineRequest) -> None:
        catalog = get_catalog()
        models_list = []
        for key, defn in catalog.items():
            models_list.append({
                "key": key,
                "name": defn.name,
                "type": defn.type.value,
                "source": defn.source.value,
                "description": defn.description,
                "needs_tokens": defn.needs_tokens,
                "needs_seg_dict": defn.needs_seg_dict,
                "estimated_size_bytes": defn.estimated_size_bytes,
                "min_memory_gb": defn.min_memory_gb,
            })
        self.writer.emit("available_models", {"models": models_list}, request.request_id)

    def _list_installed_models(self, request: EngineRequest) -> None:
        self._list_installed_models_impl(request.request_id)

    def _list_installed_models_impl(self, request_id: str | None = None) -> None:
        statuses = self._registry.get_all_slot_status()
        installed = self._registry.list_installed()
        installed_list = []
        for key, mod in installed.items():
            status = self._registry.get_slot_status(
                next((s for s, a in self._registry.list_slot_assignments().items() if a.model_key == key), "")
            ) if any(a.model_key == key for a in self._registry.list_slot_assignments().values()) else None
            installed_list.append({
                "key": key,
                "local_path": mod.local_path,
                "installed_at": mod.installed_at,
                "size_bytes": mod.size_bytes,
                "ready": status["ready"] if status else False,
                "path": status["path"] if status else None,
                "errors": status["errors"] if status else ["未知状态"],
            })
        assignments = self._registry.list_slot_assignments()
        slot_list = [
            {
                "slot": slot,
                "model_key": a.model_key,
                "enabled": a.enabled,
                "status": self._registry.get_slot_status(slot),
            }
            for slot, a in assignments.items()
        ]
        self.writer.emit("installed_models", {
            "installed": installed_list,
            "slots": slot_list,
            "all_ready": self._registry.are_all_slots_ready(),
        }, request_id)

    def _install_model(self, request: EngineRequest) -> None:
        model_key = request.payload.get("model_key")
        if not model_key:
            raise RuntimeError("缺少 model_key 参数")
        if self._download_thread and self._download_thread.is_alive():
            raise RuntimeError("已有下载任务在进行")
        self._registry.cancel_download()

        def run() -> None:
            try:
                self._registry.install_model(model_key)
                self.writer.emit("model_install_status", {
                    "key": model_key,
                    "status": "ready",
                })
            except Exception as exc:
                is_cancelled = "取消" in str(exc)
                self.writer.emit("model_install_status", {
                    "key": model_key,
                    "status": "cancelled" if is_cancelled else "error",
                    "error": str(exc),
                })
                if not is_cancelled:
                    self.writer.error(str(exc), request.request_id, "model_install_failed")
            finally:
                # Refresh lists after install completes/fails
                self._list_installed_models_impl()

        self._download_thread = threading.Thread(target=run, name=f"ev-install-{model_key}", daemon=True)
        self._download_thread.start()
        self._ack(request)

    def _cancel_download(self, request: EngineRequest) -> None:
        self._download_cancel.set()
        self._registry.cancel_download()
        self._ack(request)

    def _uninstall_model(self, request: EngineRequest) -> None:
        model_key = request.payload.get("model_key")
        if not model_key:
            raise RuntimeError("缺少 model_key 参数")
        self._registry.uninstall_model(model_key)
        self._list_installed_models_impl()
        # Trigger final ASR reload if the running worker exists
        if self._segment_worker is not None:
            self._segment_worker.reload_final_asr()
        self._ack(request)

    def _set_active_model(self, request: EngineRequest) -> None:
        slot = request.payload.get("slot")
        raw_key = request.payload.get("model_key")
        if not slot:
            raise RuntimeError("缺少 slot 参数")
        # Normalize: empty string means unassign
        model_key = raw_key if (raw_key and raw_key != "") else None
        self._registry.set_active_model(slot, model_key)
        self._list_installed_models_impl()
        # Reload final ASR if slot affects asr_final and worker is running
        if slot == "asr_final" and self._segment_worker is not None:
            self._segment_worker.reload_final_asr()
        self._ack(request)

    def _reload_registry(self, request: EngineRequest) -> None:
        """刷新注册表状态（从磁盘重新加载）。"""
        self._registry._lock.acquire()
        try:
            self._registry.state = self._registry._load_state()
        finally:
            self._registry._lock.release()
        statuses = self._registry.get_all_slot_status()
        self.writer.emit("model_status", {
            "status": "ready" if self._registry.are_all_slots_ready() else "missing",
            "models": statuses,
            "all_ready": self._registry.are_all_slots_ready(),
        }, request.request_id)
        self._ack(request)

    def _start_listening(self, request: EngineRequest) -> None:
        if self._listen_thread and self._listen_thread.is_alive():
            raise RuntimeError("监听已经启动")
        # asr_final 由 registry 动态管理，不通过旧 toml 路径校验
        require_models(self.settings.models, skip_keys=frozenset({"asr_final"}))
        if not list_input_devices():
            raise RuntimeError("未发现可用输入设备，请检查麦克风权限和系统输入设置")
        self.device_selector = request.payload.get("device") or None
        threshold = float(
            request.payload.get("threshold",
                request.payload.get("user_threshold", self.settings.speaker.threshold)
            )
        )
        auto_learn = bool(request.payload.get("auto_learn", self._voice_auto_learn))
        if not (0.1 <= threshold <= 0.9):
            raise ValueError("阈值必须在0.1到0.9之间")
        # Remember params for potential restart (e.g. device change)
        self._last_listen_params = {
            "device": self.device_selector,
            "threshold": threshold,
            "auto_learn": auto_learn,
        }
        session_settings = replace(
            self.settings,
            speaker=replace(
                self.settings.speaker,
                threshold=threshold,
            ),
            voice_learning=replace(
                self.settings.voice_learning,
                auto_learn_enabled=auto_learn,
            ),
        )
        self._voice_auto_learn = auto_learn
        # Persist threshold for future sessions
        self.settings = session_settings
        device = resolve_device(self.device_selector)
        self._listen_stop = threading.Event()
        self._segment_worker = None
        self.state = "loading"
        self._emit_state(request.request_id)
        worker_holder: dict = {}

        # 环境感知: 通过 Registry 解析 YAMNet 模型路径
        self._env_log = EnvironmentLog(self.settings.logs_dir)
        self._env_monitor = None
        env_model = self._registry.get_active_model("environment")
        if env_model is None:
            # Fallback: registry 未分配 environment 槽位时，直接查找已安装的 yamnet
            env_model = self._registry.state.installed.get("yamnet")
        if env_model is not None:
            env_path = Path(env_model.local_path)
            yamnet_model = env_path / "yamnet.tflite"
            yamnet_labels = env_path / "yamnet_class_map.csv"
            # tar.gz 解压可能多嵌套一层目录
            if not yamnet_model.exists():
                yamnet_model = env_path / "yamnet" / "yamnet.tflite"
                yamnet_labels = env_path / "yamnet" / "yamnet_class_map.csv"
            if yamnet_model.exists() and yamnet_labels.exists():
                self._env_monitor = EnvironmentMonitor(
                    model_path=str(yamnet_model),
                    label_path=str(yamnet_labels),
                    sample_rate=self.settings.audio.sample_rate,
                )
                LOGGER.info("YAMNet environment monitor ready: %s", env_path)
            else:
                LOGGER.info("YAMNet model files incomplete in %s, environment monitoring disabled", env_path)
        else:
            LOGGER.info("YAMNet model not assigned to 'environment' slot, environment monitoring disabled")

        def resolve_final_asr_path() -> Path:
            model = self._registry.get_active_model("asr_final")
            if model is None:
                # Fallback: registry 未配置 asr_final 时用旧 toml 路径
                # （不做 require_models 严格校验，交给下层适配器在真正加载时报错）
                return self.settings.models.root / self.settings.models.asr_final
            return Path(model.local_path)

        # 解析降噪模型路径: registry → fallback installed → None
        denoiser_path: str | None = None
        denoise_model = self._registry.get_active_model("speech_enhancement")
        if denoise_model is None:
            denoise_model = self._registry.state.installed.get("dfsmn-ans")
        if denoise_model is not None:
            denoiser_path = denoise_model.local_path

        def run() -> None:
            try:
                asyncio.run(
                    transcribe_forever(
                        session_settings,
                        device,
                        output=lambda message: None,
                        stop_event=self._listen_stop,
                        emit=self._on_runtime_event,
                        worker_holder=worker_holder,
                        final_asr_resolver=resolve_final_asr_path,
                        env_monitor=self._env_monitor,
                        denoiser_path=denoiser_path,
                    )
                )
                self.state = "stopped"
                self._emit_state()
            except Exception as exc:
                self.state = "error"
                self.writer.error(str(exc), code="listening_failed")
                self._emit_state()
            finally:
                self._segment_worker = None

        self._listen_thread = threading.Thread(target=run, name="ev-listening", daemon=True)
        self._listen_thread.start()
        # Wait briefly for worker to be created
        for _ in range(20):
            if "worker" in worker_holder:
                self._segment_worker = worker_holder["worker"]
                break
            self._listen_thread.join(0.05)

    def _stop_listening(self, request: EngineRequest) -> None:
        self._listen_stop.set()
        if self.state not in ("stopped", "error"):
            self.state = "stopping"
        self._emit_state(request.request_id)

    def _set_thresholds(self, request: EngineRequest) -> None:
        threshold = request.payload.get("threshold", request.payload.get("user_threshold"))
        t = float(threshold) if threshold is not None else None
        if t is not None and not (0.1 <= t <= 0.9):
            raise ValueError("阈值必须在0.1到0.9之间")
        # Update running worker if listening
        if self._segment_worker is not None:
            self._segment_worker.update_thresholds(t)
        # Persist to settings object for future sessions
        if t is not None:
            self.settings = replace(
                self.settings,
                speaker=replace(self.settings.speaker, threshold=t),
            )
        self._ack(request)

    def _set_device(self, request: EngineRequest) -> None:
        new_device = request.payload.get("device") or None
        was_listening = self._listen_thread is not None and self._listen_thread.is_alive()
        # 监听中禁止切设备,必须先停止监听:避免静默重启造成"监听中断闪烁"感
        if was_listening:
            raise RuntimeError(
                "当前正在监听中，无法切换输入设备。请先停止监听，再切换麦克风。"
            )
        # Validate device exists if specified (空字符串表示使用系统默认)
        if new_device is not None and new_device != "":
            resolved = resolve_device(new_device)
            if resolved is None:
                raise ValueError(f"未找到匹配的输入设备: {new_device}")
        # Normalize selector: 空字符串/None 都归为 None (即系统默认)
        selector: str | None = None if (new_device is None or new_device == "") else new_device
        self.device_selector = selector
        # Update saved params for future start_listening
        if self._last_listen_params is not None:
            self._last_listen_params["device"] = selector
        # Sync engine_state.device 字段, 让 UI 显示当前选择
        self._emit_state(request.request_id)
        self._ack(request)

    def _on_runtime_event(self, event_type: str, payload: dict) -> None:
        if event_type == "capture_started":
            self.state = "listening"
            self._emit_state()
        elif event_type == "speech_started":
            self.state = "speech"
            self._emit_state()
        elif event_type == "speech_ended":
            if not self._listen_stop.is_set():
                self.state = "listening"
            self._emit_state()
        elif event_type == "voice_sample_added":
            with Store(self.settings.db_path) as store:
                profile = self._profile_status(store)
            profile["auto_learn"] = self._voice_auto_learn
            self.writer.emit("profile_status", profile)
        elif event_type == "environment_event":
            # 写入环境事件日志
            if self._env_log is not None:
                event = EnvEvent(
                    timestamp=float(payload.get("timestamp", 0)),
                    category=str(payload.get("category", "unknown")),
                    confidence=float(payload.get("confidence", 0)),
                    duration_sec=payload.get("duration_sec"),
                )
                self._env_log.append(event)
        self.writer.emit(event_type, payload)

    def _list_voice_samples(self, request: EngineRequest) -> None:
        limit = int(request.payload.get("limit", 50))
        tier = request.payload.get("tier") or None
        with Store(self.settings.db_path) as store:
            samples = store.list_voice_samples(tier=tier, limit=limit)
        self.writer.emit("voice_samples", {"samples": samples}, request.request_id)

    def _promote_voice_sample(self, request: EngineRequest) -> None:
        sample_id = str(request.payload.get("sample_id", ""))
        if not sample_id:
            raise ValueError("缺少 sample_id")
        with Store(self.settings.db_path) as store:
            vp = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
            promoted = vp.promote_sample(sample_id)
        self.writer.emit(
            "voice_sample_promoted",
            {"sample_id": sample_id, "promoted": promoted},
            request.request_id,
        )
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)
        self._ack(request)

    def _list_pending_voice_samples(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            vp = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
            pending = vp.pending_samples(
                self.settings.voice_learning.pending_distance_threshold
            )
        self.writer.emit(
            "pending_voice_samples", {"samples": pending}, request.request_id
        )

    def _confirm_voice_sample(self, request: EngineRequest) -> None:
        """确认待确认样本：晋升为核心样本。"""
        sample_id = str(request.payload.get("sample_id", ""))
        if not sample_id:
            raise ValueError("缺少 sample_id")
        with Store(self.settings.db_path) as store:
            vp = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
            promoted = vp.promote_sample(sample_id)
        self.writer.emit(
            "voice_sample_confirmed",
            {"sample_id": sample_id, "promoted": promoted},
            request.request_id,
        )
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)
        self._ack(request)

    def _reject_voice_sample(self, request: EngineRequest) -> None:
        """删除待确认样本（用户确认不是自己）。"""
        sample_id = str(request.payload.get("sample_id", ""))
        if not sample_id:
            raise ValueError("缺少 sample_id")
        with Store(self.settings.db_path) as store:
            sample = store.get_voice_sample(sample_id)
            deleted = store.delete_voice_sample(sample_id)
            if deleted and sample and sample.get("audio_path"):
                self._unlink_sample_audio(str(sample["audio_path"]))
        self.writer.emit(
            "voice_sample_rejected",
            {"sample_id": sample_id, "deleted": deleted},
            request.request_id,
        )
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)
        self._ack(request)

    def _delete_voice_sample(self, request: EngineRequest) -> None:
        sample_id = str(request.payload.get("sample_id", ""))
        if not sample_id:
            raise ValueError("缺少 sample_id")
        with Store(self.settings.db_path) as store:
            sample = store.get_voice_sample(sample_id)
            deleted = store.delete_voice_sample(sample_id)
            if deleted and sample and sample.get("audio_path"):
                self._unlink_sample_audio(str(sample["audio_path"]))
        self.writer.emit(
            "voice_sample_deleted",
            {"sample_id": sample_id, "deleted": deleted},
            request.request_id,
        )
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _reset_voice_profile(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            paths = store.list_all_voice_sample_paths()
            count = store.delete_all_voice_samples()
        for raw in paths:
            self._unlink_sample_audio(raw)
        self.writer.emit("voice_profile_reset", {"deleted": count}, request.request_id)
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _unlink_sample_audio(self, raw_path: str) -> None:
        """Delete a voice-sample wav ONLY if it lives in the managed dir.

        Historical auto-samples may still reference archive/ segment wavs —
        those belong to segment history and must not be removed here.
        """
        path = Path(raw_path)
        target = self.settings.voice_samples_dir.resolve()
        try:
            is_managed = path.resolve().is_relative_to(target)
        except (OSError, ValueError):
            is_managed = False
        if is_managed:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("failed to remove sample wav: %s", path)

    def _migrate_sample_audio(self) -> None:
        """One-time migration: copy sample wavs living outside the managed
        voice-samples dir into it, so sample audio survives history clears."""
        try:
            managed = self.settings.voice_samples_dir.resolve()
            with Store(self.settings.db_path) as store:
                for sample in store.list_voice_samples(limit=200):
                    raw = sample.get("audio_path") or ""
                    if not raw:
                        continue
                    src = Path(raw)
                    try:
                        if src.resolve().is_relative_to(managed) or not src.exists():
                            continue
                    except (OSError, ValueError):
                        continue
                    dest = managed / src.name
                    if not dest.exists():
                        dest.write_bytes(src.read_bytes())
                    store.update_voice_sample_audio_path(sample["id"], str(dest))
                    LOGGER.info("voice sample audio migrated: %s -> %s", src, dest)
        except Exception as exc:
            LOGGER.warning("voice-sample audio migration skipped: %s", exc)

    def _learn_voice_samples(self, request: EngineRequest) -> None:
        """Re-embed every sample from its wav with the current speaker model,
        then rebuild centroids. Samples whose wav is missing are skipped."""
        def run() -> None:
            try:
                paths = require_models(
                    self.settings.models,
                    self.settings.models.root,
                    skip_keys=frozenset({"asr_final"}),
                )
                speaker = SpeakerEmbeddingAdapter(str(paths["speaker"]))
                updated = 0
                missing = 0
                target_sr = self.settings.audio.sample_rate
                with Store(self.settings.db_path) as store:
                    for sample in store.list_voice_samples(limit=200):
                        raw = sample.get("audio_path") or ""
                        if not raw:
                            missing += 1
                            continue
                        path = Path(raw)
                        if not path.exists():
                            missing += 1
                            continue
                        try:
                            audio, sr = read_wav(path)
                        except Exception as exc:
                            LOGGER.warning("re-learn skip %s: %s", path, exc)
                            missing += 1
                            continue
                        if sr != target_sr:
                            audio = _resample_np(audio, sr, target_sr)
                            sr = target_sr
                        embedding = speaker.embed(audio, sr)
                        store.update_voice_sample_embedding(sample["id"], embedding)
                        updated += 1
                    vp = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
                    vp._rebuild_centroids()
                self.writer.emit(
                    "voice_samples_learned",
                    {"updated": updated, "missing": missing},
                    request.request_id,
                )
                with Store(self.settings.db_path) as store:
                    profile = self._profile_status(store)
                profile["auto_learn"] = self._voice_auto_learn
                self.writer.emit("profile_status", profile)
            except Exception as exc:
                LOGGER.exception("voice sample re-learn failed")
                self.writer.error(str(exc), request.request_id, "voice_sample_learn_failed")

        threading.Thread(target=run, name="ev-voice-learn", daemon=True).start()
        self._ack(request)

    def _set_voice_learning(self, request: EngineRequest) -> None:
        enabled = bool(request.payload.get("enabled", True))
        self._voice_auto_learn = enabled
        self._ack(request)
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _capture_manual_sample(self, request: EngineRequest) -> None:
        duration_sec = float(request.payload.get("duration_sec", 3.0))
        duration_sec = max(1.5, min(duration_sec, 10.0))
        duration_ms = int(duration_sec * 1000)
        device = resolve_device(self.device_selector)

        def run():
            try:
                self.writer.emit("manual_sample_status", {
                    "status": "recording",
                    "duration_ms": duration_ms,
                })
                sample_rate = self.settings.audio.sample_rate
                total_samples = int(sample_rate * duration_sec)
                capture = AudioCapture(self.settings.audio, device=device, frame_ms=50)
                frames = []
                collected = 0
                capture.start()
                try:
                    loop = asyncio.new_event_loop()
                    try:
                        async def collect():
                            nonlocal collected
                            async for frame in capture.frames():
                                frames.append(frame)
                                collected += len(frame)
                                if collected >= total_samples:
                                    break
                        loop.run_until_complete(collect())
                    finally:
                        loop.close()
                finally:
                    capture.stop()
                if not frames:
                    raise RuntimeError("未采集到音频")
                audio = np.concatenate(frames)[:total_samples]
                if len(audio) < total_samples * 0.5:
                    raise RuntimeError("录制音频过短")
                self.writer.emit("manual_sample_status", {"status": "processing"})
                paths = require_models(self.settings.models, self.settings.models.root, skip_keys=frozenset({"asr_final"}))
                speaker = SpeakerEmbeddingAdapter(str(paths["speaker"]))
                embedding = speaker.embed(audio, sample_rate)
                wav_path = archive_wav(
                    self.settings.voice_samples_dir,
                    "manual-" + uuid.uuid4().hex[:12],
                    audio,
                    sample_rate,
                    datetime.now(timezone.utc),
                )
                with Store(self.settings.db_path) as store:
                    vp = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
                    added, added_tier = vp.add_sample(
                        embedding=embedding,
                        audio_path=str(wav_path),
                        duration_ms=len(audio) * 1000 // sample_rate,
                        score=0.95,
                        segment_id=None,
                        is_manual=True,
                    )
                self.writer.emit("manual_sample_status", {
                    "status": "done" if added else "failed",
                    "added": added,
                    "tier": added_tier,
                }, request.request_id)
                with Store(self.settings.db_path) as store:
                    profile = self._profile_status(store)
                profile["auto_learn"] = self._voice_auto_learn
                self.writer.emit("profile_status", profile)
            except Exception as exc:
                LOGGER.exception("manual sample capture failed")
                self.writer.emit("manual_sample_status", {
                    "status": "failed",
                    "error": str(exc),
                }, request.request_id)

        threading.Thread(target=run, name="ev-manual-enroll", daemon=True).start()

    def _start_voice_enrollment(self, request: EngineRequest) -> None:
        if self._enroll_thread and self._enroll_thread.is_alive():
            raise RuntimeError("声纹录入已在进行中")
        if self.state in ("loading", "listening", "speech"):
            self._stop_listening(request)
            self._listen_thread.join(timeout=2.0)
        require_models(self.settings.models, skip_keys=frozenset({"asr_final"}))
        device = resolve_device(self.device_selector)
        self._enroll_device = request.payload.get("device") or self.device_selector
        self._enroll_frames = []
        self._enroll_stop = threading.Event()
        self._enroll_capture = AudioCapture(self.settings.audio, device=device, frame_ms=50)
        self.writer.emit("voice_enroll_status", {"status": "recording", "level": 0.0})

        def run() -> None:
            try:
                capture = self._enroll_capture
                capture.start()
                loop = asyncio.new_event_loop()
                try:
                    async def collect() -> None:
                        async for frame in capture.frames():
                            if self._enroll_stop.is_set():
                                break
                            self._enroll_frames.append(frame)
                            level = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
                            level_db = 20 * float(np.log10(max(level, 1e-8)))
                            self.writer.emit("voice_enroll_status", {
                                "status": "recording",
                                "level": max(0.0, min(1.0, (level_db + 60) / 60)),
                            })
                    loop.run_until_complete(collect())
                finally:
                    loop.close()
                    capture.stop()
            except Exception as exc:
                LOGGER.exception("voice enrollment capture failed")
                self.writer.emit("voice_enroll_status", {
                    "status": "failed",
                    "error": str(exc),
                })

        self._enroll_thread = threading.Thread(target=run, name="ev-voice-enroll", daemon=True)
        self._enroll_thread.start()

    def _stop_voice_enrollment(self, request: EngineRequest) -> None:
        if not self._enroll_thread or not self._enroll_thread.is_alive():
            raise RuntimeError("没有进行中的声纹录入")
        self._enroll_stop.set()
        self.writer.emit("voice_enroll_status", {"status": "processing"})

        def run() -> None:
            try:
                self._enroll_thread.join(timeout=5.0)
                frames = self._enroll_frames
                self._enroll_frames = []
                self._enroll_thread = None
                if not frames:
                    raise RuntimeError("未采集到音频")
                audio = np.concatenate(frames)
                sample_rate = self.settings.audio.sample_rate
                if len(audio) < int(sample_rate * 0.5):
                    raise RuntimeError("录入音频过短（至少需要0.5秒）")
                paths = require_models(self.settings.models, self.settings.models.root, skip_keys=frozenset({"asr_final"}))
                speaker = SpeakerEmbeddingAdapter(str(paths["speaker"]))
                embedding = speaker.embed(audio, sample_rate)
                wav_path = archive_wav(
                    self.settings.voice_samples_dir,
                    "enroll-" + uuid.uuid4().hex[:12],
                    audio,
                    sample_rate,
                    datetime.now(timezone.utc),
                )
                with Store(self.settings.db_path) as store:
                    vp = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
                    added, added_tier = vp.add_sample(
                        embedding=embedding,
                        audio_path=str(wav_path),
                        duration_ms=len(audio) * 1000 // sample_rate,
                        score=0.95,
                        segment_id=None,
                        is_manual=True,
                    )
                self.writer.emit("voice_enroll_status", {
                    "status": "done" if added else "failed",
                    "added": added,
                    "tier": added_tier,
                }, request.request_id)
                with Store(self.settings.db_path) as store:
                    profile = self._profile_status(store)
                profile["auto_learn"] = self._voice_auto_learn
                self.writer.emit("profile_status", profile)
            except Exception as exc:
                LOGGER.exception("voice enrollment processing failed")
                self.writer.emit("voice_enroll_status", {
                    "status": "failed",
                    "error": str(exc),
                }, request.request_id)

        threading.Thread(target=run, name="ev-voice-enroll-process", daemon=True).start()

    def _list_segments(self, request: EngineRequest) -> None:
        payload = request.payload
        with Store(self.settings.db_path) as store:
            segments = store.list_segments(
                limit=int(payload.get("limit", 100)),
                offset=int(payload.get("offset", 0)),
                speaker_label=payload.get("speaker_label") or None,
                query_only=bool(payload.get("query_only", False)),
                date_prefix=payload.get("date") or None,
            )
            queries = store.list_queries(limit=100)
        self.writer.emit("segment_list", {"segments": segments, "queries": queries}, request.request_id)

    def _delete_segment(self, request: EngineRequest) -> None:
        segment_id = str(request.payload.get("segment_id", ""))
        if not segment_id:
            raise ValueError("缺少 segment_id")
        with Store(self.settings.db_path) as store:
            deleted = store.delete_segment(segment_id)
        self.writer.emit("segment_deleted", {"segment_id": segment_id, "deleted": deleted}, request.request_id)
        # 历史删除与声纹解耦：样本保留，segment_id 置空（schema v14 ON DELETE SET NULL）
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _delete_all_segments(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            count = store.delete_all_segments()
        self.writer.emit("segments_deleted", {"count": count}, request.request_id)
        # 同上：清空历史不影响声纹样本
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _delete_quality_rejected_segments(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            count = store.delete_quality_rejected_segments()
        self.writer.emit(
            "segments_deleted",
            {"count": count, "reason": "quality_rejected"},
            request.request_id,
        )
        with Store(self.settings.db_path) as store:
            profile = self._profile_status(store)
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _submit_manual_query(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            query = store.submit_manual_query(str(request.payload.get("text", "")))
        payload = asdict(query)
        self.writer.emit("query_candidate", payload, request.request_id)

    def _delete_query(self, request: EngineRequest) -> None:
        query_id = str(request.payload.get("query_id", ""))
        if not query_id:
            raise ValueError("缺少 query_id")
        with Store(self.settings.db_path) as store:
            deleted = store.delete_query(query_id)
        self.writer.emit("query_deleted", {"query_id": query_id, "deleted": deleted}, request.request_id)

    def _delete_all_queries(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            count = store.delete_all_queries()
        self.writer.emit("queries_deleted", {"count": count}, request.request_id)

    def _broadcast_hotwords(self) -> None:
        """Rebuild hotwords string + entries from store and push to the active segment worker."""
        with Store(self.settings.db_path) as store:
            hotwords = store.get_hotwords_string()
            entries = store.get_hotword_entries()
        word_count = len(hotwords.split()) if hotwords else 0
        if word_count > 0:
            logging.getLogger(__name__).info(
                "Broadcasting %d hotwords: %s", word_count, hotwords[:200]
            )
        if self._segment_worker is not None:
            self._segment_worker.update_hotwords(hotwords, entries)

    def _list_lexicon(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            words = store.list_lexicon()
        self.writer.emit("lexicon_list", {"words": words}, request.request_id)

    def _add_lexicon_word(self, request: EngineRequest) -> None:
        word = str(request.payload.get("word", "")).strip()
        weight = float(request.payload.get("weight", 3.0))
        if not word:
            raise ValueError("词语不能为空")
        with Store(self.settings.db_path) as store:
            entry = store.add_lexicon_word(word, weight, source="manual")
            # Record as implicit correction signal: user manually added a word ASR didn't know
            store.record_correction(
                asr_text="",
                corrected_text=word,
                source="manual_add_word",
            )
        self._broadcast_hotwords()
        self.writer.emit("lexicon_updated", {"added": True, "word": entry}, request.request_id)
        self._ack(request)

    def _update_lexicon_word(self, request: EngineRequest) -> None:
        word_id = str(request.payload.get("id", ""))
        if not word_id:
            raise ValueError("缺少 id")
        word = request.payload.get("word")
        weight = request.payload.get("weight")
        promote = bool(request.payload.get("promote_to_manual", False))
        w = float(weight) if weight is not None else None
        wd = str(word) if word is not None else None
        with Store(self.settings.db_path) as store:
            updated = store.update_lexicon_word(word_id, word=wd, weight=w, promote_to_manual=promote)
        if updated:
            self._broadcast_hotwords()
        self.writer.emit("lexicon_updated", {"updated": updated, "id": word_id}, request.request_id)
        self._ack(request)

    def _delete_lexicon_word(self, request: EngineRequest) -> None:
        word_id = str(request.payload.get("id", ""))
        if not word_id:
            raise ValueError("缺少 id")
        with Store(self.settings.db_path) as store:
            deleted = store.delete_lexicon_word(word_id)
        if deleted:
            self._broadcast_hotwords()
        self.writer.emit("lexicon_updated", {"deleted": deleted, "id": word_id}, request.request_id)
        self._ack(request)

    def _clear_auto_lexicon(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            count = store.clear_auto_words()
        if count > 0:
            self._broadcast_hotwords()
        self.writer.emit("lexicon_updated", {"cleared_auto": count}, request.request_id)
        self._ack(request)

    def _correct_segment(self, request: EngineRequest) -> None:
        segment_id = str(request.payload.get("segment_id", ""))
        corrected_text = str(request.payload.get("corrected_text", "")).strip()
        if not segment_id:
            raise ValueError("缺少 segment_id")
        if not corrected_text:
            raise ValueError("修正文本不能为空")
        learned_words: list[str] = []
        with Store(self.settings.db_path) as store:
            original = store.connection.execute(
                "SELECT transcript_final, speaker_label, speaker_score, audio_path FROM segments WHERE id=?",
                (segment_id,),
            ).fetchone()
            if original is None:
                raise ValueError(f"未找到语音记录: {segment_id}")
            asr_text = original["transcript_final"]
            if asr_text == corrected_text:
                # No change needed
                self.writer.emit("segment_corrected", {"segment_id": segment_id, "changed": False}, request.request_id)
                self._ack(request)
                return
            updated = store.update_segment_transcript(segment_id, corrected_text)
            store.record_correction(
                segment_id=segment_id,
                asr_text=asr_text,
                corrected_text=corrected_text,
                source="manual_edit",
                speaker_label=original["speaker_label"],
                speaker_score=original["speaker_score"],
                audio_path=original["audio_path"],
            )
            learned_words: list[str] = []
        self.writer.emit(
            "segment_corrected",
            {"segment_id": segment_id, "changed": True, "corrected_text": corrected_text, "learned_words": learned_words},
            request.request_id,
        )
        self._ack(request)

    def _list_corrections(self, request: EngineRequest) -> None:
        payload = request.payload
        with Store(self.settings.db_path) as store:
            corrections = store.list_corrections(
                limit=int(payload.get("limit", 100)),
                offset=int(payload.get("offset", 0)),
                source=payload.get("source") or None,
            )
        self.writer.emit("correction_list", {"corrections": corrections}, request.request_id)

    def _learn_corrections(self, request: EngineRequest) -> None:
        """Auto-learning from corrections is disabled. Returns empty result."""
        learned_words: list[str] = []
        self.writer.emit(
            "corrections_learned",
            {"added": 0, "words": learned_words},
            request.request_id,
        )
        self._ack(request)

    def _shutdown_command(self, request: EngineRequest) -> None:
        LOGGER.info("engine shutdown requested")
        self._ack(request)
        self._running = False
        self.shutdown()

    def shutdown(self) -> None:
        self._listen_stop.set()
        self._download_cancel.set()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join()
        if self._env_monitor is not None:
            self._env_monitor.stop()
            self._env_monitor = None
        self.state = "stopped"

    def _emit_state(self, request_id: str | None = None) -> None:
        self.writer.emit(
            "engine_state",
            {
                "state": self.state,
                "device": self.device_selector,
                "data_dir": str(self.settings.data_dir),
                "model_root": str(self.settings.models.root),
            },
            request_id,
        )

    def _ensure_registry_populated(self) -> None:
        """确保注册表已从旧配置迁移/填充。"""
        if not self._registry.list_installed():
            # 尝试从旧配置迁移
            try:
                migrate_from_legacy(
                    self._registry,
                    old_slots={
                        "vad": self.settings.models.vad,
                        "asr_streaming": self.settings.models.asr_streaming,
                        "asr_final": self.settings.models.asr_final,
                        "speaker": self.settings.models.speaker,
                    },
                    old_root=self.settings.models.root,
                )
            except Exception:
                LOGGER.debug("旧配置迁移失败（可能无需迁移）", exc_info=True)

    def _ack(self, request: EngineRequest) -> None:
        self.writer.emit("command_result", {"command": request.command, "ok": True}, request.request_id)
