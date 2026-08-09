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
from ..config import Settings
from ..model_download import DownloadCancelled, ModelDownloader
from ..models import require_models, verify_models
from ..pipeline.runtime import transcribe_forever
from ..speaker.profile import VoiceProfileManager
from ..speaker.verification import build_profile, normalize_embedding
from ..store.audio import archive_wav
from ..store.db import Store
from .protocol import EngineRequest, ProtocolWriter


LOGGER = logging.getLogger(__name__)


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

    def serve(self, input_stream: TextIO = sys.stdin) -> int:
        self.settings.ensure_dirs()
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
            "start_listening": self._start_listening,
            "stop_listening": self._stop_listening,
            "set_thresholds": self._set_thresholds,
            "list_voice_samples": self._list_voice_samples,
            "delete_voice_sample": self._delete_voice_sample,
            "promote_voice_sample": self._promote_voice_sample,
            "reset_voice_profile": self._reset_voice_profile,
            "set_voice_learning": self._set_voice_learning,
            "capture_manual_sample": self._capture_manual_sample,
            "list_segments": self._list_segments,
            "delete_segment": self._delete_segment,
            "delete_all_segments": self._delete_all_segments,
            "submit_manual_query": self._submit_manual_query,
            "delete_query": self._delete_query,
            "delete_all_queries": self._delete_all_queries,
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
            profile = store.profile_status()
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile, request.request_id)

    def _list_devices(self, request: EngineRequest) -> None:
        devices = [asdict(device) for device in list_input_devices()]
        self.writer.emit("device_list", {"devices": devices}, request.request_id)

    def _verify_models(self, request: EngineRequest) -> None:
        checks = verify_models(self.settings.models)
        models = [
            {"key": item.key, "path": str(item.path), "ready": item.ok, "errors": list(item.errors)}
            for item in checks
        ]
        self.writer.emit(
            "model_status",
            {"status": "ready" if all(item.ok for item in checks) else "missing", "models": models},
            request.request_id,
        )

    def _download_models(self, request: EngineRequest) -> None:
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
        self._ack(request)

    def _start_listening(self, request: EngineRequest) -> None:
        if self._listen_thread and self._listen_thread.is_alive():
            raise RuntimeError("监听已经启动")
        require_models(self.settings.models)
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
                profile = store.profile_status()
            profile["auto_learn"] = self._voice_auto_learn
            self.writer.emit("profile_status", profile)
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
            profile = store.profile_status()
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)
        self._ack(request)

    def _delete_voice_sample(self, request: EngineRequest) -> None:
        sample_id = str(request.payload.get("sample_id", ""))
        if not sample_id:
            raise ValueError("缺少 sample_id")
        with Store(self.settings.db_path) as store:
            deleted = store.delete_voice_sample(sample_id)
        self.writer.emit(
            "voice_sample_deleted",
            {"sample_id": sample_id, "deleted": deleted},
            request.request_id,
        )
        with Store(self.settings.db_path) as store:
            profile = store.profile_status()
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _reset_voice_profile(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            count = store.delete_all_voice_samples()
        self.writer.emit("voice_profile_reset", {"deleted": count}, request.request_id)
        with Store(self.settings.db_path) as store:
            profile = store.profile_status()
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _set_voice_learning(self, request: EngineRequest) -> None:
        enabled = bool(request.payload.get("enabled", True))
        self._voice_auto_learn = enabled
        self._ack(request)
        with Store(self.settings.db_path) as store:
            profile = store.profile_status()
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
                paths = require_models(self.settings.models, self.settings.models.root)
                speaker = SpeakerEmbeddingAdapter(str(paths["speaker"]))
                embedding = speaker.embed(audio, sample_rate)
                wav_path = archive_wav(
                    self.settings.archive_dir,
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
                    profile = store.profile_status()
                profile["auto_learn"] = self._voice_auto_learn
                self.writer.emit("profile_status", profile)
            except Exception as exc:
                LOGGER.exception("manual sample capture failed")
                self.writer.emit("manual_sample_status", {
                    "status": "failed",
                    "error": str(exc),
                }, request.request_id)

        threading.Thread(target=run, name="ev-manual-enroll", daemon=True).start()

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
        # Cascade may have deleted voice samples - refresh profile
        with Store(self.settings.db_path) as store:
            profile = store.profile_status()
        profile["auto_learn"] = self._voice_auto_learn
        self.writer.emit("profile_status", profile)

    def _delete_all_segments(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            count = store.delete_all_segments()
        self.writer.emit("segments_deleted", {"count": count}, request.request_id)
        # Deleting all segments removes non-manual voice samples - refresh profile
        with Store(self.settings.db_path) as store:
            profile = store.profile_status()
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

    def _ack(self, request: EngineRequest) -> None:
        self.writer.emit("command_result", {"command": request.command, "ok": True}, request.request_id)
