"""EV GUI engine 命令调度、生命周期与后台任务。"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from dataclasses import asdict
from dataclasses import replace
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
from ..speaker.verification import build_profile, normalize_embedding
from ..store.db import Store
from .protocol import EngineRequest, ProtocolWriter


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
        self._enrollment_lock = threading.Lock()
        self._enrollment_embeddings: list[np.ndarray] = []
        self._enrollment_expected = 0
        self._enrollment_device: str | None = None
        self._enrollment_adapter: SpeakerEmbeddingAdapter | None = None

    def serve(self, input_stream: TextIO = sys.stdin) -> int:
        self.settings.ensure_dirs()
        self._emit_state()
        for line in input_stream:
            if not self._running:
                break
            if not line.strip():
                continue
            try:
                request = EngineRequest.parse(line)
                self.handle(request)
            except Exception as exc:
                self.writer.error(str(exc), code="invalid_request")
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
            "begin_enrollment": self._begin_enrollment,
            "capture_enrollment_sample": self._capture_enrollment_sample,
            "cancel_enrollment": self._cancel_enrollment,
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
        self.device_selector = request.payload.get("device") or None
        user_threshold = float(
            request.payload.get("user_threshold", self.settings.speaker.user_threshold)
        )
        non_user_threshold = float(
            request.payload.get("non_user_threshold", self.settings.speaker.non_user_threshold)
        )
        if non_user_threshold >= user_threshold:
            raise ValueError("非用户阈值必须小于用户阈值")
        session_settings = replace(
            self.settings,
            speaker=replace(
                self.settings.speaker,
                user_threshold=user_threshold,
                non_user_threshold=non_user_threshold,
            ),
        )
        device = resolve_device(self.device_selector)
        self._listen_stop = threading.Event()
        self.state = "loading"
        self._emit_state(request.request_id)

        def run() -> None:
            try:
                self.state = "listening"
                self._emit_state()
                asyncio.run(
                    transcribe_forever(
                        session_settings,
                        device,
                        output=lambda message: None,
                        stop_event=self._listen_stop,
                        emit=self._on_runtime_event,
                    )
                )
                self.state = "stopped"
                self._emit_state()
            except Exception as exc:
                self.state = "error"
                self.writer.error(str(exc), code="listening_failed")
                self._emit_state()

        self._listen_thread = threading.Thread(target=run, name="ev-listening", daemon=True)
        self._listen_thread.start()

    def _stop_listening(self, request: EngineRequest) -> None:
        self._listen_stop.set()
        if self.state not in ("stopped", "error"):
            self.state = "stopping"
        self._emit_state(request.request_id)

    def _on_runtime_event(self, event_type: str, payload: dict) -> None:
        if event_type == "speech_started":
            self.state = "speech"
            self._emit_state()
        elif event_type == "speech_ended":
            self.state = "listening"
            self._emit_state()
        self.writer.emit(event_type, payload)

    def _begin_enrollment(self, request: EngineRequest) -> None:
        expected = int(request.payload.get("segments", 8))
        if expected < 1 or expected > 12:
            raise ValueError("segments 必须在 1 到 12 之间")
        require_models(self.settings.models)
        self._enrollment_embeddings = []
        self._enrollment_expected = expected
        self._enrollment_device = request.payload.get("device") or None
        self._enrollment_adapter = None
        self.writer.emit(
            "enrollment_progress",
            {"status": "ready", "completed": 0, "total": expected},
            request.request_id,
        )

    def _capture_enrollment_sample(self, request: EngineRequest) -> None:
        if not self._enrollment_expected:
            raise RuntimeError("请先开始声纹录入")
        if self._enrollment_lock.locked():
            raise RuntimeError("当前录音尚未结束")

        def run() -> None:
            with self._enrollment_lock:
                try:
                    self.writer.emit(
                        "enrollment_progress",
                        {
                            "status": "recording",
                            "completed": len(self._enrollment_embeddings),
                            "total": self._enrollment_expected,
                        },
                        request.request_id,
                    )
                    audio = asyncio.run(self._record_sample(4.0))
                    if self._enrollment_adapter is None:
                        paths = require_models(self.settings.models)
                        self._enrollment_adapter = SpeakerEmbeddingAdapter(str(paths["speaker"]))
                    embedding = normalize_embedding(
                        self._enrollment_adapter.embed(audio, self.settings.audio.sample_rate)
                    )
                    self._enrollment_embeddings.append(embedding)
                    completed = len(self._enrollment_embeddings)
                    if completed >= self._enrollment_expected:
                        profile = build_profile(self._enrollment_embeddings)
                        with Store(self.settings.db_path) as store:
                            store.save_profile(
                                "user-v1",
                                "user",
                                self._enrollment_device,
                                self.settings.models.speaker,
                                profile,
                                completed,
                            )
                        status = "complete"
                        self._enrollment_expected = 0
                    else:
                        status = "sample_complete"
                    self.writer.emit(
                        "enrollment_progress",
                        {"status": status, "completed": completed, "total": completed if status == "complete" else self._enrollment_expected},
                        request.request_id,
                    )
                except Exception as exc:
                    self.writer.error(str(exc), request.request_id, "enrollment_failed")

        threading.Thread(target=run, name="ev-enrollment", daemon=True).start()
        self._ack(request)

    async def _record_sample(self, seconds: float) -> np.ndarray:
        capture = AudioCapture(
            self.settings.audio, device=resolve_device(self._enrollment_device)
        )
        chunks: list[np.ndarray] = []
        target = int(self.settings.audio.sample_rate * seconds)
        capture.start()
        try:
            async for frame in capture.frames():
                chunks.append(frame)
                if sum(item.size for item in chunks) >= target:
                    break
        finally:
            capture.stop()
        return np.concatenate(chunks)[:target]

    def _cancel_enrollment(self, request: EngineRequest) -> None:
        if self._enrollment_lock.locked():
            raise RuntimeError("录音进行中，请等待本段完成")
        self._enrollment_embeddings = []
        self._enrollment_expected = 0
        self._enrollment_adapter = None
        self.writer.emit("enrollment_progress", {"status": "cancelled", "completed": 0, "total": 0}, request.request_id)

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

    def _delete_all_segments(self, request: EngineRequest) -> None:
        with Store(self.settings.db_path) as store:
            count = store.delete_all_segments()
        self.writer.emit("segments_deleted", {"count": count}, request.request_id)

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
        self._ack(request)
        self._running = False
        self.shutdown()

    def shutdown(self) -> None:
        self._listen_stop.set()
        self._download_cancel.set()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=8)
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
