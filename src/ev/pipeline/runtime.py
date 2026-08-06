"""Phase 1a 实时运行时。"""

from __future__ import annotations

import asyncio
import signal
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..asr.adapters import FinalASRAdapter, SpeakerEmbeddingAdapter, StreamingASRAdapter
from ..audio.capture import AudioCapture
from ..config import Settings
from ..speaker.verification import build_profile, verify_speaker
from ..store.audio import archive_wav
from ..store.db import SegmentRecord, Store
from ..vui import decide_query
from ..vad.adapters import EndpointState, VADAdapter


class SegmentProcessor:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        stream_asr: StreamingASRAdapter,
        speaker: SpeakerEmbeddingAdapter,
        vad_model_id: str,
        profile: np.ndarray | None = None,
        final_asr: FinalASRAdapter | None = None,
        output: Callable[[str], None] = print,
    ):
        self.settings = settings
        self.store = store
        self.stream_asr = stream_asr
        self.speaker = speaker
        self.vad_model_id = vad_model_id
        self.profile = profile
        self.final_asr = final_asr
        self.output = output

    def process(
        self,
        audio: np.ndarray,
        started_at: datetime,
        ended_at: datetime,
        partial: str = "",
    ) -> SegmentRecord:
        segment_id = uuid.uuid4().hex
        final = self._final(audio)
        embedding = self.speaker.embed(audio, self.settings.audio.sample_rate)
        if self.profile is None:
            speaker_label, score = "unknown", None
        else:
            result = verify_speaker(
                embedding,
                self.profile,
                self.settings.speaker.user_threshold,
                self.settings.speaker.non_user_threshold,
            )
            speaker_label, score = result.label, result.score
        decision = decide_query(final or partial, speaker_label, self.settings.vui.wake_words)
        wav_path = archive_wav(
            self.settings.archive_dir, segment_id, audio,
            self.settings.audio.sample_rate, started_at,
        )
        record = SegmentRecord(
            id=segment_id,
            started_at=started_at.isoformat(), ended_at=ended_at.isoformat(),
            duration_ms=round(len(audio) * 1000 / self.settings.audio.sample_rate),
            audio_path=str(wav_path), sample_rate=self.settings.audio.sample_rate,
            channels=self.settings.audio.channels, transcript_raw=partial,
            transcript_final=final, speaker_label=speaker_label, speaker_score=score,
            wake_detected=decision.wake_detected, query_candidate=decision.query_candidate,
            query_text=decision.query_text, vad_model=self.vad_model_id,
            asr_stream_model=self.settings.models.asr_streaming,
            asr_final_model=self.settings.models.asr_final if self.final_asr else "unavailable",
            speaker_model=self.settings.models.speaker,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.store.insert_segment(record)
        self.output(
            f"[{segment_id[:8]}] final={final!r} speaker={speaker_label} "
            f"score={score if score is not None else '-'} query={decision.query_candidate}"
        )
        return record

    def _final(self, audio: np.ndarray) -> str:
        if self.final_asr is None:
            return ""
        return self.final_asr.transcribe(audio, self.settings.audio.sample_rate)


async def transcribe_forever(
    settings: Settings,
    device: int | None = None,
    model_root: Path | None = None,
    output: Callable[[str], None] = print,
) -> None:
    from ..models import require_models

    paths = require_models(settings.models, model_root)
    settings.ensure_dirs()
    vad = VADAdapter(str(paths["vad"]))
    stream = StreamingASRAdapter(str(paths["asr_streaming"]))
    speaker = SpeakerEmbeddingAdapter(str(paths["speaker"]))
    final: FinalASRAdapter | None = None
    with Store(settings.db_path) as store:
        profile = store.load_profile()
        processor = SegmentProcessor(settings, store, stream, speaker, settings.models.vad, profile, final, output)
        capture = AudioCapture(settings.audio, device=device)
        endpoints = EndpointState(pre_roll_frames=3, hangover_frames=10)
        frames: list[np.ndarray] = []
        recent_frames: list[np.ndarray] = []
        started_at: datetime | None = None
        partial = ""
        stop_requested = asyncio.Event()

        def request_stop(*_args) -> None:
            stop_requested.set()

        previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_stop)
        capture.start()
        try:
            async for frame in capture.frames():
                recent_frames.append(frame)
                if len(recent_frames) > endpoints.pre_roll_frames + 1:
                    recent_frames.pop(0)
                speech = vad.is_speech(frame, settings.audio.sample_rate)
                state = endpoints.update(speech)
                if state.started:
                    # 将端点前的短暂声音一并交给 ASR，避免首字丢失。
                    frames = recent_frames[:-1][-endpoints.pre_roll_frames :]
                    frames.append(frame)
                    started_at = datetime.now(timezone.utc)
                    vad.reset()
                    stream.reset()
                    partial = ""
                    for preroll in frames:
                        candidate = stream.accept(preroll, settings.audio.sample_rate)
                        if candidate:
                            partial = candidate
                            output(f"partial: {partial}")
                if (endpoints.active or state.ended) and not state.started:
                    frames.append(frame)
                if endpoints.active and not state.ended and not state.started:
                    candidate = stream.accept(frame, settings.audio.sample_rate)
                    if candidate:
                        partial = candidate
                        output(f"partial: {partial}")
                if state.ended and started_at is not None:
                    if final is None:
                        final = FinalASRAdapter(str(paths["asr_final"]))
                        processor.final_asr = final
                    processor.process(np.concatenate(frames), started_at, datetime.now(timezone.utc), partial)
                    frames, started_at, partial = [], None, ""
                if stop_requested.is_set():
                    break
        finally:
            if endpoints.active and started_at is not None and frames:
                if final is None:
                    final = FinalASRAdapter(str(paths["asr_final"]))
                    processor.final_asr = final
                processor.process(np.concatenate(frames), started_at, datetime.now(timezone.utc), partial)
            capture.stop()
            signal.signal(signal.SIGINT, previous)
