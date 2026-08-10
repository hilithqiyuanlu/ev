"""Phase 1a/1b.2 实时运行时。"""

from __future__ import annotations

import asyncio
import queue
import re
import signal
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..asr.adapters import FinalASRAdapter, SpeakerEmbeddingAdapter, StreamingASRAdapter
from ..audio.capture import AudioCapture
from ..audio.preprocess import AudioPreprocessor, PreprocessParams
from ..audio.energy_vad import EnergyVAD, EnergyVADParams
from ..config import Settings
from ..speaker.profile import VoiceProfileManager
from ..speaker.verification import (
    cosine_score,
    normalize_loudness,
    verify_speaker,
)
from ..store.audio import archive_wav
from ..store.db import SegmentRecord, Store
from ..vui import decide_query, match_wake_prefix
from ..vad.adapters import VADAdapter, CompositeVAD

_FILLER_WORDS = frozenset({
    "嗯", "啊", "呃", "哦", "诶", "唉", "哈", "喂", "哎", "噢",
    "那个", "这个", "就是", "然后", "所以", "嗯啊", "啊对",
})
_PUNCT_RE = re.compile(r"[\s，。！？、；：""''（）【】…—\[\]().,!?;:+-]+")


def _is_filler_only(text: str) -> bool:
    cleaned = _PUNCT_RE.sub("", text or "").strip()
    if not cleaned:
        return True
    # Check if all remaining tokens are filler words
    i = 0
    while i < len(cleaned):
        matched = False
        for w in sorted(_FILLER_WORDS, key=len, reverse=True):
            if cleaned[i:].startswith(w):
                i += len(w)
                matched = True
                break
        if not matched:
            return False
    return True


class SegmentProcessor:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        stream_asr: StreamingASRAdapter,
        speaker: SpeakerEmbeddingAdapter,
        vad_model_id: str,
        voice_profile: VoiceProfileManager,
        final_asr: FinalASRAdapter | None = None,
        output: Callable[[str], None] = print,
        emit: Callable[[str, dict], None] | None = None,
        hotwords: str = "",
    ):
        self.settings = settings
        self.store = store
        self.stream_asr = stream_asr
        self.speaker = speaker
        self.vad_model_id = vad_model_id
        self.voice_profile = voice_profile
        self.centroids = voice_profile.centroids
        self.final_asr = final_asr
        self.output = output
        self.emit = emit or (lambda *_: None)
        # Mutable threshold - can be updated at runtime without restart
        self.threshold: float = settings.speaker.threshold
        # Hotwords for final ASR - can be updated at runtime
        self.hotwords: str = hotwords

    def update_thresholds(self, threshold: float | None = None) -> None:
        if threshold is not None:
            self.threshold = float(threshold)

    def update_hotwords(self, hotwords: str) -> None:
        self.hotwords = hotwords or ""

    def process(
        self,
        audio: np.ndarray,
        started_at: datetime,
        ended_at: datetime,
        partial: str = "",
        segment_id: str | None = None,
    ) -> SegmentRecord | None:
        segment_id = segment_id or uuid.uuid4().hex
        duration_ms = round(len(audio) * 1000 / self.settings.audio.sample_rate)

        # Loudness normalize before embedding for robustness to mic distance/volume
        audio_for_embedding = normalize_loudness(audio) if self.settings.speaker.loudness_normalize else audio

        final = self._final(audio)
        transcript = final or partial
        # Empty or filler-only segments are discarded (no WAV, no DB, no speaker check).
        if not transcript.strip():
            self.output(f"[{segment_id[:8]}] discarded: empty transcript ({duration_ms}ms)")
            return None
        if self.settings.segment.discard_filler_only and _is_filler_only(transcript):
            self.output(f"[{segment_id[:8]}] discarded: filler-only {transcript!r} ({duration_ms}ms)")
            return None
        embedding = self.speaker.embed(audio_for_embedding, self.settings.audio.sample_rate)
        wav_path = archive_wav(
            self.settings.archive_dir, segment_id, audio,
            self.settings.audio.sample_rate, started_at,
        )
        profile_ready = self.voice_profile.state.is_ready
        if not self.centroids or not profile_ready:
            # Cold start (profile building): treat all as user for learning
            speaker_label, score = "user", None
            effective_for_decision = "user"
        else:
            result = verify_speaker(
                embedding,
                self.centroids,
                self.threshold,
            )
            score = result.score
            speaker_label = result.label
            effective_for_decision = speaker_label

        decision = decide_query(
            transcript,
            effective_for_decision,
            self.settings.vui.wake_words,
            profile_ready=profile_ready,
        )

        record = SegmentRecord(
            id=segment_id,
            started_at=started_at.isoformat(), ended_at=ended_at.isoformat(),
            duration_ms=duration_ms,
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
        is_filler = self.settings.segment.discard_filler_only and _is_filler_only(transcript)

        # Sample collection: collect only from user (score >= threshold), never from non-user.
        # Cold start (profile not ready): treat all as collectible with high score.
        should_try_collect = False
        collect_score = 0.8 if not profile_ready else (score if score is not None else 0.0)
        if not profile_ready:
            should_try_collect = True
            collect_score = 0.8
        elif speaker_label == "user" and score is not None:
            should_try_collect = score >= self.settings.voice_learning.collect_min_score

        if should_try_collect and self.voice_profile.should_collect(
            duration_ms=duration_ms,
            score=collect_score,
            transcript=transcript,
            is_filler_only=is_filler,
        ):
            added, added_tier = self.voice_profile.add_sample(
                embedding=embedding,
                audio_path=str(wav_path),
                duration_ms=duration_ms,
                score=collect_score,
                segment_id=segment_id,
            )
            if added:
                self.centroids = self.voice_profile.centroids
                state = self.voice_profile.state
                self.output(f"[{segment_id[:8]}] voice sample added tier={added_tier} (core={state.core_count}, cache={state.cache_count}, centroids={state.centroid_count})")
                self.emit(
                    "voice_sample_added",
                    {
                        "segment_id": segment_id,
                        "tier": added_tier,
                        "core_count": state.core_count,
                        "cache_count": state.cache_count,
                        "sample_count": state.sample_count,
                        "centroid_count": state.centroid_count,
                        "is_ready": state.is_ready,
                    },
                )
        if not profile_ready and self.voice_profile.state.is_ready:
            state = self.voice_profile.state
            self.emit(
                "voice_profile_ready",
                {"sample_count": state.sample_count, "core_count": state.core_count},
            )
        self.output(
            f"[{segment_id[:8]}] final={final!r} speaker={speaker_label} "
            f"score={score if score is not None else '-'} query={decision.query_candidate}"
        )
        return record

    def _final(self, audio: np.ndarray) -> str:
        if self.final_asr is None:
            return ""
        return self.final_asr.transcribe(
            audio,
            self.settings.audio.sample_rate,
            hotword=self.hotwords,
        )


@dataclass(frozen=True)
class SegmentJob:
    audio: np.ndarray
    started_at: datetime
    ended_at: datetime
    partial: str
    segment_id: str


class SegmentWorker:
    """串行完成终稿、声纹与持久化，避免阻塞实时采集。"""

    def __init__(
        self,
        settings: Settings,
        paths: dict[str, Path],
        stream: StreamingASRAdapter,
        speaker: SpeakerEmbeddingAdapter,
        output: Callable[[str], None],
        emit: Callable[[str, dict], None],
        shared_threshold: dict | None = None,
    ):
        self.settings = settings
        self.paths = paths
        self.stream = stream
        self.speaker = speaker
        self.output = output
        self.emit = emit
        self._processor: SegmentProcessor | None = None
        self._shared_threshold = shared_threshold  # mutable shared state: {"threshold": float}
        self.jobs: queue.Queue[SegmentJob | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="ev-segment-worker", daemon=True)
        self.thread.start()

    def update_thresholds(self, threshold: float | None = None) -> None:
        if self._processor is not None:
            self._processor.update_thresholds(threshold)
        if self._shared_threshold is not None and threshold is not None:
            self._shared_threshold["threshold"] = float(threshold)

    def update_hotwords(self, hotwords: str) -> None:
        if self._processor is not None:
            self._processor.update_hotwords(hotwords)

    def submit(self, job: SegmentJob) -> None:
        self.jobs.put(job)

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join()

    def _run(self) -> None:
        final: FinalASRAdapter | None = None
        with Store(self.settings.db_path) as store:
            from ..speaker.profile import VoiceProfileManager
            voice_profile = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
            # Initial hotword load and one-time correction learning on startup
            try:
                store.learn_from_corrections()
            except Exception:
                pass
            hotwords_str = store.get_hotwords_string()
            processor = SegmentProcessor(
                self.settings,
                store,
                self.stream,
                self.speaker,
                self.settings.models.vad,
                voice_profile,
                final,
                self.output,
                self.emit,
                hotwords=hotwords_str,
            )
            self._processor = processor
            # Sync initial threshold to shared state
            if self._shared_threshold is not None:
                self._shared_threshold["threshold"] = processor.threshold
            while True:
                job = self.jobs.get()
                if job is None:
                    self.jobs.task_done()
                    return
                try:
                    self.emit(
                        "segment_processing",
                        {
                            "segment_id": job.segment_id,
                            "phase": "finalizing",
                            "queue_depth": self.jobs.qsize(),
                        },
                    )
                    if final is None:
                        final = FinalASRAdapter(str(self.paths["asr_final"]))
                        processor.final_asr = final
                    duration_ms = round(len(job.audio) * 1000 / self.settings.audio.sample_rate)
                    record = processor.process(
                        job.audio,
                        job.started_at,
                        job.ended_at,
                        job.partial,
                        job.segment_id,
                    )
                    if record is None:
                        self.emit(
                            "segment_discarded",
                            {
                                "segment_id": job.segment_id,
                                "reason": "empty_or_filler",
                                "duration_ms": duration_ms,
                            },
                        )
                    else:
                        _emit_record(self.emit, record)
                except Exception as exc:
                    self.emit(
                        "segment_failed",
                        {
                            "segment_id": job.segment_id,
                            "code": "segment_processing_failed",
                            "message": str(exc),
                        },
                    )
                finally:
                    self.jobs.task_done()


async def transcribe_forever(
    settings: Settings,
    device: int | None = None,
    model_root: Path | None = None,
    output: Callable[[str], None] = print,
    stop_event: threading.Event | None = None,
    emit: Callable[[str, dict], None] | None = None,
    worker_holder: dict | None = None,
) -> None:
    from ..models import require_models

    paths = require_models(settings.models, model_root)
    settings.ensure_dirs()

    def send(event_type: str, payload: dict) -> None:
        if emit is not None:
            emit(event_type, payload)

    # --- 预处理 & VAD 初始化 ---
    sr = settings.audio.sample_rate
    frame_ms_default = 30
    # 1) 预处理管线 (DC → preemphasis → AGC → NoiseGate)
    preprocessor: AudioPreprocessor | None = None
    if settings.preprocess.enabled:
        pp_params = PreprocessParams(
            preemphasis_coeff=settings.preprocess.preemphasis_coeff,
            agc_target_rms=settings.preprocess.agc_target_rms,
            agc_min_gain=settings.preprocess.agc_min_gain,
            agc_max_gain=settings.preprocess.agc_max_gain,
            agc_attack_ms=settings.preprocess.agc_attack_ms,
            agc_release_ms=settings.preprocess.agc_release_ms,
            noisegate_enabled=settings.preprocess.noisegate_enabled,
            noisegate_snr_db=settings.preprocess.noisegate_snr_db,
            noisegate_floor_track_sec=settings.preprocess.noisegate_floor_track_sec,
        )
        preprocessor = AudioPreprocessor(
            sample_rate=sr, frame_ms=frame_ms_default, params=pp_params
        )
        output(
            f"[preprocess] enabled target_rms={pp_params.agc_target_rms} "
            f"gain=[{pp_params.agc_min_gain:.2f}x..{pp_params.agc_max_gain:.1f}x] "
            f"ng={pp_params.noisegate_enabled} snr={pp_params.noisegate_snr_db}dB"
        )
    # 2) VAD: FSMN + (可选) EnergyVAD, 组合 start=OR/end=AND
    vad_model = VADAdapter(
        str(paths["vad"]),
        threshold=settings.vad.fsmn_threshold,
    )
    energy_vad: EnergyVAD | None = None
    if settings.vad.energy_vad_enabled:
        ev_params = EnergyVADParams(
            snr_threshold_linear=settings.vad.energy_snr_linear,
            abs_min_rms=settings.vad.energy_abs_min_rms,
            start_frames=settings.vad.energy_start_frames,
            hangover_frames=settings.vad.energy_hangover_frames,
        )
        energy_vad = EnergyVAD(
            sample_rate=sr, frame_ms=frame_ms_default, params=ev_params
        )
        output(
            f"[vad] energy enabled snr={ev_params.snr_threshold_linear:.1f}x "
            f"start={ev_params.start_frames} hangover={ev_params.hangover_frames}"
        )
    vad = CompositeVAD(
        fsmn_vad=vad_model,
        energy_vad=energy_vad,
        start_mode=settings.vad.combine_start_mode,
        end_mode=settings.vad.combine_end_mode,
        sample_rate=sr,
        frame_ms=frame_ms_default,
    )
    output(
        f"[vad] composite start={settings.vad.combine_start_mode} "
        f"end={settings.vad.combine_end_mode} "
        f"fsmn_th={settings.vad.fsmn_threshold}"
    )

    stream = StreamingASRAdapter(str(paths["asr_streaming"]))
    speaker = SpeakerEmbeddingAdapter(str(paths["speaker"]))
    capture = AudioCapture(settings.audio, device=device, preprocessor=preprocessor)

    # Shared mutable threshold for cross-thread access
    shared_threshold = {
        "threshold": settings.speaker.threshold,
    }
    worker = SegmentWorker(settings, paths, stream, speaker, output, send, shared_threshold)
    if worker_holder is not None:
        worker_holder["worker"] = worker

    # Speaker switch detection parameters
    SPEAKER_CHECK_MIN_DURATION_MS = 1500   # start checking after 1.5s of speech
    SPEAKER_CHECK_INTERVAL_MS = 1000       # check every 1s after that
    SPEAKER_CHECK_WINDOW_MS = 1500         # use latest 1.5s audio for embedding
    SPEAKER_LOW_CONFIRM_COUNT = 2          # require 2 consecutive low scores

    def load_centroids():
        with Store(settings.db_path) as s:
            vp = VoiceProfileManager(s, settings.voice_learning, settings.speaker)
            return vp.centroids if vp.state.is_ready else []

    profile_centroids = load_centroids()

    try:
        pre_roll_frames = max(1, 600 // 30)
        frames: list[np.ndarray] = []
        recent_frames: list[np.ndarray] = []
        started_at: datetime | None = None
        partial = ""
        local_stop = asyncio.Event()
        current_segment_id: str | None = None
        active = False
        last_speaker_check_samples = 0
        speaker_low_streak = 0

        def force_segment_end(trigger_reason: str) -> None:
            nonlocal active, frames, started_at, partial, current_segment_id
            nonlocal last_speaker_check_samples, speaker_low_streak, profile_centroids
            if not active or started_at is None or current_segment_id is None:
                return
            ended_at = datetime.now(timezone.utc)
            final_partial = stream.accept(
                np.empty(0, dtype=np.float32),
                settings.audio.sample_rate,
                is_final=True,
            )
            send("speech_ended", {
                "segment_id": current_segment_id,
                "ended_at": ended_at.isoformat(),
                "trigger": trigger_reason,
            })
            seg_audio = np.concatenate(frames)
            seg_duration_ms = round(len(seg_audio) * 1000 / settings.audio.sample_rate)
            if seg_duration_ms >= settings.segment.min_duration_ms:
                worker.submit(
                    SegmentJob(seg_audio, started_at, ended_at, final_partial, current_segment_id)
                )
            else:
                output(f"[{current_segment_id[:8]}] discarded (too_short, {seg_duration_ms}ms, trigger={trigger_reason})")
                send("segment_discarded", {
                    "segment_id": current_segment_id,
                    "reason": "too_short",
                    "trigger": trigger_reason,
                    "duration_ms": seg_duration_ms,
                })
            active = False
            frames, started_at, partial, current_segment_id = [], None, "", None
            last_speaker_check_samples = 0
            speaker_low_streak = 0
            vad.reset()
            stream.reset()
            # Reload centroids after segment ends (profile may have been updated)
            profile_centroids = load_centroids()

        def request_stop(*_args) -> None:
            local_stop.set()

        can_install_signal = threading.current_thread() is threading.main_thread()
        previous = signal.getsignal(signal.SIGINT) if can_install_signal else None
        if can_install_signal:
            signal.signal(signal.SIGINT, request_stop)
        capture.start()
        send(
            "capture_started",
            {
                "device": device,
                "sample_rate": settings.audio.sample_rate,
                "channels": settings.audio.channels,
            },
        )
        try:
            iterator = capture.frames().__aiter__()
            try:
                first_frame = await asyncio.wait_for(iterator.__anext__(), timeout=2.0)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("麦克风已打开，但两秒内没有收到音频帧") from exc

            async def handle_frame(frame: np.ndarray) -> None:
                nonlocal active, frames, recent_frames, started_at, partial, current_segment_id
                nonlocal last_speaker_check_samples, speaker_low_streak
                rms = float(np.sqrt(np.mean(frame**2)))
                send("audio_level", {"rms": rms})
                recent_frames.append(frame)
                if len(recent_frames) > pre_roll_frames:
                    recent_frames.pop(0)
                if active:
                    frames.append(frame)
                    candidate = stream.accept(frame, settings.audio.sample_rate)
                    if candidate and candidate != partial:
                        partial = candidate
                        output(f"partial: {partial}")
                        send("transcript_partial", {"segment_id": current_segment_id, "text": partial})

                    # Speaker switch detection (only when profile is ready)
                    if profile_centroids:
                        total_samples = sum(f.size for f in frames)
                        min_samples = SPEAKER_CHECK_MIN_DURATION_MS * settings.audio.sample_rate // 1000
                        interval_samples = SPEAKER_CHECK_INTERVAL_MS * settings.audio.sample_rate // 1000
                        if total_samples >= min_samples and (total_samples - last_speaker_check_samples) >= interval_samples:
                            last_speaker_check_samples = total_samples
                            window_samples = SPEAKER_CHECK_WINDOW_MS * settings.audio.sample_rate // 1000
                            # Build window audio (most recent N samples)
                            window_buf = np.empty(0, dtype=np.float32)
                            acc = 0
                            for f in reversed(frames):
                                if acc >= window_samples:
                                    break
                                window_buf = np.concatenate([f, window_buf])
                                acc += f.size
                            try:
                                # Apply loudness normalization for consistent scoring
                                if settings.speaker.loudness_normalize:
                                    window_buf = normalize_loudness(window_buf)
                                emb = speaker.embed(window_buf, settings.audio.sample_rate)
                                # Score against all centroids, take best (centroids already normalized)
                                from .verification import normalize_embedding as _norm
                                norm_emb = _norm(emb)
                                best_score = max(float(np.dot(norm_emb, _norm(c))) for c in profile_centroids)
                                score = best_score
                                if score < shared_threshold["threshold"]:
                                    speaker_low_streak += 1
                                else:
                                    speaker_low_streak = 0
                                if speaker_low_streak >= SPEAKER_LOW_CONFIRM_COUNT:
                                    output(f"speaker switch detected (score={score:.3f}, streak={speaker_low_streak})")
                                    send("speaker_switch_detected", {
                                        "segment_id": current_segment_id,
                                        "score": score,
                                    })
                                    force_segment_end("speaker_switch")
                                    return
                            except Exception as exc:
                                output(f"speaker check failed: {exc}")

                for boundary in vad.accept(frame, settings.audio.sample_rate):
                    if boundary.started and not active:
                        active = True
                        frames = list(recent_frames)
                        duration = sum(item.size for item in frames) / settings.audio.sample_rate
                        started_at = datetime.now(timezone.utc) - timedelta(seconds=duration)
                        current_segment_id = uuid.uuid4().hex
                        last_speaker_check_samples = 0
                        speaker_low_streak = 0
                        stream.reset()
                        partial = stream.accept(
                            np.concatenate(frames), settings.audio.sample_rate
                        )
                        send(
                            "speech_started",
                            {"segment_id": current_segment_id, "started_at": started_at.isoformat()},
                        )
                        if partial:
                            send("transcript_partial", {"segment_id": current_segment_id, "text": partial})
                    if boundary.ended and active and started_at is not None and current_segment_id:
                        force_segment_end("vad_endpoint")
                        return

            await handle_frame(first_frame)
            async for frame in iterator:
                await handle_frame(frame)
                if local_stop.is_set() or (stop_event is not None and stop_event.is_set()):
                    break
        finally:
            final_boundaries = vad.accept(
                np.empty(0, dtype=np.float32), settings.audio.sample_rate, is_final=True
            )
            if any(item.started for item in final_boundaries) and not active:
                active = True
                frames = list(recent_frames)
                duration = sum(item.size for item in frames) / settings.audio.sample_rate
                started_at = datetime.now(timezone.utc) - timedelta(seconds=duration)
                current_segment_id = uuid.uuid4().hex
                stream.reset()
                partial = stream.accept(np.concatenate(frames), settings.audio.sample_rate)
                send("speech_started", {"segment_id": current_segment_id, "started_at": started_at.isoformat()})
            if active and started_at is not None and frames and current_segment_id:
                ended_at = datetime.now(timezone.utc)
                partial = stream.accept(
                    np.empty(0, dtype=np.float32), settings.audio.sample_rate, is_final=True
                )
                send("speech_ended", {"segment_id": current_segment_id, "ended_at": ended_at.isoformat()})
                seg_audio = np.concatenate(frames)
                seg_duration_ms = round(len(seg_audio) * 1000 / settings.audio.sample_rate)
                if seg_duration_ms < settings.segment.min_duration_ms:
                    output(f"[{current_segment_id[:8]}] discarded: too short on stop ({seg_duration_ms}ms)")
                    send(
                        "segment_discarded",
                        {
                            "segment_id": current_segment_id,
                            "reason": "too_short",
                            "duration_ms": seg_duration_ms,
                        },
                    )
                else:
                    worker.submit(
                        SegmentJob(seg_audio, started_at, ended_at, partial, current_segment_id)
                    )
            capture.stop()
            if can_install_signal and previous is not None:
                signal.signal(signal.SIGINT, previous)
    finally:
        worker.close()


def _emit_record(send: Callable[[str, dict], None], record: SegmentRecord) -> None:
    send(
        "speaker_result",
        {
            "segment_id": record.id,
            "label": record.speaker_label,
            "score": record.speaker_score,
        },
    )
    payload = asdict(record)
    send("segment_committed", payload)
    if record.query_candidate and record.query_text:
        send(
            "query_candidate",
            {"segment_id": record.id, "source": "voice", "text": record.query_text},
        )
