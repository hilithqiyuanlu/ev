"""Phase 1a/1b.2 实时运行时 — 三态状态机（第二人耳模式）。"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import queue
import re
import signal
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..asr.adapters import (
    FinalASRAdapter,
    SpeakerEmbeddingAdapter,
    StreamingASRAdapter,
    TranscriptionResult,
    TranscriptionSegment,
    _find_model_root,
)
from ..asr.qwen3_adapter import Qwen3ASRAdapter
from ..audio.capture import AudioCapture
from ..audio.preprocess import AudioPreprocessor, PreprocessParams
from ..audio.energy_vad import EnergyVAD, EnergyVADParams
from ..config import Settings
from ..speaker.profile import VoiceProfileManager
from ..speaker.verification import (
    cosine_score,
    normalize_embedding,
    normalize_loudness,
    verify_speaker,
)
from ..store.audio import archive_wav
from ..store.db import SegmentRecord, Store
from ..vui import decide_query, decide_query_from_utterances, match_wake_prefix
from ..vad.adapters import VADAdapter, CompositeVAD

_FILLER_WORDS = frozenset({
    "嗯", "啊", "呃", "哦", "诶", "唉", "哈", "喂", "哎", "噢",
    "那个", "这个", "就是", "然后", "所以", "嗯啊", "啊对",
})
_PUNCT_RE = re.compile(r"[][\s，。！？、；：""''（）【】…—().,!?;:+=~`@#$%^&*|\\/<>-]+")


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


_PUNCT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*")

_OBSERVING_GATE_MS = 900
_SPEAKER_CHECK_INTERVAL_MS = 600
# Hysteresis for speaker switches: harder to lose "user" label than gain it
_SWITCH_CONFIRM_NONUSER_TO_USER = 1   # quick to recognize user coming back
_SWITCH_CONFIRM_USER_TO_NONUSER = 3   # conservative: need consistent evidence
_MIN_SPEAKER_SWITCH_GAP_MS = 800
# Margin below threshold where we still treat as "user" during initial gate
# (short window embeddings are noisy; false-positive initial label is harmless
#  because full-segment scoring corrects it at segment end)
_OBSERVING_MARGIN = 0.06


class PipelineState(enum.Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    RECORDING = "recording"


@dataclass
class SpeakerTurn:
    start_offset_ms: int
    end_offset_ms: int
    label: str
    score: float = 0.0


def _align_utterances(
    transcript: str,
    speaker_turns: tuple,
    duration_ms: int,
    asr_segments: list[TranscriptionSegment] | None = None,
) -> list[dict]:
    """Split transcript into utterances, assign speaker based on timestamp alignment.

    When asr_segments are provided (with real timestamps from the ASR model),
    use those directly for accurate sentence boundaries. Otherwise fall back to
    P0 proportional character mapping.

    Returns list of dicts: {speaker, text, start_ms, end_ms}
    """
    if not transcript.strip():
        return []

    # Use ASR-provided segments with real timestamps if available
    if asr_segments:
        utterances = []
        for seg in asr_segments:
            text = seg.text.strip()
            if not text:
                continue
            start_ms = seg.start_ms
            end_ms = seg.end_ms if seg.end_ms > 0 else duration_ms
            mid_ms = (start_ms + end_ms) // 2
            # Find which speaker turn contains this segment
            speaker = speaker_turns[0].label if speaker_turns else "user"
            for turn in speaker_turns:
                if turn.start_offset_ms <= mid_ms <= turn.end_offset_ms:
                    speaker = turn.label
                    break
            utterances.append({
                "speaker": speaker,
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
            })
        if utterances:
            return utterances

    # Fallback: proportional character mapping (P0)
    if not speaker_turns:
        return [{"speaker": "user", "text": transcript.strip(), "start_ms": 0, "end_ms": duration_ms}]

    parts = _PUNCT_SPLIT_RE.split(transcript)
    sentences = [p.strip() for p in parts if p.strip()]
    if not sentences:
        return [{"speaker": speaker_turns[0].label, "text": transcript.strip(),
                 "start_ms": 0, "end_ms": duration_ms}]
    total_chars = sum(len(s) for s in sentences)
    utterances = []
    char_offset = 0
    for sent in sentences:
        sent_chars = len(sent)
        proportion = sent_chars / total_chars if total_chars > 0 else 1.0 / len(sentences)
        start_char_prop = char_offset / total_chars if total_chars > 0 else 0
        end_char_prop = (char_offset + sent_chars) / total_chars if total_chars > 0 else 1.0
        start_ms = round(start_char_prop * duration_ms)
        end_ms = round(end_char_prop * duration_ms)
        mid_ms = (start_ms + end_ms) // 2
        speaker = speaker_turns[0].label
        for turn in speaker_turns:
            if turn.start_offset_ms <= mid_ms <= turn.end_offset_ms:
                speaker = turn.label
                break
        utterances.append({
            "speaker": speaker,
            "text": sent,
            "start_ms": start_ms,
            "end_ms": end_ms,
        })
        char_offset += sent_chars
    return utterances


def _compute_dominant_speaker(speaker_turns: tuple) -> tuple[str, bool]:
    """Returns (dominant_speaker_label, contains_user_flag) from speaker turns."""
    if not speaker_turns:
        return "user", True
    user_ms = 0
    other_ms = 0
    for turn in speaker_turns:
        dur = turn.end_offset_ms - turn.start_offset_ms
        if turn.label == "user":
            user_ms += dur
        else:
            other_ms += dur
    contains_user = user_ms > 0
    dominant = "user" if user_ms >= other_ms else "non-user"
    return dominant, contains_user


def _create_final_asr_adapter(model_path: Path) -> FinalASRAdapter | Qwen3ASRAdapter:
    """根据模型目录内容自动检测并创建对应的 ASR 适配器。

    检测逻辑 (优先级):
    1. config.json 且 model_type 包含 qwen3/qwen2/omni/whisper → Qwen3ASRAdapter
    2. configuration.json/config.yaml (FunASR格式) → FinalASRAdapter (Paraformer)
    3. 仅 config.json (transformers格式但未识别) → 尝试Qwen3ASRAdapter
    """
    path = _find_model_root(str(model_path))
    if not path.is_dir():
        raise RuntimeError(f"模型目录不存在: {path}")

    has_transformers_config = (path / "config.json").exists()
    has_funasr_config = (path / "configuration.json").exists() or (path / "config.yaml").exists()

    _TRANSFORMER_ASR_KEYWORDS = ("qwen3", "qwen2", "omni", "whisper", "qwen3_asr")

    # Check transformers config FIRST — Qwen3-ASR from ModelScope has BOTH config.json
    # AND configuration.json; we must detect it via model_type before defaulting to FunASR.
    if has_transformers_config:
        try:
            config = json.loads((path / "config.json").read_text())
            model_type = (config.get("model_type", "") or "").lower()
            architectures = config.get("architectures", []) or []
            arch_str = " ".join(str(a).lower() for a in architectures)
            is_transformer_asr = any(
                kw in model_type or kw in arch_str
                for kw in _TRANSFORMER_ASR_KEYWORDS
            )
            if is_transformer_asr:
                LOGGER = logging.getLogger(__name__)
                LOGGER.info(
                    "Detected transformer ASR model (model_type=%s, arch=%s) → Qwen3ASRAdapter",
                    model_type, arch_str[:80],
                )
                return Qwen3ASRAdapter(str(path))
        except (json.JSONDecodeError, OSError):
            pass

    # FunASR format (Paraformer uses configuration.json; SenseVoice uses config.yaml)
    if has_funasr_config:
        return FinalASRAdapter(str(path))

    # Unknown transformers format — try Qwen3 as fallback
    if has_transformers_config:
        return Qwen3ASRAdapter(str(path))

    raise RuntimeError(f"无法识别的模型配置格式: {path}")


class SegmentProcessor:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        stream_asr: StreamingASRAdapter,
        speaker: SpeakerEmbeddingAdapter,
        vad_model_id: str,
        voice_profile: VoiceProfileManager,
        final_asr: FinalASRAdapter | Qwen3ASRAdapter | None = None,
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
        raw_audio: np.ndarray | None = None,
        speaker_turns: tuple = (),
        end_trigger: str | None = None,
    ) -> SegmentRecord | None:
        segment_id = segment_id or uuid.uuid4().hex
        duration_ms = round(len(audio) * 1000 / self.settings.audio.sample_rate)

        # Compute multi-speaker metadata
        dominant_speaker, contains_user = _compute_dominant_speaker(speaker_turns)

        # Loudness normalize before embedding for robustness to mic distance/volume
        audio_for_embedding = normalize_loudness(audio) if self.settings.speaker.loudness_normalize else audio

        final_result = self._final(audio)
        final_text = final_result.text if final_result else ""
        # Defensive fallback: if final ASR output is dramatically shorter than the
        # streaming partial (e.g. the model was token-truncated), prefer the fuller
        # partial text instead of storing a broken half-transcript.
        if (
            final_text
            and partial
            and len(final_text.strip()) < int(len(partial.strip()) * 0.6)
        ):
            self.output(
                f"[{segment_id[:8]}] final ASR truncated: final={len(final_text.strip())}ch "
                f"vs partial={len(partial.strip())}ch; falling back to partial text"
            )
            final_text = partial
        transcript = final_text or partial
        # Empty or filler-only segments are discarded (no WAV, no DB, no speaker check).
        if not transcript.strip():
            self.output(f"[{segment_id[:8]}] discarded: empty transcript ({duration_ms}ms)")
            return None
        if self.settings.segment.discard_filler_only and _is_filler_only(transcript):
            self.output(f"[{segment_id[:8]}] discarded: filler-only {transcript!r} ({duration_ms}ms)")
            return None

        # Utterance alignment: use ASR timestamps if available, otherwise proportional mapping
        asr_segments = final_result.segments if final_result and final_result.has_timestamps else None
        utterances = _align_utterances(
            transcript, speaker_turns, duration_ms, asr_segments=asr_segments,
        )

        embedding = self.speaker.embed(audio_for_embedding, self.settings.audio.sample_rate)
        wav_path = archive_wav(
            self.settings.archive_dir, segment_id, audio,
            self.settings.audio.sample_rate, started_at,
        )
        raw_to_save = raw_audio if raw_audio is not None and raw_audio.size > 0 else audio
        raw_path = archive_wav(
            self.settings.archive_dir, segment_id, raw_to_save,
            self.settings.audio.sample_rate, started_at,
            suffix=".raw",
        )
        profile_ready = self.voice_profile.state.is_ready
        if not self.centroids or not profile_ready:
            speaker_label, score = "user", None
            effective_for_decision = "user"
        else:
            result = verify_speaker(
                embedding,
                self.centroids,
                self.threshold,
            )
            score = result.score
            fullseg_label = result.label  # "user" if score >= threshold

            # Fusion strategy: full-segment embedding is far more stable than
            # the 600ms sliding-window turns. Trust it as primary signal.
            # - fullseg says "user" → trust it (real-time turns may have initial mislabel)
            # - fullseg says "non-user" but turns contain user fragments → "user"
            #   (mid-segment switches or embedding contamination from concatenation)
            # - both agree "non-user" → non-user
            if fullseg_label == "user":
                speaker_label = "user"
            elif contains_user:
                speaker_label = "user"
            else:
                speaker_label = "non-user"

            effective_for_decision = speaker_label
            # Align dominant_speaker with the final fused label so DB records
            # and UI display reflect the corrected decision (not the noisy
            # real-time sliding-window turns that may have mislabeled at onset).
            dominant_speaker = speaker_label
            if score is not None:
                self.output(
                    f"[{segment_id[:8]}] speaker: fullseg_score={score:.3f} "
                    f"label={fullseg_label} turns_dominant={dominant_speaker} "
                    f"contains_user={contains_user} → final={speaker_label}"
                )

        # Query decision: use utterance-level if we have turns, otherwise fall back.
        # Gate query_candidate on the segment-level fused speaker label (effective_for_decision),
        # not the noisy real-time per-utterance turns — see decide_query_from_utterances.
        if speaker_turns and utterances:
            decision = decide_query_from_utterances(
                utterances,
                self.settings.vui.wake_words,
                profile_ready=profile_ready,
                dominant_speaker=effective_for_decision,
            )
        else:
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
            audio_path=str(wav_path), raw_audio_path=str(raw_path),
            sample_rate=self.settings.audio.sample_rate,
            channels=self.settings.audio.channels, transcript_raw=partial,
            transcript_final=final_text, speaker_label=speaker_label, speaker_score=score,
            wake_detected=decision.wake_detected, query_candidate=decision.query_candidate,
            query_text=decision.query_text, vad_model=self.vad_model_id,
            asr_stream_model=self.settings.models.asr_streaming,
            asr_final_model=self._final_model_label(),
            speaker_model=self.settings.models.speaker,
            created_at=datetime.now(timezone.utc).isoformat(),
            end_trigger=end_trigger,
            speaker_turns=json.dumps([
                {"start_ms": t.start_offset_ms, "end_ms": t.end_offset_ms,
                 "label": t.label, "score": t.score}
                for t in speaker_turns
            ]) if speaker_turns else None,
            utterances=json.dumps(utterances, ensure_ascii=False) if utterances else None,
            source_type="voice",
            dominant_speaker=dominant_speaker,
            contains_user=contains_user,
        )
        self.store.insert_segment(record)
        is_filler = self.settings.segment.discard_filler_only and _is_filler_only(transcript)

        # Sample collection: collect only from user-only segments (score >= threshold).
        # For multi-speaker segments (contains non-user turns), skip learning since
        # the full-audio embedding is contaminated.
        # Cold start (profile not ready): treat all as collectible with high score.
        has_non_user_turn = any(t.label == "non-user" for t in speaker_turns)
        should_try_collect = False
        collect_score = 0.8 if not profile_ready else (score if score is not None else 0.0)
        if not profile_ready:
            should_try_collect = True
            collect_score = 0.8
        elif speaker_label == "user" and score is not None and not has_non_user_turn:
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
            f"[{segment_id[:8]}] final={final_text!r} speaker={speaker_label} "
            f"score={score if score is not None else '-'} query={decision.query_candidate}"
        )
        return record

    def _final(self, audio: np.ndarray) -> TranscriptionResult | None:
        if self.final_asr is None:
            return None
        result = self.final_asr.transcribe(
            audio,
            self.settings.audio.sample_rate,
            hotword=self.hotwords,
        )
        if isinstance(result, str):
            return TranscriptionResult(text=result)
        return result

    def _final_model_label(self) -> str:
        """Record the ACTUAL loaded final-ASR model dir name, not the config default.

        The config default (settings.models.asr_final) can differ from what's
        actually loaded (the registry slot decides at runtime), which made the
        old asr_final_model column misleading.
        """
        if self.final_asr is None:
            return "unavailable"
        raw = getattr(self.final_asr, "model_path", None)
        if not raw:
            raw = getattr(self.final_asr, "model_id", None)
        if not raw:
            return type(self.final_asr).__name__
        return Path(str(raw)).name


def _safe_unload(model_obj: Any) -> None:
    """Safely call unload() on a model adapter, swallowing any exceptions."""
    import gc as _gc
    if model_obj is None:
        return
    try:
        if hasattr(model_obj, "unload"):
            model_obj.unload()
        else:
            # Best-effort cleanup for objects without unload()
            for attr in ("model", "processor", "tokenizer"):
                if hasattr(model_obj, attr):
                    try:
                        delattr(model_obj, attr)
                    except Exception:
                        pass
    except Exception:
        logging.getLogger(__name__).debug("Error during model unload", exc_info=True)
    _gc.collect()


def _log_memory(tag: str) -> None:
    """Log current process RSS for memory monitoring."""
    _logger = logging.getLogger(__name__)
    try:
        import psutil
        proc = psutil.Process()
        rss_mb = proc.memory_info().rss / (1024 * 1024)
        _logger.info("[memory] %s: RSS=%.0fMB", tag, rss_mb)
    except ImportError:
        pass
    except Exception:
        pass


@dataclass(frozen=True)
class SegmentJob:
    audio: np.ndarray          # enhanced (preprocessed) audio → ASR/speaker/enhanced wav
    started_at: datetime
    ended_at: datetime
    partial: str
    segment_id: str
    speaker_turns: tuple = ()
    raw_audio: np.ndarray | None = None  # raw (unprocessed) audio → archived as .raw.wav for future SE/context
    end_trigger: str | None = None       # vad_endpoint/max_duration/silence_timeout/... → 落库可观测


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
        final_asr_resolver: Callable[[], Path] | None = None,
    ):
        self.settings = settings
        self.paths = paths
        self.stream = stream
        self.speaker = speaker
        self.output = output
        self.emit = emit
        self._processor: SegmentProcessor | None = None
        self._shared_threshold = shared_threshold
        self._final_asr_resolver = final_asr_resolver
        self._final_asr_lock = threading.Lock()
        self._final_asr: FinalASRAdapter | Qwen3ASRAdapter | None = None
        self._reload_final = threading.Event()
        self.jobs: queue.Queue[SegmentJob | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="ev-segment-worker", daemon=True)
        self.thread.start()

    def update_thresholds(self, threshold: float | None = None) -> None:
        if self._processor is not None:
            self._processor.update_thresholds(threshold)
        if self._shared_threshold is not None and threshold is not None:
            self._shared_threshold["threshold"] = float(threshold)

    def submit(self, job: SegmentJob) -> None:
        self.jobs.put(job)

    def update_hotwords(self, hotwords: str) -> None:
        if self._processor is not None:
            self._processor.update_hotwords(hotwords)

    def reload_final_asr(self) -> None:
        """Signal the worker to reload the final ASR model on next segment.

        Unloads the old model immediately (if loaded) to free memory before
        the new model is loaded on the next segment.
        """
        old: Any = None
        with self._final_asr_lock:
            if self._final_asr is not None:
                old = self._final_asr
                self._final_asr = None
        if old is not None:
            _safe_unload(old)
            _log_memory("final-asr unloaded (switch)")
        self._reload_final.set()
        if self._processor is not None:
            self._processor.final_asr = None

    def close(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=10)
        with self._final_asr_lock:
            if self._final_asr is not None:
                _safe_unload(self._final_asr)
                self._final_asr = None
                _log_memory("final-asr unloaded (close)")

    def _run(self) -> None:
        with Store(self.settings.db_path) as store:
            from ..speaker.profile import VoiceProfileManager
            voice_profile = VoiceProfileManager(store, self.settings.voice_learning, self.settings.speaker)
            hotwords_str = store.get_hotwords_string()
            processor = SegmentProcessor(
                self.settings,
                store,
                self.stream,
                self.speaker,
                self.settings.models.vad,
                voice_profile,
                None,
                self.output,
                self.emit,
                hotwords=hotwords_str,
            )
            self._processor = processor
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
                    duration_ms = round(len(job.audio) * 1000 / self.settings.audio.sample_rate)

                    # Very short segments (< 800ms) with no speaker turns AND empty/meaningless partial
                    # skip final ASR to save inference cost. Final ASR provides hotword boosting
                    # and higher accuracy, so we only skip for truly tiny/no-content segments.
                    _VERY_SHORT_MS = 800
                    partial_clean = job.partial.strip() if job.partial else ""
                    is_very_short = duration_ms < _VERY_SHORT_MS and not job.speaker_turns
                    has_no_content = (not partial_clean) or _is_filler_only(partial_clean)
                    skip_final = is_very_short and has_no_content

                    if not skip_final:
                        with self._final_asr_lock:
                            final = self._final_asr
                        if final is None:
                            final_path = self._resolve_final_asr_path()
                            self.output(f"[final-asr] loading from {final_path.name}...")
                            final = _create_final_asr_adapter(final_path)
                            with self._final_asr_lock:
                                self._final_asr = final
                            processor.final_asr = final
                            self._reload_final.clear()
                            self.output(f"[final-asr] {type(final).__name__} ready")
                            _log_memory(f"final-asr loaded ({type(final).__name__})")
                    else:
                        # Temporarily bypass final ASR for this very short empty segment
                        processor.final_asr = None
                        self.output(
                            f"[{job.segment_id[:8]}] very short no-content segment "
                            f"({duration_ms}ms), skipping final ASR"
                        )

                    record = processor.process(
                        job.audio,
                        job.started_at,
                        job.ended_at,
                        job.partial,
                        job.segment_id,
                        raw_audio=job.raw_audio,
                        speaker_turns=job.speaker_turns,
                        end_trigger=job.end_trigger,
                    )

                    # Restore final_asr on processor after short-segment skip
                    if skip_final:
                        with self._final_asr_lock:
                            processor.final_asr = self._final_asr

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

    def _resolve_final_asr_path(self) -> Path:
        if self._final_asr_resolver is not None:
            try:
                return self._final_asr_resolver()
            except Exception:
                pass
        return self.paths["asr_final"]


async def transcribe_forever(
    settings: Settings,
    device: int | None = None,
    model_root: Path | None = None,
    output: Callable[[str], None] = print,
    stop_event: threading.Event | None = None,
    emit: Callable[[str, dict], None] | None = None,
    worker_holder: dict | None = None,
    final_asr_resolver: Callable[[], Path] | None = None,
) -> None:
    from ..models import resolve_model_paths, verify_models

    # Resolve base paths for non-switchable models (vad, streaming, speaker)
    base_paths = resolve_model_paths(settings.models, model_root)
    # Strictly verify only vad, asr_streaming, speaker — asr_final comes from resolver
    strict_keys = ("vad", "asr_streaming", "speaker")
    checks = verify_models(settings.models, model_root)
    failed = [
        f"{c.key}: {', '.join(c.errors)} ({c.path})"
        for c in checks
        if not c.ok and c.key in strict_keys
    ]
    if failed:
        raise RuntimeError("本地模型校验失败:\n" + "\n".join(failed))
    paths = {c.key: c.path for c in checks if c.key in strict_keys}

    # Resolve asr_final from registry or fall back to old settings path
    if final_asr_resolver is not None:
        paths["asr_final"] = final_asr_resolver()
    else:
        paths["asr_final"] = base_paths["asr_final"]

    settings.ensure_dirs()

    def send(event_type: str, payload: dict) -> None:
        if emit is not None:
            emit(event_type, payload)

    # --- 预处理 & VAD 初始化 ---
    sr = settings.audio.sample_rate
    frame_ms_default = 30
    # Pre-declare model references so finally block can safely unload them
    # even if an exception occurs during initialization
    preprocessor: AudioPreprocessor | None = None
    vad_model: VADAdapter | None = None
    energy_vad: EnergyVAD | None = None
    vad: CompositeVAD | None = None
    stream: StreamingASRAdapter | None = None
    speaker: SpeakerEmbeddingAdapter | None = None
    capture: AudioCapture | None = None
    # 1) 预处理管线 (DC → preemphasis → AGC → NoiseGate)
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

    # Initialize DB and load voice profile BEFORE starting the worker thread
    # to avoid sqlite lock contention during first-time schema creation
    def load_centroids():
        with Store(settings.db_path) as s:
            vp = VoiceProfileManager(s, settings.voice_learning, settings.speaker)
            return vp.centroids if vp.state.is_ready else []

    profile_centroids = load_centroids()

    worker = SegmentWorker(
        settings, paths, stream, speaker, output, send, shared_threshold,
        final_asr_resolver=final_asr_resolver,
    )
    if worker_holder is not None:
        worker_holder["worker"] = worker

    try:
        pre_roll_frames = max(1, 1200 // 30)
        frame_size = int(sr * frame_ms_default / 1000)

        # --- State machine variables ---
        state = PipelineState.IDLE
        # Buffers per state
        recent_frames: list[np.ndarray] = []       # pre-roll (enhanced)
        recent_raw_frames: list[np.ndarray] = []   # pre-roll (raw)
        observing_frames: list[np.ndarray] = []
        observing_raw_frames: list[np.ndarray] = []
        observe_start_samples = 0
        frames: list[np.ndarray] = []              # RECORDING enhanced
        raw_frames: list[np.ndarray] = []          # RECORDING raw
        started_at: datetime | None = None
        partial = ""
        current_segment_id: str | None = None
        speaker_turns: list[SpeakerTurn] = []
        current_turn_label: str | None = None
        current_turn_start_samples = 0
        last_speaker_check_samples = 0
        speaker_switch_streak = 0
        last_switch_samples = 0
        silence_samples = 0  # consecutive low-RMS samples for silence timeout (raw audio)
        energy_silent_samples = 0  # consecutive samples where EnergyVAD SNR is below threshold
        peak_rms = 0.0  # peak raw RMS during this segment (for relative silence detection)
        relative_silence_samples = 0  # consecutive samples below relative silence threshold
        last_partial_samples = 0  # samples count when partial was last updated
        local_stop = asyncio.Event()

        def _score_window(buf_audio: np.ndarray) -> float:
            """Score a window of audio against profile centroids. Returns 0.0 if no profile."""
            if not profile_centroids:
                return 1.0  # cold start: treat as user
            try:
                if settings.speaker.loudness_normalize:
                    buf_audio = normalize_loudness(buf_audio)
                emb = speaker.embed(buf_audio, sr)
                norm_emb = normalize_embedding(emb)
                return max(float(np.dot(norm_emb, normalize_embedding(c))) for c in profile_centroids)
            except Exception as exc:
                output(f"speaker score failed: {exc}")
                return 0.5

        def _close_current_turn(end_samples: int) -> None:
            nonlocal speaker_turns, current_turn_label, current_turn_start_samples
            if current_turn_label is None:
                return
            start_ms = round(current_turn_start_samples * 1000 / sr)
            end_ms = round(end_samples * 1000 / sr)
            speaker_turns.append(SpeakerTurn(
                start_offset_ms=start_ms,
                end_offset_ms=end_ms,
                label=current_turn_label,
            ))
            current_turn_label = None

        def _start_turn(label: str, start_samples: int) -> None:
            nonlocal current_turn_label, current_turn_start_samples
            current_turn_label = label
            current_turn_start_samples = start_samples

        def _reset_observing() -> None:
            nonlocal observing_frames, observing_raw_frames, observe_start_samples
            observing_frames = []
            observing_raw_frames = []
            observe_start_samples = 0

        def _reset_recording() -> None:
            nonlocal frames, raw_frames, started_at, partial, current_segment_id
            nonlocal speaker_turns, current_turn_label, current_turn_start_samples
            nonlocal last_speaker_check_samples, speaker_switch_streak, last_switch_samples
            nonlocal silence_samples, energy_silent_samples, peak_rms
            nonlocal relative_silence_samples, last_partial_samples
            frames = []
            raw_frames = []
            started_at = None
            partial = ""
            current_segment_id = None
            speaker_turns = []
            current_turn_label = None
            current_turn_start_samples = 0
            last_speaker_check_samples = 0
            speaker_switch_streak = 0
            last_switch_samples = 0
            silence_samples = 0
            energy_silent_samples = 0
            peak_rms = 0.0
            relative_silence_samples = 0
            last_partial_samples = 0

        def force_segment_end(trigger_reason: str) -> None:
            nonlocal state
            if state != PipelineState.RECORDING or started_at is None or current_segment_id is None:
                state = PipelineState.IDLE
                _reset_observing()
                _reset_recording()
                vad.reset()
                stream.reset()
                return
            ended_at = datetime.now(timezone.utc)
            total_samples = sum(f.size for f in frames)
            # Close the last speaker turn
            _close_current_turn(total_samples)
            final_partial = stream.accept(
                np.empty(0, dtype=np.float32),
                sr,
                is_final=True,
            )
            send("speech_ended", {
                "segment_id": current_segment_id,
                "ended_at": ended_at.isoformat(),
                "trigger": trigger_reason,
            })
            seg_audio = np.concatenate(frames) if frames else np.empty(0, dtype=np.float32)
            seg_raw = np.concatenate(raw_frames) if raw_frames else seg_audio
            seg_duration_ms = round(len(seg_audio) * 1000 / sr)
            if seg_duration_ms >= settings.segment.min_duration_ms:
                worker.submit(
                    SegmentJob(
                        audio=seg_audio,
                        raw_audio=seg_raw,
                        started_at=started_at,
                        ended_at=ended_at,
                        partial=final_partial,
                        segment_id=current_segment_id,
                        speaker_turns=tuple(speaker_turns),
                        end_trigger=trigger_reason,
                    )
                )
            else:
                output(f"[{current_segment_id[:8]}] discarded (too_short, {seg_duration_ms}ms, trigger={trigger_reason})")
                send("segment_discarded", {
                    "segment_id": current_segment_id,
                    "reason": "too_short",
                    "trigger": trigger_reason,
                    "duration_ms": seg_duration_ms,
                })
            state = PipelineState.IDLE
            _reset_observing()
            _reset_recording()
            vad.reset()
            stream.reset()
            # Reload centroids after segment ends
            nonlocal profile_centroids
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
            iterator = capture.frames_with_raw().__aiter__()
            try:
                first_pair = await asyncio.wait_for(iterator.__anext__(), timeout=2.0)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("麦克风已打开，但两秒内没有收到音频帧") from exc

            def _append_trim(buf: list[np.ndarray], frame: np.ndarray, max_len: int) -> None:
                buf.append(frame)
                if len(buf) > max_len:
                    buf.pop(0)

            def _current_gain() -> float:
                preprocessor = getattr(capture, "preprocessor", None)
                if preprocessor is not None:
                    return float(preprocessor.last_gain)
                return 1.0

            def _enter_recording(initial_label: str, initial_frames: list[np.ndarray], initial_raw: list[np.ndarray]) -> None:
                """Transition from OBSERVING (or cold start) to RECORDING state."""
                nonlocal state, frames, raw_frames, started_at, partial, current_segment_id
                nonlocal speaker_turns, current_turn_label, current_turn_start_samples
                nonlocal last_speaker_check_samples, speaker_switch_streak, last_switch_samples
                nonlocal silence_samples, energy_silent_samples, peak_rms
                nonlocal relative_silence_samples, last_partial_samples
                state = PipelineState.RECORDING
                frames = list(initial_frames)
                raw_frames = list(initial_raw)
                duration = sum(f.size for f in frames) / sr
                started_at = datetime.now(timezone.utc) - timedelta(seconds=duration)
                current_segment_id = uuid.uuid4().hex
                speaker_turns = []
                _start_turn(initial_label, 0)
                last_speaker_check_samples = sum(f.size for f in frames)
                speaker_switch_streak = 0
                last_switch_samples = 0
                silence_samples = 0
                energy_silent_samples = 0
                peak_rms = 0.0
                relative_silence_samples = 0
                stream.reset()
                partial = stream.accept(np.concatenate(frames), sr)
                # Only start stall timer after we get actual partial content
                last_partial_samples = sum(f.size for f in frames) if partial else 0
                send(
                    "speech_started",
                    {
                        "segment_id": current_segment_id,
                        "started_at": started_at.isoformat(),
                        "speaker_label": initial_label,
                    },
                )
                if partial:
                    send("transcript_partial", {"segment_id": current_segment_id, "text": partial})
                output(f"[{current_segment_id[:8]}] recording started speaker={initial_label}")

            def _check_speaker_switch() -> None:
                """Periodic speaker check during RECORDING: record turn changes without truncating.

                Uses asymmetric hysteresis:
                - non-user → user: 1 confirmation (fast recovery when user comes back)
                - user → non-user: _SWITCH_CONFIRM_USER_TO_NONUSER confirmations
                  (conservative: avoid flipping to non-user on a single noisy window)
                """
                nonlocal last_speaker_check_samples, speaker_switch_streak, last_switch_samples, current_turn_label
                total_samples = sum(f.size for f in frames)
                interval_samples = _SPEAKER_CHECK_INTERVAL_MS * sr // 1000
                gate_samples = _OBSERVING_GATE_MS * sr // 1000
                min_gap_samples = _MIN_SPEAKER_SWITCH_GAP_MS * sr // 1000
                if total_samples < gate_samples:
                    return
                if (total_samples - last_speaker_check_samples) < interval_samples:
                    return
                last_speaker_check_samples = total_samples
                # Build sliding window (last 600ms)
                window_buf = np.empty(0, dtype=np.float32)
                acc = 0
                window_target = _SPEAKER_CHECK_INTERVAL_MS * sr // 1000
                for f in reversed(frames):
                    if acc >= window_target:
                        break
                    window_buf = np.concatenate([f, window_buf])
                    acc += f.size
                score = _score_window(window_buf)
                is_user = score >= shared_threshold["threshold"]
                new_label = "user" if is_user else "non-user"
                if current_turn_label == new_label:
                    speaker_switch_streak = 0
                    return
                speaker_switch_streak += 1
                # Asymmetric confirmation: faster to recognize user, slower to lose them
                if new_label == "user":
                    needed = _SWITCH_CONFIRM_NONUSER_TO_USER
                else:
                    needed = _SWITCH_CONFIRM_USER_TO_NONUSER
                if speaker_switch_streak >= needed and (total_samples - last_switch_samples) >= min_gap_samples:
                    # Speaker switch confirmed: close current turn, start new turn
                    _close_current_turn(total_samples - window_target // 2)
                    _start_turn(new_label, total_samples - window_target // 2)
                    last_switch_samples = total_samples
                    speaker_switch_streak = 0
                    output(f"speaker turn change: {current_turn_label} → {new_label} (score={score:.3f}, confirm={needed})")
                    send("speaker_turn_changed", {
                        "segment_id": current_segment_id,
                        "from": current_turn_label,
                        "to": new_label,
                        "score": score,
                    })

            async def handle_frame(processed: np.ndarray, raw: np.ndarray) -> None:
                nonlocal state, observing_frames, observing_raw_frames, observe_start_samples
                nonlocal frames, raw_frames, started_at, partial, current_segment_id
                nonlocal speaker_turns, current_turn_label, current_turn_start_samples
                nonlocal last_speaker_check_samples, speaker_switch_streak, last_switch_samples
                nonlocal silence_samples, energy_silent_samples, peak_rms
                nonlocal relative_silence_samples, last_partial_samples
                processed_rms = float(np.sqrt(np.mean(processed**2)))
                raw_rms = float(np.sqrt(np.mean(raw**2)))
                send("audio_level", {
                    "rms": processed_rms,
                    "raw_rms": raw_rms,
                    "gain": _current_gain(),
                })
                _append_trim(recent_frames, processed, pre_roll_frames)
                _append_trim(recent_raw_frames, raw, pre_roll_frames)

                if state == PipelineState.IDLE:
                    # Check for VAD start
                    for boundary in vad.accept(processed, sr):
                        if boundary.started:
                            if profile_centroids:
                                # Enter OBSERVING: accumulate frames for gate duration
                                state = PipelineState.OBSERVING
                                observing_frames = list(recent_frames)
                                observing_raw_frames = list(recent_raw_frames)
                                observe_start_samples = sum(f.size for f in observing_frames)
                            else:
                                # Cold start: directly record as user (for learning)
                                _enter_recording("user", list(recent_frames), list(recent_raw_frames))

                elif state == PipelineState.OBSERVING:
                    observing_frames.append(processed)
                    observing_raw_frames.append(raw)
                    obs_samples = sum(f.size for f in observing_frames)
                    gate_samples = _OBSERVING_GATE_MS * sr // 1000
                    vad_ended = False
                    for boundary in vad.accept(processed, sr):
                        if boundary.ended:
                            vad_ended = True
                    if vad_ended:
                        output(f"[observing] speech ended during gate ({obs_samples/sr*1000:.0f}ms), discarding")
                        state = PipelineState.IDLE
                        _reset_observing()
                        vad.reset()
                        return
                    if obs_samples >= gate_samples:
                        obs_audio = np.concatenate(observing_frames)
                        score = _score_window(obs_audio)
                        # Generous threshold for initial gate: scores within
                        # _OBSERVING_MARGIN below the real threshold are still
                        # labeled "user". Short-window embeddings are noisy and
                        # the full-segment score at segment end will correct
                        # any false positive. This prevents the user's own
                        # voice from being labeled non-user at the very start
                        # due to onset/breath noise.
                        init_threshold = shared_threshold["threshold"] - _OBSERVING_MARGIN
                        label = "user" if score >= init_threshold else "non-user"
                        obs_f = list(observing_frames)
                        obs_r = list(observing_raw_frames)
                        _reset_observing()
                        _enter_recording(label, obs_f, obs_r)

                elif state == PipelineState.RECORDING:
                    frames.append(processed)
                    raw_frames.append(raw)
                    frame_samples = processed.size
                    total_samples = sum(f.size for f in frames)
                    total_duration_ms = round(total_samples * 1000 / sr)
                    min_dur = settings.segment.min_duration_ms

                    # Update peak RMS using raw audio (not AGC-processed)
                    if raw_rms > peak_rms:
                        peak_rms = raw_rms

                    candidate = stream.accept(processed, sr)
                    if candidate and candidate != partial:
                        partial = candidate
                        last_partial_samples = total_samples
                        output(f"partial: {partial}")
                        send("transcript_partial", {"segment_id": current_segment_id, "text": partial})
                    # Periodic speaker check (turn change detection)
                    if profile_centroids:
                        _check_speaker_switch()
                    # Drive VAD (updates both FSMN and EnergyVAD state)
                    vad_ended = False
                    for boundary in vad.accept(processed, sr):
                        if boundary.ended:
                            vad_ended = True

                    # --- Silence detection using RAW audio (no AGC distortion) ---
                    # 1) Absolute silence (raw RMS very low = truly quiet environment)
                    if raw_rms < settings.segment.silence_rms_threshold and total_duration_ms >= min_dur:
                        silence_samples += frame_samples
                    else:
                        silence_samples = 0
                    silence_ms = round(silence_samples * 1000 / sr)

                    # 2) Relative silence: RMS dropped significantly from speaking peak
                    relative_threshold = peak_rms * settings.segment.relative_silence_ratio
                    # Only enable relative silence after we've seen some speech (peak > abs threshold * 3)
                    relative_enabled = peak_rms > settings.segment.silence_rms_threshold * 3
                    if relative_enabled and raw_rms < relative_threshold and total_duration_ms >= min_dur:
                        relative_silence_samples += frame_samples
                    else:
                        relative_silence_samples = 0
                    relative_silence_ms = round(relative_silence_samples * 1000 / sr)

                    # 3) EnergyVAD-adaptive silence: track how long SNR has indicated speech-absence
                    # energy_vad.silence_ms counts non-speech frames during its hangover;
                    # once energy_vad goes inactive (hangover exhausted), keep counting.
                    energy_total_silent_ms = 0.0
                    if energy_vad is not None:
                        if energy_vad.active:
                            energy_total_silent_ms = energy_vad.silence_ms
                            energy_silent_samples = 0
                        elif total_duration_ms >= min_dur:
                            # EnergyVAD finished hangover → FSMN is stuck; continue counting
                            energy_silent_samples += frame_samples
                            # hangover is energy_hangover_frames * 30ms = 600ms by default
                            energy_total_silent_ms = (
                                settings.vad.energy_hangover_frames * frame_ms_default
                                + round(energy_silent_samples * 1000 / sr)
                            )
                        else:
                            energy_silent_samples = 0

                    # 4) ASR stall: no new partial for too long = done speaking
                    # Only trigger after we've gotten at least some partial content
                    ms_since_last_partial = 0
                    if last_partial_samples > 0:
                        ms_since_last_partial = round((total_samples - last_partial_samples) * 1000 / sr)

                    # --- Endpoint decision (priority order) ---
                    # 1) Normal VAD end (primary path)
                    if vad_ended:
                        force_segment_end("vad_endpoint")
                        return
                    # 2) Hard max duration (safety net)
                    if total_duration_ms >= settings.segment.max_duration_ms:
                        output(f"[{current_segment_id[:8]}] force_end: max_duration ({total_duration_ms}ms)")
                        force_segment_end("max_duration")
                        return
                    # 3) Absolute silence timeout (very quiet)
                    #    Gate on a minimum segment length so a freshly-started
                    #    segment isn't cut by a brief pause right at the onset.
                    if (
                        silence_ms >= settings.segment.silence_timeout_ms
                        and total_duration_ms >= settings.segment.min_duration_for_silence_ms
                    ):
                        output(f"[{current_segment_id[:8]}] force_end: silence_timeout ({silence_ms}ms silent, peak={peak_rms:.4f}, total={total_duration_ms}ms)")
                        force_segment_end("silence_timeout")
                        return
                    # 4) Relative silence (noise floor present but RMS dropped from speech level)
                    #    Longer gate: only cuts long segments, never short ones.
                    if (
                        relative_silence_ms >= settings.segment.relative_silence_timeout_ms
                        and total_duration_ms >= settings.segment.min_duration_for_relative_silence_ms
                    ):
                        output(f"[{current_segment_id[:8]}] force_end: relative_silence ({relative_silence_ms}ms, peak={peak_rms:.4f}, rms={raw_rms:.4f}, ratio={raw_rms/peak_rms:.2f}, total={total_duration_ms}ms)")
                        force_segment_end("relative_silence")
                        return
                    # 5) EnergyVAD says silent for long enough (adaptive SNR-based, beats stuck FSMN)
                    if (
                        energy_total_silent_ms >= settings.segment.silence_timeout_ms + 500
                        and total_duration_ms >= settings.segment.min_duration_for_silence_ms
                    ):
                        output(f"[{current_segment_id[:8]}] force_end: energy_silent ({energy_total_silent_ms:.0f}ms, total={total_duration_ms}ms)")
                        force_segment_end("energy_silent")
                        return
                    # 6) ASR stall: streaming model stopped producing new text.
                    #    Only fire when audio is genuinely quiet too — if the speaker is
                    #    still talking (raw RMS above the silence floor) while ASR lags,
                    #    that's a model stall, not the end of speech; don't cut early.
                    if (
                        ms_since_last_partial >= settings.segment.asr_stall_timeout_ms
                        and total_duration_ms >= min_dur + 500
                        and raw_rms < settings.segment.silence_rms_threshold
                    ):
                        output(f"[{current_segment_id[:8]}] force_end: asr_stall ({ms_since_last_partial}ms since partial, rms={raw_rms:.4f}, total={total_duration_ms}ms)")
                        force_segment_end("asr_stall")
                        return

            await handle_frame(*first_pair)
            async for p, r in iterator:
                await handle_frame(p, r)
                if local_stop.is_set() or (stop_event is not None and stop_event.is_set()):
                    break
        finally:
            # Flush final boundaries
            final_boundaries = vad.accept(
                np.empty(0, dtype=np.float32), sr, is_final=True
            )
            if state == PipelineState.IDLE and any(item.started for item in final_boundaries):
                if profile_centroids:
                    state = PipelineState.OBSERVING
                    observing_frames = list(recent_frames)
                    observing_raw_frames = list(recent_raw_frames)
                else:
                    _enter_recording("user", list(recent_frames), list(recent_raw_frames))
            elif state == PipelineState.OBSERVING:
                if observing_frames:
                    obs_audio = np.concatenate(observing_frames)
                    score = _score_window(obs_audio)
                    label = "user" if score >= shared_threshold["threshold"] else "non-user"
                    obs_f = list(observing_frames)
                    obs_r = list(observing_raw_frames)
                    _reset_observing()
                    _enter_recording(label, obs_f, obs_r)
            if state == PipelineState.RECORDING and started_at is not None and frames and current_segment_id:
                ended_at = datetime.now(timezone.utc)
                total_samples = sum(f.size for f in frames)
                _close_current_turn(total_samples)
                partial = stream.accept(
                    np.empty(0, dtype=np.float32), sr, is_final=True
                )
                send("speech_ended", {"segment_id": current_segment_id, "ended_at": ended_at.isoformat()})
                seg_audio = np.concatenate(frames)
                seg_raw = np.concatenate(raw_frames) if raw_frames else seg_audio
                seg_duration_ms = round(len(seg_audio) * 1000 / sr)
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
                        SegmentJob(
                            audio=seg_audio,
                            raw_audio=seg_raw,
                            started_at=started_at,
                            ended_at=ended_at,
                            partial=partial,
                            segment_id=current_segment_id,
                            speaker_turns=tuple(speaker_turns),
                            end_trigger="stop",
                        )
                    )
            capture.stop()
            if can_install_signal and previous is not None:
                signal.signal(signal.SIGINT, previous)
    finally:
        worker.close()
        # Explicitly unload all models to free memory (especially PyTorch/MPS)
        for _obj in (vad, stream, speaker, energy_vad, vad_model):
            if _obj is not None:
                _safe_unload(_obj)
        # Stop audio capture if still running
        if capture is not None:
            try:
                capture.stop()
            except Exception:
                pass
        # Force Python GC and PyTorch cache clear
        import gc as _gc
        _gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        _log_memory("pipeline shutdown complete")


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
