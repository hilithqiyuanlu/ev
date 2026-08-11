"""FunASR 统一适配层。CLI 和管道不直接依赖 FunASR API。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3000-\u303f\uff00-\uffef]")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+\|>")
_TIMESTAMP_RE = re.compile(r"\(\d+(\.\d+)?-\d+(\.\d+)?\)")


@dataclass
class TranscriptionSegment:
    """A sentence/utterance-level segment from ASR with optional timestamps.

    When start_ms/end_ms are 0 (not provided by the model), alignment falls back
    to proportional character mapping.
    """
    text: str
    start_ms: int = 0
    end_ms: int = 0


@dataclass
class TranscriptionResult:
    """Structured result from final ASR transcription.

    text:        full cleaned transcript string
    segments:    sentence-level segments with timestamps (empty if model doesn't support them)
    language:    detected language code (if model provides it)
    confidence:  average confidence (if model provides it, else None)
    """
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str | None = None
    confidence: float | None = None
    avg_logprob: float | None = None  # 生成序列的平均 log 概率 (越低越不确定)

    @property
    def has_timestamps(self) -> bool:
        return bool(self.segments) and any(s.end_ms > 0 for s in self.segments)


def _find_model_root(path: str) -> Path:
    """Find the actual model directory containing config files.

    Handles cases where tar extraction creates a nested directory
    (e.g. ev-paraformer-zh-16k/ev-paraformer-zh-16k/).
    Looks for FunASR (configuration.json/config.yaml) or HF (config.json) markers.
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        return root
    _CONFIG_MARKERS = ("configuration.json", "config.yaml", "config.json")
    for marker in _CONFIG_MARKERS:
        if (root / marker).exists():
            return root
    # Single-level nesting: look for subdirectory containing a config marker
    for child in root.iterdir():
        if child.is_dir():
            for marker in _CONFIG_MARKERS:
                if (child / marker).exists():
                    return child
    return root


def _is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def _clean_text(text: str) -> str:
    text = _SPECIAL_TOKEN_RE.sub("", text)
    text = _TIMESTAMP_RE.sub("", text)
    if not text:
        return ""
    # paraformer-zh outputs spaces between every Chinese character (e.g. "欢 迎 大 家").
    # Collapse spaces between adjacent CJK characters while preserving:
    # - spaces between CJK and ASCII (English words stay separated)
    # - spaces between ASCII words
    chars = list(text)
    out: list[str] = []
    n = len(chars)
    i = 0
    while i < n:
        ch = chars[i]
        if ch == " ":
            # Look at prev non-space and next non-space to decide whether to keep this space.
            prev_ch = out[-1] if out else ""
            j = i + 1
            while j < n and chars[j] == " ":
                j += 1
            next_ch = chars[j] if j < n else ""
            prev_is_cjk = bool(prev_ch) and _is_cjk(prev_ch)
            next_is_cjk = bool(next_ch) and _is_cjk(next_ch)
            # Keep space unless both sides are CJK (in which case it was an inter-char separator).
            if not (prev_is_cjk and next_is_cjk) and prev_ch and next_ch:
                out.append(" ")
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out).strip()


def _text(result: Any) -> str:
    if isinstance(result, list):
        result = result[0] if result else {}
    if isinstance(result, dict):
        value = str(result.get("text", ""))
    else:
        value = str(result or "")
    return _clean_text(value)


class _FunASR:
    def __init__(
        self,
        model_path: str,
        model: Any | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ):
        resolved_path = _find_model_root(model_path)
        self.model_id = str(resolved_path)
        self.model = model
        if model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError("ASR 需要安装 FunASR，请先安装运行时依赖") from exc
            revision = kwargs.pop("model_revision", None)
            options = {"model": model_name or str(resolved_path), "disable_pbar": True, **kwargs}
            if model_name:
                options["model_path"] = str(resolved_path)
            if revision:
                options["model_revision"] = revision
            self.model = AutoModel(**options)

    def unload(self) -> None:
        """Release model resources. Safe to call multiple times."""
        import gc

        if self.model is not None:
            try:
                # FunASR AutoModel may hold onnxruntime sessions
                if hasattr(self.model, "model"):
                    try:
                        del self.model.model
                    except Exception:
                        pass
                # Try to close any onnxruntime inference sessions
                for attr_name in dir(self.model):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        attr = getattr(self.model, attr_name, None)
                        if attr is not None and hasattr(attr, "close"):
                            attr.close()
                    except Exception:
                        pass
            except Exception:
                pass
            self.model = None
        gc.collect()


class StreamingASRAdapter(_FunASR):
    CHUNK_SIZE = [0, 10, 5]
    ENCODER_LOOK_BACK = 4
    DECODER_LOOK_BACK = 1

    def __init__(self, model_path: str, model: Any | None = None):
        super().__init__(model_path, model, disable_update=True)
        self.cache: dict[str, Any] = {}
        self._buffer = np.empty(0, dtype=np.float32)
        self._pieces: list[str] = []

    def accept(self, frame: np.ndarray, sample_rate: int = 16000, is_final: bool = False) -> str:
        self._buffer = np.concatenate(
            [self._buffer, np.asarray(frame, dtype=np.float32).reshape(-1)]
        )
        chunk_samples = sample_rate * 600 // 1000
        while self._buffer.size >= chunk_samples:
            chunk = self._buffer[:chunk_samples]
            self._buffer = self._buffer[chunk_samples:]
            self._generate(chunk, sample_rate, False)
        if is_final and self._buffer.size:
            chunk = self._buffer
            self._buffer = np.empty(0, dtype=np.float32)
            self._generate(chunk, sample_rate, True)
        return "".join(self._pieces).strip()

    def _generate(self, audio: np.ndarray, sample_rate: int, is_final: bool) -> None:
        result = self.model.generate(
            input=audio,
            cache=self.cache,
            is_final=is_final,
            sampling_rate=sample_rate,
            chunk_size=self.CHUNK_SIZE,
            encoder_chunk_look_back=self.ENCODER_LOOK_BACK,
            decoder_chunk_look_back=self.DECODER_LOOK_BACK,
            disable_pbar=True,
        )
        piece = _text(result)
        if piece:
            self._pieces.append(piece)

    def reset(self) -> None:
        self.cache.clear()
        self._buffer = np.empty(0, dtype=np.float32)
        self._pieces = []

    def unload(self) -> None:
        self.cache.clear()
        self._buffer = np.empty(0, dtype=np.float32)
        self._pieces = []
        super().unload()


class SenseVoiceAdapter(_FunASR):
    """SenseVoice Small — 多语言终稿 ASR（FunASR 引擎）。"""

    def __init__(self, model_path: str, model: Any | None = None):
        super().__init__(
            model_path,
            model,
            model_name="iic/SenseVoiceSmall",
            vad_model=None,
            punc_model=None,
            trust_remote_code=False,
            disable_update=True,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        hotword: str = "",
    ) -> TranscriptionResult:
        kwargs: dict[str, Any] = {
            "input": audio,
            "sampling_rate": sample_rate,
            "use_itn": True,
            "batch_size_s": 300,
            "disable_pbar": True,
        }
        if hotword:
            kwargs["hotword"] = hotword
            logging.getLogger(__name__).debug(
                "SenseVoiceASR hotwords: %s", hotword[:100],
            )
        result = self.model.generate(**kwargs)
        text = _text(result)
        return TranscriptionResult(text=text)


class SpeakerEmbeddingAdapter(_FunASR):
    def __init__(self, model_path: str, model: Any | None = None):
        super().__init__(
            model_path,
            model,
            model_name="iic/speech_eres2netv2_sv_zh-cn_16k-common",
            disable_update=True,
        )

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        result = self.model.generate(
            input=audio, sampling_rate=sample_rate, disable_pbar=True
        )
        if isinstance(result, list):
            result = result[0] if result else None
        if isinstance(result, dict):
            result = result.get("spk_embedding", result.get("embedding"))
        if result is None:
            raise RuntimeError("声纹模型没有返回 embedding")
        return np.asarray(result, dtype=np.float32).reshape(-1)
