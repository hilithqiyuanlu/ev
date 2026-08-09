"""FunASR 统一适配层。CLI 和管道不直接依赖 FunASR API。"""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def _text(result: Any) -> str:
    if isinstance(result, list):
        result = result[0] if result else {}
    if isinstance(result, dict):
        value = str(result.get("text", ""))
    else:
        value = str(result or "")
    return re.sub(r"<\|[^|]+\|>", "", value).strip()


class _FunASR:
    def __init__(
        self,
        model_path: str,
        model: Any | None = None,
        model_name: str | None = None,
        **kwargs: Any,
    ):
        self.model_id = model_path
        self.model = model
        if model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError("ASR 需要安装 FunASR，请先安装运行时依赖") from exc
            revision = kwargs.pop("model_revision", None)
            options = {"model": model_name or model_path, **kwargs}
            if model_name:
                options["model_path"] = model_path
            if revision:
                options["model_revision"] = revision
            self.model = AutoModel(**options)


class StreamingASRAdapter(_FunASR):
    def __init__(self, model_path: str, model: Any | None = None):
        super().__init__(model_path, model, disable_update=True)
        self.cache: dict[str, Any] = {}

    def accept(self, frame: np.ndarray, sample_rate: int = 16000, is_final: bool = False) -> str:
        result = self.model.generate(
            input=frame, cache=self.cache, is_final=is_final, sampling_rate=sample_rate
        )
        return _text(result)

    def reset(self) -> None:
        self.cache.clear()


class FinalASRAdapter(_FunASR):
    def __init__(self, model_path: str, model: Any | None = None):
        super().__init__(model_path, model, trust_remote_code=False, disable_update=True)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        result = self.model.generate(
            input=audio, sampling_rate=sample_rate, use_itn=True, batch_size_s=300
        )
        return _text(result)


class SpeakerEmbeddingAdapter(_FunASR):
    def __init__(self, model_path: str, model: Any | None = None):
        super().__init__(
            model_path,
            model,
            model_name="iic/speech_eres2netv2_sv_zh-cn_16k-common",
            disable_update=True,
        )

    def embed(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        result = self.model.generate(input=audio, sampling_rate=sample_rate)
        if isinstance(result, list):
            result = result[0] if result else None
        if isinstance(result, dict):
            result = result.get("spk_embedding", result.get("embedding"))
        if result is None:
            raise RuntimeError("声纹模型没有返回 embedding")
        return np.asarray(result, dtype=np.float32).reshape(-1)
