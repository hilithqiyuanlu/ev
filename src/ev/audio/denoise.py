"""降噪适配器 — 封装 ModelScope DFSMN-ANLMS 语音增强模型。

通过 modelscope.pipelines 加载 DFSMN 降噪模型。
模型原生 48kHz，适配器内部处理 16kHz↔48kHz 重采样。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..store.audio import read_wav
from .utils import resample

LOGGER = logging.getLogger(__name__)

# DFSMN 模型原生采样率
_DFSMN_NATIVE_SR = 48000


class DenoiseAdapter:
    """DFSMN-ANLMS 降噪适配器。

    通过 ModelScope pipeline API 加载 DFSMN 语音增强模型。
    非流式 — 接收完整音频段，一次 forward pass 返回降噪后音频。
    输入/输出均为 16kHz，内部自动处理 48kHz 重采样。

    若 modelscope/speechbrain 未安装，enhance() 静默回退到原始音频。
    """

    def __init__(self, model_id: str | None = None, model_path: str | None = None) -> None:
        # model_path 优先（本地模型），否则回退到 model_id（ModelScope 云端）
        self.model_id = model_path or model_id or "damo/speech_dfsmn_ans_psm_48k_causal"
        self._pipeline: Any = None
        self._load_attempted = False
        self.last_error: str | None = None  # 最近一次加载失败原因，供上层暴露

    def _load(self) -> None:
        """尝试加载 ModelScope DFSMN 降噪 pipeline。"""
        if self._load_attempted:
            return
        self._load_attempted = True

        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
        except ImportError as exc:
            self.last_error = (
                "modelscope 未安装，降噪不可用。安装: pip install modelscope[framework] speechbrain"
            )
            LOGGER.info(self.last_error)
            return

        try:
            LOGGER.info("loading DFSMN denoiser: %s", self.model_id)
            self._pipeline = pipeline(
                Tasks.acoustic_noise_suppression,
                model=self.model_id,
            )
            self.last_error = None
            LOGGER.info("DFSMN denoiser loaded (native %dHz)", _DFSMN_NATIVE_SR)
        except Exception as exc:
            self.last_error = f"DFSMN denoiser failed to load: {exc}"
            LOGGER.warning(self.last_error, exc_info=True)

    @property
    def available(self) -> bool:
        """降噪模型是否已成功加载。"""
        self._load()
        return self._pipeline is not None

    def enhance(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """对一段音频做全段降噪。

        Args:
            audio: 输入音频，shape (n_samples,) float32.
            sample_rate: 采样率 (默认 16000).

        Returns:
            降噪后音频，同 shape 同 dtype。若降噪不可用则返回原始音频。
        """
        self._load()
        if self._pipeline is None:
            return np.asarray(audio, dtype=np.float32)

        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size < sample_rate * 0.1:  # 短于 100ms 不降噪
            return arr.copy()

        try:
            # 重采样到 48kHz (DFSMN 原生采样率)
            arr_48k = resample(arr, sample_rate, _DFSMN_NATIVE_SR)

            # ModelScope pipeline 返回: {"output_pcm": np.ndarray}
            # 需要先写成临时 WAV 再处理（pipeline 接受文件路径或 PCM bytes）
            import wave
            import tempfile
            from pathlib import Path

            # 将 PCM 写入临时 WAV 文件
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp_in:
                with wave.open(tmp_in, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(_DFSMN_NATIVE_SR)
                    # 归一化到 [-1, 1]，转为 int16
                    peak = max(np.abs(arr_48k).max(), 1e-10)
                    pcm_16 = (arr_48k / peak * 32767).astype(np.int16)
                    wf.writeframes(pcm_16.tobytes())
                tmp_in_path = Path(tmp_in.name)

            try:
                result = self._pipeline(str(tmp_in_path))
            finally:
                tmp_in_path.unlink(missing_ok=True)

            # 提取输出
            if isinstance(result, dict):
                output = result.get("output_pcm", result.get("output", None))
            elif isinstance(result, (list, tuple)) and len(result) > 0:
                output = result[0]
            else:
                output = result

            if output is None:
                return arr.copy()

            # 输出可能是 np.ndarray (float32 PCM)、文件路径、或 bytes (int16 PCM)
            if isinstance(output, (str, Path)):
                output_arr, file_sr = read_wav(Path(output))
                if file_sr != _DFSMN_NATIVE_SR:
                    output_arr = resample(output_arr, file_sr, _DFSMN_NATIVE_SR)
            elif isinstance(output, np.ndarray):
                output_arr = np.asarray(output, dtype=np.float32).reshape(-1)
            elif isinstance(output, (bytes, bytearray, memoryview)):
                # ModelScope ANS pipeline 以 int16 LE PCM bytes 返回 (native 48k),
                # 之前未解析 bytes 直接走 else 返回原音频 → 降噪形同虚设。
                output_arr = (
                    np.frombuffer(output, dtype="<i2").astype(np.float32) / 32768.0
                )
            else:
                return arr.copy()

            # 重采样回原始采样率
            if len(output_arr) == 0:
                return arr.copy()
            enhanced = resample(output_arr, _DFSMN_NATIVE_SR, sample_rate)

            # 对齐长度
            if len(enhanced) < len(arr):
                enhanced = np.pad(enhanced, (0, len(arr) - len(enhanced)))
            elif len(enhanced) > len(arr):
                enhanced = enhanced[: len(arr)]

            return enhanced.astype(np.float32)

        except Exception:
            LOGGER.debug("DFSMN enhance failed, fallback to original", exc_info=True)
            return arr.copy()

    def unload(self) -> None:
        """释放模型资源。"""
        import gc

        if self._pipeline is not None:
            try:
                del self._pipeline
            except Exception:
                pass
            self._pipeline = None
        self._load_attempted = False
        gc.collect()
        LOGGER.info("DFSMN denoiser unloaded")
