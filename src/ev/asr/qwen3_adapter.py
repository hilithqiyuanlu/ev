"""Qwen3-ASR 适配器 — 基于 Transformers 的中英混合 ASR。

支持模型: Qwen3-ASR-1.7B (中英混合标准版)

适配 transformers 5.x Qwen3-ASR API（chat-template 风格多模态调用）。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

_ASR_PROMPT = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n<|im_start|>assistant\n"
_ASR_TEXT_TAG = "<asr_text>"
_IM_END = "<|im_end|>"


class Qwen3ASRAdapter:
    """使用 Qwen3-ASR 模型进行语音识别。

    基于 transformers 5.x 的 Qwen3ASRProcessor + Qwen3ASRForConditionalGeneration。
    调用方式：chat template 风格（text 含 audio 占位符 + audio tensor），
    模型输出格式为：language <lang><asr_text>...<|im_end|>
    """

    def __init__(self, model_path: str, model: Any | None = None, tokenizer: Any | None = None):
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.model = model
        self.processor = None
        self.tokenizer = tokenizer
        self.device: str | None = None
        self.torch_dtype: Any | None = None
        self._suppress_tokens: list[int] | None = None
        self._loaded = model is not None and tokenizer is not None
        self._hotword_text_of_id: dict[int, str] | None = None
        self._hotword_ids_by_char: dict[str, list[int]] | None = None
        self._detect_variant()

    def _detect_variant(self) -> None:
        config_path = Path(self.model_path) / "config.json"
        if not config_path.exists():
            self._model_variant = "unknown"
            return
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            self._model_variant = "unknown"
            return
        self._model_variant = "1.7b"
        logger.info(
            "Qwen3-ASR loaded at %s", self.model_path,
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_model()
        self._loaded = True

    def _load_model(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Qwen3-ASR 需要安装 torch") from exc

        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Qwen3-ASR 需要安装 transformers>=5.0") from exc

        if torch.cuda.is_available():
            self.device = "cuda"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32

        logger.info(
            "Loading Qwen3-ASR (%s) from %s on %s (dtype=%s)",
            self._model_variant, self.model_path, self.device, self.torch_dtype,
        )

        attn_impl = None
        try:
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                attn_impl = "sdpa"
        except Exception:
            pass

        # transformers 5.x: 参数名从 torch_dtype 改为 dtype；
        # device_map="auto" 强制依赖 accelerate，仅在 CUDA+accelerate 可用时启用。
        load_kwargs: dict[str, Any] = {
            "dtype": self.torch_dtype,
            "trust_remote_code": True,
        }
        if self.device == "cuda":
            try:
                import accelerate  # noqa: F401
                load_kwargs["device_map"] = "auto"
                logger.info("Qwen3-ASR: using device_map='auto' (CUDA with accelerate)")
            except ImportError:
                logger.info("Qwen3-ASR: accelerate not installed, loading without device_map")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        if "device_map" in load_kwargs:
            load_kwargs["low_cpu_mem_usage"] = True

        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self.tokenizer = self.processor.tokenizer

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            **load_kwargs,
        )
        if "device_map" not in load_kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

        # Build suppress list: prevent audio placeholder tokens from being generated
        audio_tokens = [
            self._get_token_id("<|audio_pad|>"),
            self._get_token_id("<|audio_start|>"),
            self._get_token_id("<|audio_end|>"),
        ]
        self._suppress_tokens = [t for t in audio_tokens if t is not None]

        logger.info("Qwen3-ASR loaded successfully.")

    def _get_token_id(self, token: str) -> int | None:
        try:
            ids = self.tokenizer.convert_tokens_to_ids(token)
            return ids if ids != self.tokenizer.unk_token_id else None
        except Exception:
            return None

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        hotword: str = "",
        return_timestamps: bool = False,
        hotword_entries: list[tuple[str, float]] | None = None,
        enable_hotword_boost: bool = True,
        hotword_boost_scale: float = 2.0,
        hotword_boost_max: float = 6.0,
        hotword_min_anchor_len: int = 1,
    ) -> TranscriptionResult:
        import torch

        self._ensure_loaded()

        audio_arr = np.asarray(audio, dtype=np.float32)
        if audio_arr.ndim > 1:
            audio_arr = audio_arr.mean(axis=-1)
        if sample_rate != 16000:
            audio_arr = self._resample_np(audio_arr, sample_rate, 16000)
            sample_rate = 16000

        # Build prompt with hotword injection if provided
        prompt = _ASR_PROMPT
        if hotword and hotword.strip():
            words = [w.strip() for w in hotword.replace("\n", " ").split() if w.strip()]
            if words:
                hint = "，".join(words)
                prompt = (
                    "<|im_start|>system\n<|im_end|>\n"
                    f"<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|>"
                    f"请准确转写，注意以下词：{hint}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

        inputs = self.processor(
            text=prompt,
            audio=audio_arr,
            sampling_rate=16000,
            return_tensors="pt",
        )

        input_len = inputs["input_ids"].shape[1]

        # Move to device with correct dtype
        model_inputs: dict[str, Any] = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if v.dtype in (torch.float32, torch.float64):
                    model_inputs[k] = v.to(self.device, dtype=self.torch_dtype)
                else:
                    model_inputs[k] = v.to(self.device)
            else:
                model_inputs[k] = v

        # Token budget scales with audio duration: Chinese speech ~3-4 chars/sec
        # and ~1 token per char, so a fixed cap truncates long recordings mid-utterance.
        # Clamp to [256, 1024]; 256 floor keeps short-segment latency low.
        duration_sec = len(audio_arr) / float(sample_rate)
        max_new_tokens = max(256, min(1024, int(duration_sec * 20)))

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        # Only materialize scores when hotword hallucination detection is active
        # (output_scores stores [1, vocab_size] per step; up to ~150MB for 256 tokens)
        _compute_scores = enable_hotword_boost and bool(hotword_entries)
        if _compute_scores:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
        if self._suppress_tokens:
            gen_kwargs["suppress_tokens"] = self._suppress_tokens

        boost = self._build_hotword_processor(
            hotword_entries,
            enable_hotword_boost,
            hotword_boost_scale,
            hotword_boost_max,
            hotword_min_anchor_len,
        )
        if boost is not None:
            try:
                from transformers import LogitsProcessorList

                gen_kwargs["logits_processor"] = LogitsProcessorList([boost])
            except ImportError:
                logger.warning("transformers not importable; hotword logits boost disabled")

        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **gen_kwargs)

        if _compute_scores:
            new_tokens = outputs.sequences[0][input_len:]
        else:
            new_tokens = outputs[0][input_len:]
        if len(new_tokens) >= max_new_tokens:
            logger.warning(
                "Qwen3-ASR output truncated at %d tokens (audio=%.1fs, variant=%s); "
                "consider raising the token budget for long recordings",
                max_new_tokens, duration_sec, self._model_variant,
            )

        # Compute per-token log probabilities for hallucination detection
        avg_logprob: float | None = None
        if hasattr(outputs, "scores") and outputs.scores and len(new_tokens) > 0:
            import torch.nn.functional as F
            log_probs: list[float] = []
            for i, token_id in enumerate(new_tokens):
                if i >= len(outputs.scores):
                    break
                scores = outputs.scores[i][0].float()  # [vocab_size]
                lp = F.log_softmax(scores, dim=-1)[token_id].item()
                log_probs.append(lp)
            if log_probs:
                avg_logprob = float(sum(log_probs) / len(log_probs))

        raw_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)

        result = self._parse_output(raw_text, return_timestamps)
        result.avg_logprob = avg_logprob
        return result

    def _build_hotword_processor(
        self,
        hotword_entries: list[tuple[str, float]] | None,
        enable: bool,
        boost_scale: float,
        boost_max: float,
        min_anchor_len: int,
    ):
        """Build the anchored hotword logits processor, or None when inapplicable.

        Reuses precomputed tokenizer index caches so the (one-time) full-vocab
        scan in ``build_character_index`` is shared across transcribes.
        """
        from .hotword import HotwordLogitsProcessor

        if not enable or not hotword_entries:
            return None
        clean = [(str(w).strip(), float(weight)) for w, weight in hotword_entries if str(w).strip()]
        if not clean:
            return None
        processor = HotwordLogitsProcessor(
            self.tokenizer,
            clean,
            boost_scale=boost_scale,
            boost_max=boost_max,
            min_anchor_len=min_anchor_len,
            text_of_id=self._hotword_text_of_id,
            ids_by_char=self._hotword_ids_by_char,
        )
        processor.ensure_index()
        self._hotword_text_of_id = processor.text_of_id
        self._hotword_ids_by_char = processor.ids_by_char
        return processor

    def _parse_output(
        self, raw_text: str, with_timestamps: bool,
    ) -> TranscriptionResult:
        # Expected format: language <lang><asr_text>...<|im_end|>
        # For long audio the model may emit MULTIPLE <asr_text>...</asr_text>
        # blocks; collect all of them (old code kept only the first, dropping
        # everything after it).
        text = raw_text

        # Strip language tag
        lang_match = re.match(r"language\s+\S+\s*", text)
        if lang_match:
            text = text[lang_match.end():]

        if _ASR_TEXT_TAG in text:
            # Collect content from every <asr_text>...</asr_text> block.
            parts: list[str] = []
            pos = 0
            while True:
                start = text.find(_ASR_TEXT_TAG, pos)
                if start < 0:
                    break
                start += len(_ASR_TEXT_TAG)
                end = text.find(_IM_END, start)
                if end < 0:
                    # No closing tag: take up to the next block or end of text
                    nxt = text.find(_ASR_TEXT_TAG, start)
                    end = nxt if nxt >= 0 else len(text)
                chunk = text[start:end]
                if chunk.strip():
                    parts.append(chunk)
                pos = max(end, start)
            text = "".join(parts)
        else:
            # No <asr_text> tag (malformed output): fall back to old behavior.
            im_end_pos = text.find(_IM_END)
            if im_end_pos >= 0:
                text = text[:im_end_pos]

        text = self._clean_text(text)

        if not text.strip():
            return TranscriptionResult(text="")

        if with_timestamps:
            # Attempt to parse timestamp tokens (Qwen3-ASR uses <|x.xx|> format)
            segments = self._parse_timestamp_segments(raw_text)
            if segments:
                cleaned_segments = [
                    TranscriptionSegment(
                        text=self._clean_text(s.text),
                        start_ms=s.start_ms,
                        end_ms=s.end_ms,
                    )
                    for s in segments
                    if self._clean_text(s.text).strip()
                ]
                if cleaned_segments:
                    full = "".join(s.text for s in cleaned_segments).strip()
                    return TranscriptionResult(text=full or text, segments=cleaned_segments)

        return TranscriptionResult(text=text)

    def _parse_timestamp_segments(self, raw_text: str) -> list[TranscriptionSegment]:
        matches = list(re.finditer(r"<\|(\d+\.?\d*)\|>", raw_text))
        if len(matches) < 2:
            return []

        segments: list[TranscriptionSegment] = []
        i = 0
        while i < len(matches) - 1:
            start_match = matches[i]
            end_match = matches[i + 1]
            try:
                start_sec = float(start_match.group(1))
                end_sec = float(end_match.group(1))
            except ValueError:
                i += 1
                continue

            text_between = raw_text[start_match.end():end_match.start()]
            start_ms = int(round(start_sec * 1000))
            end_ms = int(round(end_sec * 1000))

            if end_ms > start_ms and text_between.strip():
                # Strip ASR tags from segment text
                seg_text = text_between
                for tag in (_ASR_TEXT_TAG, _IM_END, "language", "None", "zh", "en"):
                    seg_text = seg_text.replace(tag, "")
                seg_text = self._clean_text(seg_text)
                if seg_text.strip():
                    segments.append(TranscriptionSegment(
                        text=seg_text,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    ))
            i += 1

        return segments

    @staticmethod
    def _resample_np(audio: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
        if orig_rate == target_rate:
            return audio
        duration = len(audio) / orig_rate
        target_length = int(duration * target_rate)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _clean_text(text: str) -> str:
        text = re.sub(r"<\|[^|]*\|>", "", text)
        text = re.sub(r"<asr_text>", "", text)
        text = re.sub(r"\(\d+\.?\d*-\d+\.?\d*\)", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def reset(self) -> None:
        pass

    def unload(self) -> None:
        """Release model and processor resources. Safe to call multiple times."""
        import gc

        if self.model is not None:
            try:
                import torch
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            except ImportError:
                pass
            try:
                del self.model
            except Exception:
                pass
            self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None
        self.torch_dtype = None
        self._suppress_tokens = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
