"""配置加载:ev.toml 默认值 + ev.local.toml 本地覆盖 + 环境变量(EV_ 前缀)。"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "ev.toml"
LOCAL_CONFIG = PROJECT_ROOT / "ev.local.toml"


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int = 16000
    channels: int = 1


@dataclass(frozen=True)
class PreprocessSettings:
    """帧级音频预处理参数. 全部有默认值, toml 缺省时用默认 (全部开启)."""

    enabled: bool = True
    # Preemphasis
    preemphasis_coeff: float = 0.97
    # AGC
    agc_target_rms: float = 0.05
    agc_min_gain: float = 0.1       # -20dB
    agc_max_gain: float = 20.0      # +26dB
    agc_attack_ms: float = 10.0     # 增益下降 (压大音量) 快
    agc_release_ms: float = 100.0   # 增益上升 (放大小音量) 慢
    # NoiseGate
    noisegate_enabled: bool = True
    noisegate_snr_db: float = 3.0
    noisegate_floor_track_sec: float = 3.0


@dataclass(frozen=True)
class VADSettings:
    """VAD 组合策略与兜底能量 VAD 参数."""

    # FSMN model threshold (若模型支持该参数; None = 模型默认)
    fsmn_threshold: float | None = None
    # Energy VAD fallback
    energy_vad_enabled: bool = True
    energy_snr_linear: float = 2.5     # RMS > floor * 2.5x (≈4dB)
    energy_abs_min_rms: float = 0.001  # ~-60dBFS
    energy_start_frames: int = 2       # 60ms @ 30ms/帧
    energy_hangover_frames: int = 20   # 600ms @ 30ms/帧
    # 复合策略: "or"/"fsmn_only"/"energy_only"
    combine_start_mode: str = "or"       # 启动放宽松: OR
    combine_end_mode: str = "and"        # 结束保守: AND + hangover


@dataclass(frozen=True)
class ModelSettings:
    root: Path
    vad: str
    asr_streaming: str
    asr_final: str
    speaker: str


@dataclass(frozen=True)
class SpeakerSettings:
    threshold: float = 0.50
    max_core_samples: int = 20
    max_cache_samples: int = 50
    max_centroids: int = 3
    loudness_normalize: bool = True


@dataclass(frozen=True)
class VuiSettings:
    wake_words: tuple[str, ...] = ("小E",)


@dataclass(frozen=True)
class SegmentSettings:
    min_duration_ms: int = 500
    discard_filler_only: bool = True


@dataclass(frozen=True)
class VoiceLearningSettings:
    auto_learn_enabled: bool = True
    max_samples: int = 20
    ema_alpha: float = 0.05
    collect_threshold_offset: float = 0.05
    collect_min_score: float = 0.40
    min_duration_ms: int = 1500
    max_duration_ms: int = 10000
    min_interval_sec: float = 30.0


@dataclass(frozen=True)
class Settings:
    log_level: str
    audio: AudioSettings
    preprocess: PreprocessSettings
    vad: VADSettings
    data_dir: Path
    models: ModelSettings
    speaker: SpeakerSettings
    vui: VuiSettings
    segment: SegmentSettings
    voice_learning: VoiceLearningSettings

    @property
    def models_dir(self) -> Path:
        return self.models.root

    @property
    def archive_dir(self) -> Path:
        return self.data_dir / "archive"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ev.db"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.archive_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(config_path: Path | None = None) -> Settings:
    raw: dict = {}
    for path in (config_path or DEFAULT_CONFIG, LOCAL_CONFIG):
        if path.exists():
            raw = _deep_merge(raw, tomllib.loads(path.read_text(encoding="utf-8")))

    audio_raw = raw.get("audio", {})
    preprocess_raw = raw.get("preprocess", {})
    vad_raw = raw.get("vad", {})
    models_raw = raw.get("models", {})
    speaker_raw = raw.get("speaker", {})
    vui_raw = raw.get("vui", {})
    segment_raw = raw.get("segment", {})
    voice_learning_raw = raw.get("voice_learning", {})
    data_dir = Path(
        os.environ.get("EV_DATA_DIR", raw.get("paths", {}).get("data_dir", "data"))
    )
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    model_root = Path(os.environ.get("EV_MODEL_ROOT", models_raw.get("root", "data/models")))
    if not model_root.is_absolute():
        model_root = PROJECT_ROOT / model_root

    # fsmn_threshold: 允许 toml 写 null / 数值
    _fsmn_th = vad_raw.get("fsmn_threshold", None)
    fsmn_threshold: float | None
    if _fsmn_th is None:
        fsmn_threshold = None
    else:
        try:
            fsmn_threshold = float(_fsmn_th)
        except (TypeError, ValueError):
            fsmn_threshold = None

    return Settings(
        log_level=os.environ.get("EV_LOG_LEVEL", raw.get("log_level", "INFO")).upper(),
        audio=AudioSettings(
            sample_rate=int(audio_raw.get("sample_rate", 16000)),
            channels=int(audio_raw.get("channels", 1)),
        ),
        preprocess=PreprocessSettings(
            enabled=bool(preprocess_raw.get("enabled", True)),
            preemphasis_coeff=float(preprocess_raw.get("preemphasis_coeff", 0.97)),
            agc_target_rms=float(preprocess_raw.get("agc_target_rms", 0.05)),
            agc_min_gain=float(preprocess_raw.get("agc_min_gain", 0.1)),
            agc_max_gain=float(preprocess_raw.get("agc_max_gain", 20.0)),
            agc_attack_ms=float(preprocess_raw.get("agc_attack_ms", 10.0)),
            agc_release_ms=float(preprocess_raw.get("agc_release_ms", 100.0)),
            noisegate_enabled=bool(preprocess_raw.get("noisegate_enabled", True)),
            noisegate_snr_db=float(preprocess_raw.get("noisegate_snr_db", 3.0)),
            noisegate_floor_track_sec=float(preprocess_raw.get("noisegate_floor_track_sec", 3.0)),
        ),
        vad=VADSettings(
            fsmn_threshold=fsmn_threshold,
            energy_vad_enabled=bool(vad_raw.get("energy_vad_enabled", True)),
            energy_snr_linear=float(vad_raw.get("energy_snr_linear", 2.5)),
            energy_abs_min_rms=float(vad_raw.get("energy_abs_min_rms", 0.001)),
            energy_start_frames=int(vad_raw.get("energy_start_frames", 2)),
            energy_hangover_frames=int(vad_raw.get("energy_hangover_frames", 20)),
            combine_start_mode=str(vad_raw.get("combine_start_mode", "or")).lower(),
            combine_end_mode=str(vad_raw.get("combine_end_mode", "and")).lower(),
        ),
        data_dir=data_dir,
        models=ModelSettings(
            root=model_root,
            vad=str(models_raw.get("vad", "ev-fsmn-vad-zh-16k")),
            asr_streaming=str(models_raw.get("asr_streaming", "ev-paraformer-zh-streaming-16k")),
            asr_final=str(models_raw.get("asr_final", "ev-paraformer-zh-16k")),
            speaker=str(models_raw.get("speaker", "ev-eres2netv2-zh-16k")),
        ),
        speaker=SpeakerSettings(
            threshold=float(
                speaker_raw.get("threshold",
                    speaker_raw.get("user_threshold", 0.50)
                )
            ),
            max_core_samples=int(speaker_raw.get("max_core_samples", 20)),
            max_cache_samples=int(speaker_raw.get("max_cache_samples", 50)),
            max_centroids=int(speaker_raw.get("max_centroids", 3)),
            loudness_normalize=bool(speaker_raw.get("loudness_normalize", True)),
        ),
        vui=VuiSettings(tuple(str(x) for x in vui_raw.get("wake_words", ["小E"]))),
        segment=SegmentSettings(
            min_duration_ms=int(segment_raw.get("min_duration_ms", 500)),
            discard_filler_only=bool(segment_raw.get("discard_filler_only", True)),
        ),
        voice_learning=VoiceLearningSettings(
            auto_learn_enabled=bool(voice_learning_raw.get("auto_learn_enabled", True)),
            max_samples=int(voice_learning_raw.get("max_samples", 20)),
            ema_alpha=float(voice_learning_raw.get("ema_alpha", 0.05)),
            collect_threshold_offset=float(voice_learning_raw.get("collect_threshold_offset", 0.05)),
            collect_min_score=float(voice_learning_raw.get("collect_min_score", 0.40)),
            min_duration_ms=int(voice_learning_raw.get("min_duration_ms", 1500)),
            max_duration_ms=int(voice_learning_raw.get("max_duration_ms", 10000)),
            min_interval_sec=float(voice_learning_raw.get("min_interval_sec", 30.0)),
        ),
    )
