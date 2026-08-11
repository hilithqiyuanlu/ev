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
    """帧级音频预处理参数. 全部有默认值, toml 缺省时用默认 (全部开启).

    默认值按远场友好调校 (外出场景, 3m 轻中声能稳定录入).
    """

    enabled: bool = True
    # Preemphasis
    preemphasis_coeff: float = 0.97
    # AGC (远场: 更高目标响度 + 更大增益上限 + 更慢 release 防句中呼吸)
    agc_target_rms: float = 0.08
    agc_min_gain: float = 0.1       # -20dB
    agc_max_gain: float = 40.0      # +32dB
    agc_attack_ms: float = 10.0     # 增益下降 (压大音量) 快
    agc_release_ms: float = 400.0   # 增益上升 (放大小音量) 慢
    # NoiseGate (远场 SNR 本就低, 门槛调松)
    noisegate_enabled: bool = True
    noisegate_snr_db: float = 1.5
    noisegate_floor_track_sec: float = 3.0


@dataclass(frozen=True)
class VADSettings:
    """VAD 组合策略与兜底能量 VAD 参数."""

    # FSMN model threshold (若模型支持该参数; None = 模型默认)
    fsmn_threshold: float | None = None
    # Energy VAD fallback (远场: 更低绝对底噪 + 更松 SNR)
    energy_vad_enabled: bool = True
    energy_snr_linear: float = 1.8      # RMS > floor * 1.8x (≈2.5dB)
    energy_abs_min_rms: float = 0.0003  # ~-70dBFS
    energy_start_frames: int = 2        # 60ms @ 30ms/帧
    energy_hangover_frames: int = 20    # 600ms @ 30ms/帧
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
class ModelSlotConfig:
    """单个槽位的动态配置。"""
    model_key: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class ModelRegistrySettings:
    """注册表模型配置 — 支持动态槽位分配。"""
    root: Path
    slots: dict[str, ModelSlotConfig]


@dataclass(frozen=True)
class SpeakerSettings:
    threshold: float = 0.40
    max_core_samples: int = 30
    max_cache_samples: int = 100
    max_centroids: int = 5
    loudness_normalize: bool = True


@dataclass(frozen=True)
class AsrSettings:
    """Hotword boosting for the final ASR (Qwen3).

    设计原则: 只做"锚定式正增量", 尽量少干扰常见表述。
    - 仅当已解码文本末尾已命中词典词前缀时, 才对续写该词的 token 加 logits;
    - 从不抑制任何 token, 匹配常见前缀但不能续写热词时完全不受影响。
    """

    hotword_boosting_enabled: bool = True
    hotword_boost_scale: float = 2.0    # 每个权重单位叠加的 logits 增量
    hotword_boost_max: float = 4.0      # 单 token 叠加上限 (6→4, 防弱信号幻觉)
    hotword_min_anchor_len: int = 1     # 触发所需的最少已匹配前缀字符数
    hotword_inject_max_words: int = 30  # prompt 注入的最大词数


@dataclass(frozen=True)
class VuiSettings:
    wake_words: tuple[str, ...] = ("小E",)


@dataclass(frozen=True)
class SegmentSettings:
    min_duration_ms: int = 500
    max_duration_ms: int = 20000        # 硬上限: 超过20s强制切分
    silence_timeout_ms: int = 1600      # 尾部绝对静音超时: 1.6s
    silence_rms_threshold: float = 0.003  # 绝对静音RMS阈值（raw音频，~-50dBFS）
    # 相对静音检测: RMS从说话峰值下降到该比例以下视为静音
    relative_silence_ratio: float = 0.30  # 峰值的30%以下
    relative_silence_timeout_ms: int = 1900  # 相对静音持续1.9s强制切分
    # 静音类触发器的最小段长门槛: 短段(刚开始说话)不在静音时被过早切掉
    min_duration_for_silence_ms: int = 3000   # silence_timeout/energy_silent 生效所需最小段长
    min_duration_for_relative_silence_ms: int = 6000  # relative_silence 生效所需最小段长
    # ASR停滞超时: 流式ASR长时间无新partial结果视为说完
    asr_stall_timeout_ms: int = 2500
    discard_filler_only: bool = True
    # 音频质量门控 (宽松起步, 数据驱动调优)
    min_snr_db: float = 3.0            # 段 SNR 低于此值 → 质量拒绝 (宽松起步)
    min_audible_rms: float = 0.0005    # 段 raw RMS 低于此值 → 质量拒绝 (~-66dBFS)
    raw_noise_warmup_sec: float = 3.0   # 底噪追踪器启动后前 N 秒不拒绝


@dataclass(frozen=True)
class VoiceLearningSettings:
    auto_learn_enabled: bool = True
    max_samples: int = 20
    ema_alpha: float = 0.05
    collect_threshold_offset: float = 0.05
    collect_min_score: float = 0.40
    core_score_min: float = 0.70
    pending_distance_threshold: float = 0.30
    promote_min_members: int = 2
    promote_cooldown_sec: float = 60.0
    onboarding_target: int = 5
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
    model_registry: ModelRegistrySettings
    speaker: SpeakerSettings
    vui: VuiSettings
    segment: SegmentSettings
    voice_learning: VoiceLearningSettings
    asr: AsrSettings

    @property
    def models_dir(self) -> Path:
        return self.models.root

    @property
    def archive_dir(self) -> Path:
        return self.data_dir / "archive"

    @property
    def voice_samples_dir(self) -> Path:
        return self.data_dir / "voice-samples"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ev.db"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.models_dir,
            self.archive_dir,
            self.voice_samples_dir,
            self.logs_dir,
        ):
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
    asr_raw = raw.get("asr", {})
    data_dir = Path(
        os.environ.get("EV_DATA_DIR", raw.get("paths", {}).get("data_dir", "data"))
    )
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    model_root = Path(os.environ.get("EV_MODEL_ROOT", models_raw.get("root", "data/models")))
    if not model_root.is_absolute():
        model_root = PROJECT_ROOT / model_root

    # 旧格式兼容：4 硬编码字段
    old_model_settings = ModelSettings(
        root=model_root,
        vad=str(models_raw.get("vad", "ev-fsmn-vad-zh-16k")),
        asr_streaming=str(models_raw.get("asr_streaming", "ev-paraformer-zh-streaming-16k")),
        asr_final=str(models_raw.get("asr_final", "ev-paraformer-zh-16k")),
        speaker=str(models_raw.get("speaker", "ev-eres2netv2-zh-16k")),
    )

    # 新格式：动态槽位（从 toml 或默认值构建）
    slots_raw = models_raw.get("slots", {})
    default_slots = {
        "vad": "fsmn-vad",
        "asr_streaming": "paraformer-zh-streaming",
        "asr_final": "qwen3-asr-1.7b",
        "speaker": "eres2netv2",
    }
    slot_configs: dict[str, ModelSlotConfig] = {}
    for slot, default_key in default_slots.items():
        slot_data = slots_raw.get(slot, {})
        if isinstance(slot_data, str):
            model_key = slot_data
            enabled = True
        elif isinstance(slot_data, dict):
            model_key = slot_data.get("model_key", default_key)
            enabled = slot_data.get("enabled", True)
        else:
            model_key = default_key
            enabled = True
        slot_configs[slot] = ModelSlotConfig(
            model_key=model_key if model_key else None,
            enabled=bool(enabled),
        )

    registry_settings = ModelRegistrySettings(
        root=model_root,
        slots=slot_configs,
    )

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
            agc_target_rms=float(preprocess_raw.get("agc_target_rms", 0.08)),
            agc_min_gain=float(preprocess_raw.get("agc_min_gain", 0.1)),
            agc_max_gain=float(preprocess_raw.get("agc_max_gain", 40.0)),
            agc_attack_ms=float(preprocess_raw.get("agc_attack_ms", 10.0)),
            agc_release_ms=float(preprocess_raw.get("agc_release_ms", 400.0)),
            noisegate_enabled=bool(preprocess_raw.get("noisegate_enabled", True)),
            noisegate_snr_db=float(preprocess_raw.get("noisegate_snr_db", 1.5)),
            noisegate_floor_track_sec=float(preprocess_raw.get("noisegate_floor_track_sec", 3.0)),
        ),
        vad=VADSettings(
            fsmn_threshold=fsmn_threshold,
            energy_vad_enabled=bool(vad_raw.get("energy_vad_enabled", True)),
            energy_snr_linear=float(vad_raw.get("energy_snr_linear", 1.8)),
            energy_abs_min_rms=float(vad_raw.get("energy_abs_min_rms", 0.0003)),
            energy_start_frames=int(vad_raw.get("energy_start_frames", 2)),
            energy_hangover_frames=int(vad_raw.get("energy_hangover_frames", 20)),
            combine_start_mode=str(vad_raw.get("combine_start_mode", "or")).lower(),
            combine_end_mode=str(vad_raw.get("combine_end_mode", "and")).lower(),
        ),
        data_dir=data_dir,
        models=old_model_settings,
        model_registry=registry_settings,
        speaker=SpeakerSettings(
            threshold=float(
                speaker_raw.get("threshold",
                    speaker_raw.get("user_threshold", 0.40)
                )
            ),
            max_core_samples=int(speaker_raw.get("max_core_samples", 30)),
            max_cache_samples=int(speaker_raw.get("max_cache_samples", 100)),
            max_centroids=int(speaker_raw.get("max_centroids", 5)),
            loudness_normalize=bool(speaker_raw.get("loudness_normalize", True)),
        ),
        vui=VuiSettings(tuple(str(x) for x in vui_raw.get("wake_words", ["小E"]))),
        segment=SegmentSettings(
            min_duration_ms=int(segment_raw.get("min_duration_ms", 500)),
            max_duration_ms=int(segment_raw.get("max_duration_ms", 20000)),
            silence_timeout_ms=int(segment_raw.get("silence_timeout_ms", 1600)),
            silence_rms_threshold=float(segment_raw.get("silence_rms_threshold", 0.003)),
            relative_silence_ratio=float(segment_raw.get("relative_silence_ratio", 0.30)),
            relative_silence_timeout_ms=int(segment_raw.get("relative_silence_timeout_ms", 1900)),
            min_duration_for_silence_ms=int(segment_raw.get("min_duration_for_silence_ms", 3000)),
            min_duration_for_relative_silence_ms=int(segment_raw.get("min_duration_for_relative_silence_ms", 6000)),
            asr_stall_timeout_ms=int(segment_raw.get("asr_stall_timeout_ms", 2500)),
            discard_filler_only=bool(segment_raw.get("discard_filler_only", True)),
            min_snr_db=float(segment_raw.get("min_snr_db", 3.0)),
            min_audible_rms=float(segment_raw.get("min_audible_rms", 0.0005)),
            raw_noise_warmup_sec=float(segment_raw.get("raw_noise_warmup_sec", 3.0)),
        ),
        voice_learning=VoiceLearningSettings(
            auto_learn_enabled=bool(voice_learning_raw.get("auto_learn_enabled", True)),
            max_samples=int(voice_learning_raw.get("max_samples", 20)),
            ema_alpha=float(voice_learning_raw.get("ema_alpha", 0.05)),
            collect_threshold_offset=float(voice_learning_raw.get("collect_threshold_offset", 0.05)),
            collect_min_score=float(voice_learning_raw.get("collect_min_score", 0.40)),
            core_score_min=float(voice_learning_raw.get("core_score_min", 0.70)),
            pending_distance_threshold=float(voice_learning_raw.get("pending_distance_threshold", 0.30)),
            promote_min_members=int(voice_learning_raw.get("promote_min_members", 2)),
            promote_cooldown_sec=float(voice_learning_raw.get("promote_cooldown_sec", 60.0)),
            onboarding_target=int(voice_learning_raw.get("onboarding_target", 5)),
            min_duration_ms=int(voice_learning_raw.get("min_duration_ms", 1500)),
            max_duration_ms=int(voice_learning_raw.get("max_duration_ms", 10000)),
            min_interval_sec=float(voice_learning_raw.get("min_interval_sec", 30.0)),
        ),
        asr=AsrSettings(
            hotword_boosting_enabled=bool(asr_raw.get("hotword_boosting_enabled", True)),
            hotword_boost_scale=float(asr_raw.get("hotword_boost_scale", 2.0)),
            hotword_boost_max=float(asr_raw.get("hotword_boost_max", 4.0)),
            hotword_min_anchor_len=int(asr_raw.get("hotword_min_anchor_len", 1)),
            hotword_inject_max_words=int(asr_raw.get("hotword_inject_max_words", 30)),
        ),
    )
