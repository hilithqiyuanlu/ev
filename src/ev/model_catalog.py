"""模型静态目录定义 — 所有可用模型的元数据。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelType(str, Enum):
    VAD = "vad"
    ASR_FINAL = "asr_final"
    SPEAKER = "speaker"
    SPEECH_ENHANCEMENT = "speech_enhancement"
    ENVIRONMENT = "environment"


class ModelSource(str, Enum):
    MODELSCOPE = "modelscope"
    GITHUB = "github"
    LOCAL = "local"


@dataclass(frozen=True)
class ModelDefinition:
    key: str
    name: str
    type: ModelType
    source: ModelSource
    default_dirname: str
    description: str = ""
    modelscope_id: str | None = None
    github_asset_key: str | None = None
    github_url: str | None = None
    github_filename: str | None = None
    github_size: int = 0
    github_sha256: str | None = None
    estimated_size_bytes: int = 0
    min_memory_gb: float = 0.0
    needs_tokens: bool = False
    needs_seg_dict: bool = False
    config_filenames: tuple[str, ...] = ("config.yaml", "configuration.json", "config.json")
    weight_suffixes: tuple[str, ...] = (".pt", ".pth", ".bin", ".safetensors", ".onnx", ".ckpt")


# ──────────────────────────────────────────────────────────────────────
# 内置默认模型目录
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_CATALOG: tuple[ModelDefinition, ...] = (
    ModelDefinition(
        key="fsmn-vad",
        name="FSMN-VAD",
        type=ModelType.VAD,
        source=ModelSource.MODELSCOPE,
        default_dirname="ev-fsmn-vad-zh-16k",
        description="FSMN 语音活动检测模型，中文",
        modelscope_id="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        estimated_size_bytes=2_500_000,
    ),
    ModelDefinition(
        key="sensevoice-small",
        name="SenseVoice Small",
        type=ModelType.ASR_FINAL,
        source=ModelSource.GITHUB,
        default_dirname="ev-sensevoice-small",
        description="SenseVoice Small 多语言自动语音识别，支持中英日韩粤",
        github_asset_key="asr_final",
        github_url="https://github.com/hilithqiyuanlu/ev/releases/download/models-v0.1.0/ev-sensevoice-small.tar.gz",
        github_filename="ev-sensevoice-small.tar.gz",
        github_size=870_507_697,
        github_sha256="f61dcfc2fd855e25c1990eda841113e45d6521bfbd115a9bdd559106554c03cb",
        needs_tokens=True,
    ),
    ModelDefinition(
        key="qwen3-asr-1.7b",
        name="Qwen3-ASR 1.7B",
        type=ModelType.ASR_FINAL,
        source=ModelSource.MODELSCOPE,
        default_dirname="qwen3-asr-1.7b",
        description="通义 Qwen3-ASR 标准版，SOTA 开源 ASR，支持词级时间戳",
        modelscope_id="Qwen/Qwen3-ASR-1.7B-hf",
        estimated_size_bytes=4_087_000_000,
        min_memory_gb=4.0,
    ),
    ModelDefinition(
        key="eres2netv2",
        name="ERes2NetV2",
        type=ModelType.SPEAKER,
        source=ModelSource.GITHUB,
        default_dirname="ev-eres2netv2-zh-16k",
        description="声纹识别模型",
        github_asset_key="speaker",
        github_url="https://github.com/hilithqiyuanlu/ev/releases/download/models-v0.1.0/ev-eres2netv2-zh-16k.tar.gz",
        github_filename="ev-eres2netv2-zh-16k.tar.gz",
        github_size=68820465,
        github_sha256="a921151bb77ff72221a1b39759f12ec8ee891e167d5cc43932d350eb80dc5d3c",
    ),
    ModelDefinition(
        key="dfsmn-ans",
        name="DFSMN-ANS",
        type=ModelType.SPEECH_ENHANCEMENT,
        source=ModelSource.MODELSCOPE,
        default_dirname="ev-dfsmn-ans-zh-16k",
        description="DFSMN 语音降噪模型（FunASR 前端增强）",
        modelscope_id="damo/speech_dfsmn_ans_psm_48k_causal",
        estimated_size_bytes=20_000_000,
    ),
    ModelDefinition(
        key="yamnet",
        name="YAMNet",
        type=ModelType.ENVIRONMENT,
        source=ModelSource.GITHUB,
        default_dirname="yamnet",
        description="环境声音分类模型（AudioSet 521 类），独立于语音路径",
        github_asset_key="yamnet",
        github_url="https://github.com/hilithqiyuanlu/ev/releases/download/models-v0.1.0/ev-yamnet-16k.tar.gz",
        github_filename="ev-yamnet-16k.tar.gz",
        github_size=3_239_056,
        github_sha256="d19fae6afc9a05537cf7960c6b038eb247c212378909833849e43c10d61203ee",
        estimated_size_bytes=4_500_000,
        config_filenames=("yamnet_class_map.csv",),
        weight_suffixes=(".tflite",),
    ),
)

# 槽位默认分配（5 槽，流式 ASR 已移除）
_DEFAULT_SLOTS: dict[str, str] = {
    "vad": "fsmn-vad",
    "speech_enhancement": "dfsmn-ans",
    "asr_final": "sensevoice-small",
    "speaker": "eres2netv2",
    "environment": "yamnet",
}

_ALL_SLOTS: tuple[str, ...] = tuple(_DEFAULT_SLOTS.keys())


def get_catalog() -> dict[str, ModelDefinition]:
    """返回所有可用模型定义，key→definition 映射。"""
    return {m.key: m for m in _DEFAULT_CATALOG}


def get_definition(key: str) -> ModelDefinition | None:
    """按 key 查询模型定义，不存在返回 None。"""
    return next((m for m in _DEFAULT_CATALOG if m.key == key), None)


def get_definitions_for_type(model_type: ModelType) -> tuple[ModelDefinition, ...]:
    """按类型筛选模型定义。"""
    return tuple(m for m in _DEFAULT_CATALOG if m.type == model_type)


def get_default_slot(slot: str) -> str | None:
    """获取某槽位的默认模型 key。"""
    return _DEFAULT_SLOTS.get(slot)


def get_all_slots() -> tuple[str, ...]:
    """返回所有槽位名。"""
    return _ALL_SLOTS
