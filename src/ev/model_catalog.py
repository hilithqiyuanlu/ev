"""模型静态目录定义 — 所有可用模型的元数据。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ModelType(str, Enum):
    VAD = "vad"
    ASR_STREAMING = "asr_streaming"
    ASR_FINAL = "asr_final"
    SPEAKER = "speaker"


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
        key="paraformer-zh-streaming",
        name="Paraformer Streaming",
        type=ModelType.ASR_STREAMING,
        source=ModelSource.GITHUB,
        default_dirname="ev-paraformer-zh-streaming-16k",
        description="FunASR Paraformer 中文流式识别",
        github_asset_key="asr_streaming",
        github_url="https://github.com/hilithqiyuanlu/ev/releases/download/models-v0.1.0/ev-paraformer-zh-streaming-16k.tar.gz",
        github_filename="ev-paraformer-zh-streaming-16k.tar.gz",
        github_size=820157164,
        github_sha256="4ff1d661db592f59dc869940fb1bec6db10f61f595cbcec0fd1a20d2cb4aebcb",
        needs_tokens=True,
        needs_seg_dict=True,
    ),
    ModelDefinition(
        key="paraformer-zh",
        name="Paraformer Large",
        type=ModelType.ASR_FINAL,
        source=ModelSource.GITHUB,
        default_dirname="ev-paraformer-zh-16k",
        description="FunASR Paraformer 中文非流式识别",
        github_asset_key="asr_final",
        github_url="https://github.com/hilithqiyuanlu/ev/releases/download/models-v0.1.0/ev-paraformer-zh-16k.tar.gz",
        github_filename="ev-paraformer-zh-16k.tar.gz",
        github_size=921180174,
        github_sha256="13b8c7b7ca8bdd15dc472885a9b9d3009e2659728098188a70f50b0b375193c9",
        needs_tokens=True,
    ),
    ModelDefinition(
        key="qwen3-asr-0.6b",
        name="Qwen3-ASR 0.6B",
        type=ModelType.ASR_FINAL,
        source=ModelSource.MODELSCOPE,
        default_dirname="qwen3-asr-0.6b",
        description="通义 Qwen3-ASR 轻量版，支持中英混合识别、词级时间戳",
        modelscope_id="Qwen/Qwen3-ASR-0.6B-hf",
        estimated_size_bytes=1_576_000_000,
        min_memory_gb=2.0,
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
)

# 槽位默认分配
_DEFAULT_SLOTS: dict[str, str] = {
    "vad": "fsmn-vad",
    "asr_streaming": "paraformer-zh-streaming",
    "asr_final": "qwen3-asr-1.7b",
    "speaker": "eres2netv2",
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
