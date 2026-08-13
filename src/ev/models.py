"""本地模型校验 — 同时支持旧配置和新注册表。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ModelSettings
from .model_catalog import (
    ModelDefinition,
    get_all_slots,
    get_definition,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    dirname: str
    needs_tokens: bool = False
    needs_seg_dict: bool = False


@dataclass(frozen=True)
class ModelCheck:
    key: str
    path: Path
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


_CONFIG_NAMES = ("config.yaml", "configuration.json", "config.json")
_WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors", ".onnx", ".ckpt")


# ── 旧 API 兼容层 [DEPRECATED] ─────────────────────────────────────────
# 以下函数仅保留用于向后兼容。新代码应使用 ModelRegistry。
# 计划在 v0.3.0 移除。


def specs(settings: ModelSettings) -> tuple[ModelSpec, ...]:
    """从旧 ModelSettings (3硬编码字段) 生成 spec 列表。"""
    return (
        ModelSpec("vad", settings.vad),
        ModelSpec("asr_final", settings.asr_final, True),
        ModelSpec("speaker", settings.speaker),
    )


def resolve_model_paths(
    settings: ModelSettings, root: Path | None = None
) -> dict[str, Path]:
    """从旧 ModelSettings 解析模型路径。"""
    base = (root or settings.root).expanduser().resolve()
    return {spec.key: base / spec.dirname for spec in specs(settings)}


def verify_models(
    settings: ModelSettings, root: Path | None = None,
    skip_keys: frozenset[str] = frozenset(),
) -> tuple[ModelCheck, ...]:
    """旧 API — 校验硬编码的4个模型。保留用于向后兼容。

    skip_keys: 跳过指定 key 的校验（如注册表管理的 asr_final 槽位）。
    """
    paths = resolve_model_paths(settings, root)
    checks: list[ModelCheck] = []
    for spec in specs(settings):
        if spec.key in skip_keys:
            continue
        path = paths[spec.key]
        errors: list[str] = []
        if not path.is_dir():
            errors.append("目录不存在")
        else:
            if not _has_named_file(path, _CONFIG_NAMES):
                errors.append("缺少模型配置文件")
            if not _has_nonempty_weight(path):
                errors.append("缺少非空权重文件")
            if spec.needs_tokens and not _has_named_file(path, ("tokens.json",)):
                errors.append("缺少 tokens.json")
            if spec.needs_seg_dict and not any(
                item.is_file() and "seg_dict" in item.name for item in path.rglob("*")
            ):
                errors.append("缺少 seg_dict")
        checks.append(ModelCheck(spec.key, path, tuple(errors)))
    return tuple(checks)


def require_models(
    settings: ModelSettings, root: Path | None = None,
    skip_keys: frozenset[str] = frozenset(),
) -> dict[str, Path]:
    """旧 API — 要求模型全部就绪（可跳过指定 key）。"""
    checks = verify_models(settings, root, skip_keys=skip_keys)
    failed = [f"{item.key}: {', '.join(item.errors)} ({item.path})" for item in checks if not item.ok]
    if failed:
        raise RuntimeError("本地模型校验失败:\n" + "\n".join(failed))
    return {item.key: item.path for item in checks}


# ── 新 Registry 驱动 API ──────────────────────────────────────────────


def verify_definition(
    path: Path, definition: ModelDefinition
) -> tuple[str, ...]:
    """根据 ModelDefinition 校验本地模型目录。"""
    errors: list[str] = []
    if not path.is_dir():
        return ("目录不存在",)

    if not _has_named_file(path, definition.config_filenames):
        errors.append("缺少模型配置文件")
    if not _has_nonempty_weight(path, definition.weight_suffixes):
        errors.append("缺少非空权重文件")
    if definition.needs_tokens and not _has_named_file(path, ("tokens.json",)):
        errors.append("缺少 tokens.json")
    if definition.needs_seg_dict and not any(
        item.is_file() and "seg_dict" in item.name for item in path.rglob("*")
    ):
        errors.append("缺少 seg_dict")
    return tuple(errors)


def verify_all_definitions(
    assignments: dict[str, tuple[str, str]],
    models_root: Path,
) -> tuple[ModelCheck, ...]:
    """批量校验：assignments = {slot: (model_key, local_path_str)}"""
    checks: list[ModelCheck] = []
    for slot, (model_key, local_path) in assignments.items():
        path = Path(local_path)
        definition = get_definition(model_key)
        if not definition:
            checks.append(ModelCheck(slot, path, ("未知模型",)))
            continue
        errors = verify_definition(path, definition)
        checks.append(ModelCheck(slot, path, errors))
    return tuple(checks)


# ── 辅助函数 ──────────────────────────────────────────────────────────


def _has_named_file(path: Path, names: tuple[str, ...]) -> bool:
    return any(item.name in names and item.is_file() for item in path.rglob("*"))


def _has_nonempty_weight(path: Path, suffixes: tuple[str, ...] | None = None) -> bool:
    suffixes = suffixes or _WEIGHT_SUFFIXES
    return any(
        item.is_file() and item.suffix.lower() in suffixes and item.stat().st_size > 0
        for item in path.rglob("*")
    )
