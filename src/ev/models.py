"""本地模型目录解析与离线完整性检查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ModelSettings


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


def specs(settings: ModelSettings) -> tuple[ModelSpec, ...]:
    return (
        ModelSpec("vad", settings.vad),
        ModelSpec("asr_streaming", settings.asr_streaming, True, True),
        ModelSpec("asr_final", settings.asr_final, True),
        ModelSpec("speaker", settings.speaker),
    )


def resolve_model_paths(
    settings: ModelSettings, root: Path | None = None
) -> dict[str, Path]:
    base = (root or settings.root).expanduser().resolve()
    return {spec.key: base / spec.dirname for spec in specs(settings)}


def _has_named_file(path: Path, names: tuple[str, ...]) -> bool:
    return any(item.name in names and item.is_file() for item in path.rglob("*"))


def _has_nonempty_weight(path: Path) -> bool:
    return any(
        item.is_file() and item.suffix.lower() in _WEIGHT_SUFFIXES and item.stat().st_size > 0
        for item in path.rglob("*")
    )


def verify_models(
    settings: ModelSettings, root: Path | None = None
) -> tuple[ModelCheck, ...]:
    paths = resolve_model_paths(settings, root)
    checks: list[ModelCheck] = []
    for spec in specs(settings):
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


def require_models(settings: ModelSettings, root: Path | None = None) -> dict[str, Path]:
    checks = verify_models(settings, root)
    failed = [f"{item.key}: {', '.join(item.errors)} ({item.path})" for item in checks if not item.ok]
    if failed:
        raise RuntimeError("本地模型校验失败:\n" + "\n".join(failed))
    return {item.key: item.path for item in checks}
