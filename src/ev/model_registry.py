"""模型注册表 — 管理已安装模型、槽位分配与持久化。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .model_catalog import (
    ModelDefinition,
    ModelSource,
    ModelType,
    get_all_slots,
    get_default_slot,
    get_definition,
)


@dataclass
class InstalledModel:
    definition_key: str
    local_path: str
    installed_at: str
    size_bytes: int = 0


@dataclass
class SlotAssignment:
    slot: str
    model_key: str | None = None
    enabled: bool = True


@dataclass
class ModelRegistryState:
    installed: dict[str, InstalledModel] = field(default_factory=dict)
    slots: dict[str, SlotAssignment] = field(default_factory=dict)
    version: str = "1"


class ModelRegistry:
    """模型注册表 — 线程安全的模型管理核心。

    负责模型的安装、卸载、槽位分配、校验和持久化。
    支持 ModelScope 自动下载和 GitHub Release 下载。
    """

    def __init__(
        self,
        models_root: Path,
        emit: Callable[[str, dict], None] | None = None,
        state_path: Path | None = None,
    ):
        self.models_root = models_root
        self.emit = emit or (lambda kind, payload: None)
        self.state_path = state_path or (models_root.parent / "models_registry.json")
        self._lock = threading.RLock()
        self._cancel_event = threading.Event()
        self._download_thread: threading.Thread | None = None
        self.state = self._load_state()

    # ── 持久化 ──────────────────────────────────────────────────────

    def _load_state(self) -> ModelRegistryState:
        """从磁盘加载注册表状态，不存在则初始化默认值。"""
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                slots = {
                    k: SlotAssignment(**v)
                    for k, v in data.get("slots", {}).items()
                }
                installed = {
                    k: InstalledModel(**v)
                    for k, v in data.get("installed", {}).items()
                }
                return ModelRegistryState(
                    installed=installed,
                    slots=slots,
                    version=data.get("version", "1"),
                )
            except (json.JSONDecodeError, TypeError):
                pass

        state = ModelRegistryState()
        for slot in get_all_slots():
            default_key = get_default_slot(slot)
            state.slots[slot] = SlotAssignment(
                slot=slot,
                model_key=default_key,
                enabled=True,
            )
        return state

    def _save_state(self) -> None:
        """持久化注册表状态到磁盘。"""
        data = {
            "version": self.state.version,
            "installed": {
                k: {
                    "definition_key": v.definition_key,
                    "local_path": v.local_path,
                    "installed_at": v.installed_at,
                    "size_bytes": v.size_bytes,
                }
                for k, v in self.state.installed.items()
            },
            "slots": {
                k: {
                    "slot": v.slot,
                    "model_key": v.model_key,
                    "enabled": v.enabled,
                }
                for k, v in self.state.slots.items()
            },
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── 查询接口 ──────────────────────────────────────────────────────

    def list_installed(self) -> dict[str, InstalledModel]:
        """返回所有已安装模型。"""
        with self._lock:
            return dict(self.state.installed)

    def list_slot_assignments(self) -> dict[str, SlotAssignment]:
        """返回所有槽位分配。"""
        with self._lock:
            return dict(self.state.slots)

    def get_active_model(self, slot: str) -> InstalledModel | None:
        """获取某槽位当前激活的已安装模型。"""
        with self._lock:
            assignment = self.state.slots.get(slot)
            if not assignment or not assignment.model_key:
                return None
            return self.state.installed.get(assignment.model_key)

    def get_active_definition(self, slot: str) -> ModelDefinition | None:
        """获取某槽位当前激活模型的定义。"""
        assignment = self.state.slots.get(slot)
        if not assignment or not assignment.model_key:
            return None
        return get_definition(assignment.model_key)

    def get_slot_status(self, slot: str) -> dict:
        """获取槽位状态摘要，用于前端展示。"""
        assignment = self.state.slots.get(slot)
        if not assignment or not assignment.model_key:
            return {
                "slot": slot,
                "model_key": None,
                "ready": False,
                "path": None,
                "errors": ["未配置模型"],
            }
        definition = get_definition(assignment.model_key)
        installed = self.state.installed.get(assignment.model_key)
        if not definition or not installed:
            return {
                "slot": slot,
                "model_key": assignment.model_key,
                "ready": False,
                "path": None,
                "errors": ["模型未安装"],
            }
        path = Path(installed.local_path)
        errors = self._verify_local(path, definition)
        return {
            "slot": slot,
            "model_key": assignment.model_key,
            "model_name": definition.name,
            "ready": not errors,
            "path": str(path),
            "errors": list(errors),
            "size_bytes": installed.size_bytes,
        }

    def get_all_slot_status(self) -> list[dict]:
        """获取所有槽位状态。"""
        return [self.get_slot_status(slot) for slot in get_all_slots()]

    def are_all_slots_ready(self) -> bool:
        """判断所有已启用槽位的模型是否就绪。"""
        for slot in get_all_slots():
            assignment = self.state.slots.get(slot)
            if not assignment or not assignment.enabled:
                continue
            status = self.get_slot_status(slot)
            if not status["ready"]:
                return False
        return True

    # ── 安装/卸载 ──────────────────────────────────────────────────────

    def install_model(self, model_key: str) -> None:
        """下载并安装指定模型。"""
        with self._lock:
            definition = get_definition(model_key)
            if not definition:
                raise ValueError(f"未知模型: {model_key}")

            existing = self.state.installed.get(model_key)
            if existing:
                path = Path(existing.local_path)
                if path.is_dir() and not self._verify_local(path, definition):
                    self.emit("model_status", {
                        "key": model_key,
                        "status": "repairing",
                        "message": "模型文件损坏，正在重新下载",
                    })
                    self._remove_installation(model_key)
                else:
                    self.emit("model_status", {
                        "key": model_key,
                        "status": "ready",
                        "path": str(path),
                    })
                    return

            self._cancel_event.clear()

        self._emit_progress(model_key, "starting")

        try:
            if definition.source == ModelSource.MODELSCOPE:
                local_path = self._download_modelscope(definition)
            elif definition.source == ModelSource.GITHUB:
                local_path = self._download_github(definition)
            else:
                raise ValueError(f"不支持的下载源: {definition.source}")

            errors = self._verify_local(local_path, definition)
            if errors:
                raise RuntimeError(f"模型校验失败: {', '.join(errors)}")

            size = self._dir_size(local_path)
            with self._lock:
                self.state.installed[model_key] = InstalledModel(
                    definition_key=model_key,
                    local_path=str(local_path),
                    installed_at=datetime.now(timezone.utc).isoformat(),
                    size_bytes=size,
                )
                self._save_state()

            self.emit("model_status", {
                "key": model_key,
                "status": "ready",
                "path": str(local_path),
                "size_bytes": size,
            })

        except Exception:
            self.emit("model_status", {
                "key": model_key,
                "status": "error",
            })
            raise

    def uninstall_model(self, model_key: str) -> None:
        """卸载指定模型（删除文件 + 清理缓存 + 清理注册表）。"""
        with self._lock:
            installed = self.state.installed.get(model_key)
            if not installed:
                return

            definition = get_definition(model_key)

            path = Path(installed.local_path)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

            # Clean up modelscope cache so reinstall actually re-downloads
            if definition and definition.source == ModelSource.MODELSCOPE and definition.modelscope_id:
                # Project-local cache
                project_cache = self.models_root.parent / ".modelscope_cache"
                if project_cache.exists():
                    shutil.rmtree(project_cache, ignore_errors=True)
                # Global modelscope cache (~/.cache/modelscope/hub/{org}/{model})
                global_cache = Path.home() / ".cache" / "modelscope" / "hub"
                if global_cache.exists() and definition.modelscope_id:
                    parts = definition.modelscope_id.split("/")
                    if len(parts) == 2:
                        org, model = parts
                        org_dir = global_cache / org
                        model_dir_name = model.lower().replace("_", "-")
                        for child in org_dir.iterdir() if org_dir.exists() else []:
                            if child.is_dir() and model_dir_name in child.name.lower():
                                shutil.rmtree(child, ignore_errors=True)

            del self.state.installed[model_key]

            for slot, assignment in self.state.slots.items():
                if assignment.model_key == model_key:
                    self.state.slots[slot] = SlotAssignment(
                        slot=slot,
                        model_key=None,
                        enabled=assignment.enabled,
                    )

            self._save_state()

        self.emit("model_status", {
            "key": model_key,
            "status": "removed",
        })

    def set_active_model(self, slot: str, model_key: str | None) -> None:
        """为指定槽位设置激活的模型。"""
        with self._lock:
            if model_key is not None:
                definition = get_definition(model_key)
                if not definition:
                    raise ValueError(f"未知模型: {model_key}")
                installed = self.state.installed.get(model_key)
                if not installed:
                    raise RuntimeError(f"模型未安装: {model_key}")

            assignment = self.state.slots.get(slot)
            if assignment:
                self.state.slots[slot] = SlotAssignment(
                    slot=slot,
                    model_key=model_key,
                    enabled=assignment.enabled,
                )
            else:
                self.state.slots[slot] = SlotAssignment(
                    slot=slot,
                    model_key=model_key,
                    enabled=True,
                )
            self._save_state()

        self.emit("model_status", {
            "slot": slot,
            "model_key": model_key,
            "status": "active",
        })

    def set_slot_enabled(self, slot: str, enabled: bool) -> None:
        """启用/禁用某槽位。"""
        with self._lock:
            assignment = self.state.slots.get(slot)
            if not assignment:
                return
            self.state.slots[slot] = SlotAssignment(
                slot=slot,
                model_key=assignment.model_key,
                enabled=enabled,
            )
            self._save_state()

    def cancel_download(self) -> None:
        """取消当前下载。"""
        self._cancel_event.set()

    # ── 下载实现 ──────────────────────────────────────────────────────

    def _format_bytes(self, b: int) -> str:
        if b < 1024:
            return f"{b} B"
        if b < 1024 * 1024:
            return f"{b / 1024:.1f} KB"
        if b < 1024 * 1024 * 1024:
            return f"{b / 1024 / 1024:.1f} MB"
        return f"{b / 1024 / 1024 / 1024:.2f} GB"

    def _download_modelscope(self, definition: ModelDefinition) -> Path:
        """从 ModelScope 下载模型（分阶段进度追踪 + 速度显示）。"""
        try:
            from modelscope import snapshot_download
            from modelscope.hub.api import HubApi
        except ImportError:
            raise RuntimeError("未安装 modelscope，请先执行: pip install modelscope")

        target = self.models_root / definition.default_dirname
        cache_dir = self.models_root.parent / ".modelscope_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        total_estimate = definition.estimated_size_bytes or 1_000_000_000

        # Stage 1: 获取文件列表
        self.emit("download_progress", {
            "key": definition.key,
            "stage": "listing",
            "message": "正在获取文件列表...",
            "downloaded": 0,
            "size": float(total_estimate),
            "total_size": float(total_estimate),
            "speed": 0,
        })

        try:
            api = HubApi()
            files = api.get_model_files(definition.modelscope_id or "")
        except Exception:
            files = []

        # Calculate real total size from file list if available
        real_total = 0
        for f in files:
            size = f.get("Size", 0)
            path = f.get("Path", "")
            if size > 0 and not path.startswith(("fig", "example", "assets")):
                if not path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".md")):
                    if path not in (".gitattributes",):
                        real_total += size

        if real_total > 0:
            total_estimate = real_total

        # Stage 2: 下载中 (ModelScope uses atomic download, so we show indeterminate progress)
        self.emit("download_progress", {
            "key": definition.key,
            "stage": "downloading",
            "source": "modelscope",
            "message": "正在从 ModelScope 下载模型文件...",
            "downloaded": 0,
            "size": float(total_estimate),
            "total_size": float(total_estimate),
            "speed": 0,
            "indeterminate": True,
        })

        # Monitor thread: track cache directory size + calculate speed
        downloaded_bytes = [0]
        last_report_time = [time.time()]
        last_report_bytes = [0]
        stop_monitor = threading.Event()

        def get_downloaded_size() -> int:
            total = 0
            try:
                # Check both cache and target
                for base in [cache_dir, target]:
                    if base.exists():
                        for f in base.rglob("*"):
                            if f.is_file():
                                total += f.stat().st_size
            except OSError:
                pass
            return total

        def progress_monitor() -> None:
            while not stop_monitor.is_set():
                if self._cancel_event.is_set():
                    break
                current = get_downloaded_size()
                now = time.time()
                elapsed = now - last_report_time[0]

                if elapsed >= 1.0 and current > 0:
                    speed = (current - last_report_bytes[0]) / max(elapsed, 0.1)
                    downloaded_bytes[0] = current
                    # Only report determinate progress if we see actual file growth
                    # (modelscope may download to temp first, so size may jump at the end)
                    if current < total_estimate * 0.95:
                        self.emit("download_progress", {
                            "key": definition.key,
                            "stage": "downloading",
                            "source": "modelscope",
                            "message": "正在从 ModelScope 下载模型文件...",
                            "downloaded": float(current),
                            "size": float(total_estimate),
                            "total_size": float(total_estimate),
                            "speed": float(speed),
                            "indeterminate": False,
                        })
                    last_report_time[0] = now
                    last_report_bytes[0] = current
                stop_monitor.wait(0.5)

        monitor_thread = threading.Thread(target=progress_monitor, daemon=True)
        monitor_thread.start()

        try:
            local_path = Path(snapshot_download(
                definition.modelscope_id,
                cache_dir=str(cache_dir),
                local_files_only=False,
            ))
        finally:
            stop_monitor.set()
            monitor_thread.join(timeout=2)

        if self._cancel_event.is_set():
            raise RuntimeError("下载已取消")

        # Stage 3: 复制到目标目录
        self.emit("download_progress", {
            "key": definition.key,
            "stage": "installing",
            "message": "正在安装模型...",
            "downloaded": float(total_estimate),
            "size": float(total_estimate),
            "total_size": float(total_estimate),
            "speed": 0,
        })

        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(local_path, target)

        final_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) if target.exists() else total_estimate

        # Final update
        self.emit("download_progress", {
            "key": definition.key,
            "stage": "done",
            "message": "下载完成",
            "downloaded": float(final_size),
            "size": float(final_size),
            "total_size": float(final_size),
            "speed": 0,
        })

        return target

    def _download_github(self, definition: ModelDefinition) -> Path:
        """从 GitHub Release 下载模型（分阶段进度 + 速度）。"""
        if not definition.github_url:
            raise ValueError(f"模型 {definition.key} 未配置 GitHub URL")

        target = self.models_root / definition.default_dirname
        total_size = definition.github_size or 100_000_000

        self.emit("download_progress", {
            "key": definition.key,
            "stage": "downloading",
            "message": "正在下载模型文件...",
            "downloaded": 0,
            "size": float(total_size),
            "total_size": float(total_size),
            "speed": 0,
        })

        with tempfile.TemporaryDirectory(
            prefix=f".{definition.default_dirname}-",
            dir=self.models_root,
        ) as temp:
            temp_dir = Path(temp)
            archive = temp_dir / definition.github_filename
            digest = hashlib.sha256()
            downloaded = 0
            last_time = time.time()
            last_bytes = 0

            request = urllib.request.Request(
                definition.github_url,
                headers={"User-Agent": "EV/0.2"},
            )
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
                while True:
                    if self._cancel_event.is_set():
                        raise RuntimeError("下载已取消")
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    elapsed = now - last_time
                    speed = (downloaded - last_bytes) / max(elapsed, 0.1) if elapsed > 0.3 else 0
                    if elapsed > 0.3:
                        last_time = now
                        last_bytes = downloaded
                    self.emit("download_progress", {
                        "key": definition.key,
                        "stage": "downloading",
                        "message": "正在下载模型文件...",
                        "downloaded": float(downloaded),
                        "size": float(total_size),
                        "total_size": float(total_size),
                        "speed": float(speed),
                    })

            if definition.github_sha256 and digest.hexdigest() != definition.github_sha256:
                raise RuntimeError(f"{definition.github_filename} SHA256 校验失败")

            self.emit("download_progress", {
                "key": definition.key,
                "stage": "installing",
                "message": "正在解压模型...",
                "downloaded": float(total_size),
                "size": float(total_size),
                "total_size": float(total_size),
                "speed": 0,
            })

            staging = temp_dir / "staging"
            staging.mkdir()
            self._safe_extract(archive, staging)

            if target.exists():
                backup = self.models_root / f".{definition.default_dirname}.backup"
                if backup.exists():
                    shutil.rmtree(backup)
                os.replace(str(target), str(backup))
            try:
                shutil.move(str(staging), str(target))
                if backup.exists():
                    shutil.rmtree(backup)
            except Exception:
                if backup.exists():
                    os.replace(str(backup), str(target))
                raise

        final_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) if target.exists() else total_size
        self.emit("download_progress", {
            "key": definition.key,
            "stage": "done",
            "message": "下载完成",
            "downloaded": float(final_size),
            "size": float(final_size),
            "total_size": float(final_size),
            "speed": 0,
        })

        return target

    @staticmethod
    def _safe_extract(archive: Path, target: Path) -> None:
        resolved = target.resolve()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                destination = (target / member.name).resolve()
                if destination != resolved and resolved not in destination.parents:
                    raise RuntimeError("模型压缩包包含不安全路径")
                if member.issym() or member.islnk():
                    raise RuntimeError("模型压缩包不得包含链接")
            tar.extractall(target)

        entries = list(target.iterdir())
        dirs = [p for p in entries if p.is_dir() and not p.name.startswith(".")]
        files = [p for p in entries if p.is_file() and not p.name.startswith("._")]
        if len(dirs) == 1 and len(files) == 0:
            nested = dirs[0]
            if (nested / "configuration.json").exists() or (nested / "config.json").exists():
                for item in nested.iterdir():
                    shutil.move(str(item), str(target / item.name))
                nested.rmdir()

    # ── 校验 ──────────────────────────────────────────────────────────

    def _verify_local(self, path: Path, definition: ModelDefinition) -> list[str]:
        """校验本地模型文件完整性。"""
        errors: list[str] = []
        if not path.is_dir():
            return ["目录不存在"]

        if not self._has_named_file(path, definition.config_filenames):
            errors.append("缺少模型配置文件")
        if not self._has_nonempty_weight(path, definition.weight_suffixes):
            errors.append("缺少非空权重文件")
        if definition.needs_tokens and not self._has_named_file(path, ("tokens.json",)):
            errors.append("缺少 tokens.json")
        if definition.needs_seg_dict and not any(
            item.is_file() and "seg_dict" in item.name for item in path.rglob("*")
        ):
            errors.append("缺少 seg_dict")
        return errors

    @staticmethod
    def _has_named_file(path: Path, names: tuple[str, ...]) -> bool:
        return any(item.name in names and item.is_file() for item in path.rglob("*"))

    @staticmethod
    def _has_nonempty_weight(path: Path, suffixes: tuple[str, ...]) -> bool:
        return any(
            item.is_file() and item.suffix.lower() in suffixes and item.stat().st_size > 0
            for item in path.rglob("*")
        )

    @staticmethod
    def _dir_size(path: Path) -> int:
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

    def _remove_installation(self, model_key: str) -> None:
        """清理安装记录和文件。"""
        installed = self.state.installed.get(model_key)
        if installed:
            path = Path(installed.local_path)
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            del self.state.installed[model_key]

    def _emit_progress(self, model_key: str, stage: str, **kwargs) -> None:
        self.emit("download_progress", {
            "key": model_key,
            "stage": stage,
            **kwargs,
        })


# ── 迁移工具 ──────────────────────────────────────────────────────────

def migrate_from_legacy(
    registry: ModelRegistry,
    old_slots: dict[str, str],
    old_root: Path,
) -> None:
    """从旧配置（硬编码4槽位）迁移到新注册表。"""
    for slot, dirname in old_slots.items():
        if not dirname:
            continue
        path = old_root / dirname
        if not path.is_dir():
            continue

        matching_def = None
        definition = get_definition(def_key_to_model_key(slot))
        if definition and definition.default_dirname == dirname:
            matching_def = definition

        if not matching_def:
            matching_def = get_definition(_guess_model_key(slot, dirname))

        if matching_def:
            with registry._lock:
                registry.state.installed[matching_def.key] = InstalledModel(
                    definition_key=matching_def.key,
                    local_path=str(path),
                    installed_at=datetime.now(timezone.utc).isoformat(),
                    size_bytes=registry._dir_size(path),
                )
                assignment = registry.state.slots.get(slot)
                if assignment:
                    registry.state.slots[slot] = SlotAssignment(
                        slot=slot,
                        model_key=matching_def.key,
                        enabled=assignment.enabled,
                    )
            registry._save_state()


def _guess_model_key(slot: str, dirname: str) -> str | None:
    """根据目录名猜测模型 key。"""
    catalog = {
        "ev-fsmn-vad-zh-16k": "fsmn-vad",
        "ev-paraformer-zh-streaming-16k": "paraformer-zh-streaming",
        "ev-paraformer-zh-16k": "paraformer-zh",
        "ev-eres2netv2-zh-16k": "eres2netv2",
        "qwen3-asr-0.6b": "qwen3-asr-0.6b",
        "qwen3-asr-1.7b": "qwen3-asr-1.7b",
    }
    if dirname in catalog:
        return catalog[dirname]
    # Handle ModelScope-style directory names (e.g. "Qwen3-ASR-0.6B-hf")
    lower = dirname.lower()
    if "qwen3-asr" in lower or "qwen3_asr" in lower:
        if "1.7b" in lower or "1_7b" in lower:
            return "qwen3-asr-1.7b"
        return "qwen3-asr-0.6b"
    return None


def def_key_to_model_key(slot: str) -> str | None:
    """将旧槽位名映射到模型 key。"""
    mapping = {
        "vad": "fsmn-vad",
        "asr_streaming": "paraformer-zh-streaming",
        "asr_final": "paraformer-zh",
        "speaker": "eres2netv2",
    }
    return mapping.get(slot)
