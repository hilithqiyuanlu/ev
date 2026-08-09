"""固定 Release 模型下载、校验与原子安装。"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from threading import Event
from typing import Callable

from .config import ModelSettings
from .models import verify_models


@dataclass(frozen=True)
class ModelAsset:
    key: str
    directory: str
    filename: str
    url: str
    size: int
    sha256: str


def load_manifest() -> tuple[str, tuple[ModelAsset, ...]]:
    resource = files("ev.resources").joinpath("models-v0.1.0.json")
    raw = json.loads(resource.read_text(encoding="utf-8"))
    return raw["version"], tuple(ModelAsset(**item) for item in raw["assets"])


class DownloadCancelled(RuntimeError):
    pass


class ModelDownloader:
    def __init__(
        self,
        settings: ModelSettings,
        emit: Callable[[str, dict], None],
        cancel_event: Event | None = None,
        opener: Callable = urllib.request.urlopen,
    ):
        self.settings = settings
        self.emit = emit
        self.cancel_event = cancel_event or Event()
        self.opener = opener

    def download_all(self) -> None:
        version, assets = load_manifest()
        self.settings.root.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(asset.size for asset in assets)
        completed_bytes = 0
        for index, asset in enumerate(assets):
            self._check_cancelled()
            target = self.settings.root / asset.directory
            if self._installed(asset):
                completed_bytes += asset.size
                self.emit("model_status", {"key": asset.key, "status": "ready", "path": str(target)})
                continue
            self.emit(
                "model_status",
                {"key": asset.key, "status": "downloading", "version": version},
            )
            self._download_asset(asset, completed_bytes, total_bytes, index, len(assets))
            completed_bytes += asset.size
        self.emit("model_status", {"status": "complete", "version": version})

    def _installed(self, asset: ModelAsset) -> bool:
        checks = verify_models(self.settings)
        return next(check.ok for check in checks if check.key == asset.key)

    def _download_asset(
        self,
        asset: ModelAsset,
        completed_before: int,
        total_bytes: int,
        asset_index: int,
        asset_count: int,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix=f".{asset.directory}-", dir=self.settings.root) as temp:
            temp_dir = Path(temp)
            archive = temp_dir / asset.filename
            digest = hashlib.sha256()
            downloaded = 0
            request = urllib.request.Request(asset.url, headers={"User-Agent": "EV/0.1"})
            with self.opener(request, timeout=60) as response, archive.open("wb") as output:
                while True:
                    self._check_cancelled()
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    self.emit(
                        "download_progress",
                        {
                            "key": asset.key,
                            "downloaded": downloaded,
                            "size": asset.size,
                            "total_downloaded": completed_before + downloaded,
                            "total_size": total_bytes,
                            "asset_index": asset_index + 1,
                            "asset_count": asset_count,
                        },
                    )
            if digest.hexdigest() != asset.sha256:
                raise RuntimeError(f"{asset.filename} SHA256 校验失败")
            staging = temp_dir / "staging"
            staging.mkdir()
            self._safe_extract(archive, staging)
            self._install(asset, staging)

    def _install(self, asset: ModelAsset, staging: Path) -> None:
        target = self.settings.root / asset.directory
        test_root = Path(tempfile.mkdtemp(prefix=".verify-", dir=self.settings.root))
        test_target = test_root / asset.directory
        shutil.move(str(staging), test_target)
        try:
            check = next(item for item in verify_models(self.settings, test_root) if item.key == asset.key)
            if not check.ok:
                raise RuntimeError(f"{asset.key} 模型结构无效: {', '.join(check.errors)}")
            backup = self.settings.root / f".{asset.directory}.backup"
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                os.replace(target, backup)
            try:
                os.replace(test_target, target)
            except Exception:
                if backup.exists():
                    os.replace(backup, target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            self.emit("model_status", {"key": asset.key, "status": "ready", "path": str(target)})
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

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

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise DownloadCancelled("模型下载已取消")
