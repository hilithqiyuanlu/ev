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
class Settings:
    log_level: str
    audio: AudioSettings
    data_dir: Path

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

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
    data_dir = Path(
        os.environ.get("EV_DATA_DIR", raw.get("paths", {}).get("data_dir", "data"))
    )
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir

    return Settings(
        log_level=os.environ.get("EV_LOG_LEVEL", raw.get("log_level", "INFO")).upper(),
        audio=AudioSettings(
            sample_rate=int(audio_raw.get("sample_rate", 16000)),
            channels=int(audio_raw.get("channels", 1)),
        ),
        data_dir=data_dir,
    )
