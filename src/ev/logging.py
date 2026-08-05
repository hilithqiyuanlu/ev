"""日志初始化:stderr 控制台 + 文件(data/logs/ev.log)。"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-5s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level, format=_FMT, datefmt=_DATEFMT, handlers=handlers, force=True
    )
