from __future__ import annotations

from ..config import load_settings
from .service import EngineService


def main() -> int:
    return EngineService(load_settings()).serve()


if __name__ == "__main__":
    raise SystemExit(main())
