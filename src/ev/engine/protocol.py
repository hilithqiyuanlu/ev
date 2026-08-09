"""版本化 JSONL 协议。"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TextIO

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class EngineRequest:
    request_id: str
    command: str
    payload: dict

    @classmethod
    def parse(cls, line: str) -> "EngineRequest":
        raw = json.loads(line)
        if raw.get("version") != PROTOCOL_VERSION:
            raise ValueError(f"不支持的协议版本: {raw.get('version')}")
        request_id = str(raw.get("request_id") or uuid.uuid4())
        command = raw.get("command")
        payload = raw.get("payload", {})
        if not isinstance(command, str) or not command:
            raise ValueError("command 必须是非空字符串")
        if not isinstance(payload, dict):
            raise ValueError("payload 必须是对象")
        return cls(request_id, command, payload)


class ProtocolWriter:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self._lock = threading.Lock()

    def emit(self, event_type: str, payload: dict, request_id: str | None = None) -> None:
        event = {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        with self._lock:
            self.stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.stream.flush()

    def error(self, message: str, request_id: str | None = None, code: str = "engine_error") -> None:
        self.emit("error", {"code": code, "message": message}, request_id)
