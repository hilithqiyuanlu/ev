import io
import json

import asyncio
import numpy as np

from ev.config import load_settings
from ev.engine.protocol import EngineRequest, ProtocolWriter
from ev.engine.service import EngineService


def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EV_MODEL_ROOT", str(tmp_path / "models"))
    return load_settings()


def test_protocol_parse_and_thread_safe_shape():
    request = EngineRequest.parse(
        '{"version":1,"request_id":"r1","command":"get_status","payload":{}}'
    )
    assert request.command == "get_status"
    output = io.StringIO()
    ProtocolWriter(output).emit("engine_state", {"state": "stopped"}, "r1")
    event = json.loads(output.getvalue())
    assert event["version"] == 1
    assert event["request_id"] == "r1"
    assert event["timestamp"]


def test_engine_status_manual_query_and_history(tmp_path, monkeypatch):
    output = io.StringIO()
    service = EngineService(_settings(tmp_path, monkeypatch), output)
    requests = io.StringIO(
        "\n".join(
            [
                '{"version":1,"request_id":"1","command":"get_status","payload":{}}',
                '{"version":1,"request_id":"2","command":"submit_manual_query","payload":{"text":"hello"}}',
                '{"version":1,"request_id":"3","command":"list_segments","payload":{}}',
                '{"version":1,"request_id":"4","command":"shutdown","payload":{}}',
            ]
        )
    )
    assert service.serve(requests) == 0
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert any(item["type"] == "engine_state" for item in events)
    query = next(item for item in events if item["type"] == "query_candidate")
    assert query["payload"]["source"] == "manual"
    history = next(item for item in events if item["type"] == "segment_list")
    assert history["payload"]["queries"][0]["text"] == "hello"


def test_unknown_command_returns_structured_error(tmp_path, monkeypatch):
    output = io.StringIO()
    service = EngineService(_settings(tmp_path, monkeypatch), output)
    service.handle(EngineRequest("x", "nope", {}))
    event = json.loads(output.getvalue())
    assert event["type"] == "error"
    assert event["payload"]["code"] == "unknown_command"
