import io
import json

import asyncio
import numpy as np

from ev.config import load_settings
from ev.engine.protocol import EngineRequest, ProtocolWriter
from ev.engine.service import EngineService
from ev.audio.environment import EnvEvent
from ev.store.environment import EnvironmentLog
from ev.store.db import Store


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


def test_environment_history_commands(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    log = EnvironmentLog(settings.logs_dir)
    log.append(EnvEvent("env-1", "typing", 100.0, 105.0, 0.8, 5.0))
    log.append(EnvEvent("env-2", "music", 200.0, 210.0, 0.9, 10.0))
    output = io.StringIO()
    service = EngineService(settings, output)

    service.handle(EngineRequest("list", "list_environment_events", {}))
    service.handle(EngineRequest("clear", "clear_environment_events", {}))

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    listed = next(item for item in events if item["type"] == "environment_event_list")
    assert [item["id"] for item in listed["payload"]["events"]] == ["env-2", "env-1"]
    cleared = next(item for item in events if item["type"] == "environment_events_cleared")
    assert cleared["payload"]["count"] == 2
    assert log.query() == []


def test_environment_monitoring_command_updates_setting(tmp_path, monkeypatch):
    output = io.StringIO()
    service = EngineService(_settings(tmp_path, monkeypatch), output)

    service.handle(EngineRequest("toggle", "set_environment_monitoring", {"enabled": False}))

    event = json.loads(output.getvalue())
    assert event["type"] == "environment_monitoring_changed"
    assert event["payload"]["enabled"] is False
    assert service._environment_enabled is False


def test_lexicon_status_commands_preserve_auto_source(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    with Store(settings.db_path) as store:
        item = store.add_lexicon_word("自动候选", source="auto")
    output = io.StringIO()
    service = EngineService(settings, output)

    service.handle(EngineRequest("confirm", "confirm_lexicon_word", {"id": item["id"]}))
    service.handle(EngineRequest(
        "disable", "set_lexicon_word_status", {"id": item["id"], "status": "disabled"}
    ))
    service.handle(EngineRequest(
        "enable", "set_lexicon_word_status", {"id": item["id"], "status": "active"}
    ))
    service.handle(EngineRequest("reject", "reject_lexicon_word", {"id": item["id"]}))

    with Store(settings.db_path) as store:
        row = next(value for value in store.list_lexicon() if value["id"] == item["id"])
        assert row["source"] == "auto"
        assert row["status"] == "disabled"
        assert row["confirmed_at"]
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert sum(event["type"] == "lexicon_updated" for event in events) == 4
