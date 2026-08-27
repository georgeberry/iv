from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tyke import record
from tyke.errors import StateError


def recorder(path, *, depth=0):
    return SimpleNamespace(
        trace_path=path,
        _depth=depth,
        _trace_fh=None,
        node=lambda: "stages/train.py::train",
    )


def test_emit_writes_a_versioned_json_event(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "trace.jsonl"
    tyke = recorder(path)
    monkeypatch.setattr(record.time, "time", lambda: 12.3456)
    monkeypatch.setattr(record.os, "getpid", lambda: 42)

    record.emit(tyke, "built", output=tmp_path / "result")

    event = json.loads(path.read_text())
    assert event == {
        "v": record.RECORDER_VERSION,
        "kind": "built",
        "node": "stages/train.py::train",
        "pid": 42,
        "t": 12.346,
        "output": str(tmp_path / "result"),
    }


@pytest.mark.parametrize("trace, depth", [(None, 0), ("trace.jsonl", 1)])
def test_emit_is_disabled_without_a_trace_or_during_bookkeeping(tmp_path, trace, depth):
    path = None if trace is None else tmp_path / trace
    record.emit(recorder(path, depth=depth), "ignored")
    assert path is None or not path.exists()


def test_load_handles_missing_blank_and_old_events(tmp_path, capsys):
    path = tmp_path / "trace.jsonl"
    assert record.load(path) == []
    path.write_text(
        "\n"
        + json.dumps({"v": record.RECORDER_VERSION - 1, "kind": "old"})
        + "\n"
        + json.dumps({"v": record.RECORDER_VERSION, "kind": "new", "t": 1})
        + "\n"
    )

    assert record.load(path) == [
        {"v": record.RECORDER_VERSION, "kind": "new", "t": 1}
    ]
    assert "dropped 1 event(s)" in capsys.readouterr().out


def test_load_reports_the_torn_line(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text('{"v": 2}\n{"broken"')

    with pytest.raises(StateError, match=r"trace\.jsonl:2 is not parseable"):
        record.load(path)


def test_age_of_uses_the_newest_numeric_timestamp(monkeypatch):
    monkeypatch.setattr(record.time, "time", lambda: 20)
    assert record.age_of([{"t": 3}, {"t": "4"}, {"t": 8.5}]) == 11.5
    assert record.age_of([{"kind": "empty"}]) is None
