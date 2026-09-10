from pathlib import Path
import json
import re
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from iv import Pipeline
from iv import graph
from iv.cli import app
from iv.errors import DeclError


def test_stage_selection_prefers_exact_names_over_longer_prefixes():
    from iv.cli import _stage_name

    g = SimpleNamespace(stages={"pipeline.py::dq": None, "pipeline.py::dq_player_games": None})
    assert _stage_name(g, "dq") == "pipeline.py::dq"
    assert _stage_name(g, "pipeline.py::dq") == "pipeline.py::dq"
    assert _stage_name(g, "player_games") == "pipeline.py::dq_player_games"


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for key in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(key, raising=False)
    return Pipeline(tree=tmp_path / "tree", project=tmp_path, stage_dir=tmp_path / "tmp")


def snapshots(iv):
    return {Path(p).name: Path(p).read_bytes() for p in iv.reads("audit/", why="verify retained audit history")}


def test_append_runs_with_current_inputs_and_retains_history_through_gc(iv, monkeypatch):
    import iv.cli as cli

    @iv.data(dataset="input/", ext=".json", once=True, why="stable input")
    def source():
        return {"value": 1}

    @iv.data(dataset="audit/", ext=".json", append=True, part="run", why="immutable audit snapshots")
    def audit(run, source=iv.all_of(source, why="audit current input")):
        return {"run": run, "value": source["value"]}

    monkeypatch.setattr(cli, "_load", lambda: iv)
    runner = CliRunner()
    for _ in range(2):
        result = runner.invoke(app, ["run", "--up-to", "audit"])
        assert result.exit_code == 0, result.output
    before = snapshots(iv)
    assert len(before) == 2
    assert all(re.match(r"run=\d{8}T\d{6}_\d{6}Z-[a-f0-9]{32}\.", name) for name in before)
    assert not audit.may_skip
    assert not audit.is_current()
    g = graph.build(iv)
    assert g.is_terminal("audit/")
    errors, warnings = graph.check(g)
    assert not errors
    assert not any("::audit" in warning or "NOBODY WRITES" in warning for warning in warnings)
    assert iv._expected_part_keys("audit/") == {("run",)}
    for command in (["gc"], ["gc", "audit/"], ["run", "--only", "audit", "--force"]):
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.output
    after = snapshots(iv)
    assert len(after) == 3
    assert all(after[k] == v for k, v in before.items())


def test_append_direct_calls_and_out_writer_receive_distinct_ids(iv):
    @iv.data(dataset="audit/", ext=".json", append=True, part="audit", why="retain every call")
    def audit(audit, out):
        out.write_text(json.dumps(audit))

    first, second = audit(), audit()
    assert first != second
    assert len(snapshots(iv)) == 2
    with pytest.raises(DeclError, match="no partition value"):
        audit(first)
    with pytest.raises(DeclError, match="never rebuilt"):
        audit.build({"audit": first})


def test_append_multi_output_shares_one_run_id(iv):
    @iv.step(output={"a": "audit/", "b": "detail/"}, ext=".json", part="run", append=True, why="retain report and details")
    def audit(run):
        return {"a": {"run": run}, "b": {"run": run}}

    assert audit() is True
    paths = iv.reads("audit/", why="inspect report") + iv.reads("detail/", why="inspect detail")
    assert len({Path(p).read_text() for p in paths}) == 1


@pytest.mark.parametrize("options", [
    {}, {"part": ["a", "b"]}, {"part": {"run": "fixed"}},
    {"part": "run", "split": True}, {"part": "run", "once": True},
    {"part": "run", "universe": ["fixed"]},
])
def test_invalid_append_declarations_are_rejected(iv, options):
    with pytest.raises(DeclError, match="append=True"):
        iv.data(dataset="audit/", append=True, why="invalid append declaration", **options)(lambda: {})


def test_determinism_skips_append_outputs_without_writing(iv, monkeypatch):
    import iv.cli as cli

    @iv.data(dataset="audit/", ext=".json", append=True, part="run", why="audit snapshot")
    def audit():
        raise AssertionError("determinism must not execute append outputs")

    monkeypatch.setattr(cli, "_load", lambda: iv)
    runner = CliRunner()
    sample = runner.invoke(app, ["determinism", "--sample"])
    assert sample.exit_code == 0, sample.output
    assert "append-only" in sample.output
    explicit = runner.invoke(app, ["determinism", "--only", "audit"])
    assert explicit.exit_code != 0
    assert "timestamped" in explicit.output
