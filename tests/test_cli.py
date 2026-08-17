"""The CLI, against the real stage files from the integration fixture.

The CLI has to find the project's `Invalidator`, which is the one piece of discovery in
the package: `[tool.invalidator] instance = "pipeline:iv"`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from test_integration import GAMES, pipeline, run_all  # noqa: F401

CLI = [sys.executable, "-m", "invalidator.cli"]


def cli(project, *args) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(project), "NO_COLOR": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("VIRTUAL_ENV", None)
    env.pop("INVALIDATOR_TRACE", None)
    return subprocess.run([*CLI, *args], cwd=project, env=env,
                          capture_output=True, text=True)


def test_graph_works_before_anything_has_run(pipeline):
    """The graph half reads your source. No data, no state file, no run."""
    r = cli(pipeline, "graph")
    assert r.returncode == 0, r.stderr
    assert "build_stats" in r.stdout and "build_ratings" in r.stdout


def test_a_missing_instance_says_exactly_what_to_add(pipeline):
    (pipeline / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    r = cli(pipeline, "graph")
    assert r.returncode == 1
    assert "[tool.invalidator]" in r.stderr and "instance" in r.stderr


def test_the_instance_can_be_passed_per_invocation(pipeline):
    (pipeline / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    r = cli(pipeline, "-i", "pipeline:iv", "graph")
    assert r.returncode == 0, r.stderr
    assert "build_stats" in r.stdout


def test_check_is_clean_and_exits_zero(pipeline):
    r = cli(pipeline, "check")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ok — 2 stages" in r.stdout


def test_check_exits_one_on_an_error(pipeline):
    (pipeline / "stages" / "orphan.py").write_text(
        'from pipeline import iv\n\n'
        '@iv.step("processed/orphan.parquet", why="nobody reads this")\n'
        'def build(out):\n'
        '    iv.reads("processed/team_stats.parquet", why="season points")\n')
    r = cli(pipeline, "check")
    assert r.returncode == 1
    assert "WRITE WITH NO CONSUMER" in r.stdout


def test_stage_prints_both_ends_and_the_why(pipeline):
    r = cli(pipeline, "stage", "build_ratings")
    assert r.returncode == 0, r.stderr
    assert "processed/team_stats.parquet" in r.stdout
    assert "<- build_stats" in r.stdout
    assert "season points by team" in r.stdout
    assert "(terminal — the app or a human)" in r.stdout


def test_status_and_why_after_a_run(pipeline):
    run_all(pipeline)

    r = cli(pipeline, "status")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2/2 current" in r.stdout
    assert "data_version='v1'" in r.stdout, "status names which pipeline it is talking about"

    r = cli(pipeline, "why", "processed/ratings.parquet")
    assert "current" in r.stdout
    assert "id   " in r.stdout and "meta " in r.stdout


def test_status_exits_one_and_names_the_mover(pipeline):
    run_all(pipeline)
    GAMES.with_columns(GAMES["pts"] * 2).write_parquet(
        pipeline / "data" / "raw" / "games.parquet")

    r = cli(pipeline, "status")
    assert r.returncode == 1
    assert "input moved: raw/games.parquet" in r.stdout


def test_plan_separates_definite_from_downstream(pipeline):
    run_all(pipeline)
    GAMES.with_columns(GAMES["pts"] * 2).write_parquet(
        pipeline / "data" / "raw" / "games.parquet")

    r = cli(pipeline, "plan")
    assert "rebuild  processed/team_stats.parquet" in r.stdout
    assert "maybe    processed/ratings.parquet" in r.stdout
    assert "downstream of a rebuild" in r.stdout


def test_export_is_json_in_the_dbt_shape(pipeline):
    r = cli(pipeline, "export")
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["data_version"] == "v1"
    assert out["stage_parent_map"]["stages/build_ratings.py"] == ["stages/build_stats.py"]
    assert out["nodes"]["processed/ratings.parquet"]["terminal"] is True


def test_drift_without_a_trace_says_so(pipeline):
    r = cli(pipeline, "drift")
    assert r.returncode == 1
    assert "INVALIDATOR_TRACE" in r.stderr


def test_viz_writes_a_png(pipeline, tmp_path):
    out = tmp_path / "dag.png"
    r = cli(pipeline, "viz", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists() and out.stat().st_size > 1000
