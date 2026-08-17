"""The real thing: stage files on disk, run as separate processes, guarded on their own
outputs — which is how a pipeline actually works, one process per stage.

The in-process tests can only exercise the state file. These exercise the whole loop: the
static scan reads the source, the guard asks it what the stage reads now, the run stamps,
and the next run answers from the state file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import polars as pl
import pytest

from dagio import config as _config
from dagio import static as _static

VENV_PY = sys.executable

STATS = '''
    """Season points by team."""
    import argparse
    import polars as pl
    import dagio as dg

    def build():
        games = pl.read_parquet(dg.reads(
            "raw/games.parquet", why="one row per team per game; the only source of points"))
        out = games.group_by("team", maintain_order=True).agg(pl.col("pts").sum())
        with dg.writes("processed/team_stats.parquet",
                       why="season points by team; the rating denominator") as p:
            out.write_parquet(p)

    ap = argparse.ArgumentParser()
    dg.add_guard_args(ap)
    a = ap.parse_args()
    dg.build_if_needed("processed/team_stats.parquet", build,
                       if_needed=a.if_needed, force=a.force)
'''

RATINGS = '''
    """Team ratings."""
    import argparse
    import polars as pl
    import dagio as dg

    def build():
        stats = pl.read_parquet(dg.reads(
            "processed/team_stats.parquet", why="season points by team"))
        with dg.writes("processed/ratings.parquet",
                       why="team ratings the app renders", terminal=True) as p:
            stats.with_columns((pl.col("pts") / 100).alias("rating")).write_parquet(p)

    ap = argparse.ArgumentParser()
    dg.add_guard_args(ap)
    a = ap.parse_args()
    dg.build_if_needed("processed/ratings.parquet", build,
                       if_needed=a.if_needed, force=a.force)
'''

GAMES = pl.DataFrame({"game_id": [1, 2, 3], "team": ["A", "B", "A"], "pts": [90, 85, 100]})


@pytest.fixture
def pipeline(project):
    for rel, body in (("stages/build_stats.py", STATS),
                      ("stages/build_ratings.py", RATINGS)):
        (project / rel).write_text(textwrap.dedent(body).lstrip())
    (project / "pyproject.toml").write_text(
        (project / "pyproject.toml").read_text().replace(
            'data_root = "data"',
            'data_root = "data"\n'
            'stages = ["stages/build_stats.py", "stages/build_ratings.py"]'))
    _config.reset()
    _static.reset()
    root = project / "data" / "raw"
    root.mkdir(parents=True, exist_ok=True)
    GAMES.write_parquet(root / "games.parquet")
    return project


def run(project, stage: str, *args) -> str:
    env = {**os.environ, "DAGIO_PROJECT": str(project),
           "PYTHONPATH": str(project)}
    env.pop("VIRTUAL_ENV", None)
    r = subprocess.run([VENV_PY, stage, *args], cwd=project, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def run_all(project, *args) -> str:
    return (run(project, "stages/build_stats.py", *args)
            + run(project, "stages/build_ratings.py", *args))


def state(project) -> dict:
    p = project / "data" / ".dagio" / "state.json"
    return json.loads(p.read_text())["artifacts"] if p.exists() else {}


def test_cold_build_then_a_no_op_run(pipeline):
    run_all(pipeline, "--if-needed")
    assert (pipeline / "data" / "processed" / "ratings.parquet").exists()
    st = state(pipeline)
    assert set(st) == {"processed/team_stats.parquet", "processed/ratings.parquet"}
    assert st["processed/ratings.parquet"]["in"]["processed/team_stats.parquet"]["id"] \
        == st["processed/team_stats.parquet"]["id"], "the id is folded in, not the fingerprint"

    out = run_all(pipeline, "--if-needed")
    assert out.count("is current — skipping") == 2


def test_a_new_row_at_the_root_rebuilds_the_chain(pipeline):
    run_all(pipeline, "--if-needed")
    GAMES.vstack(pl.DataFrame({"game_id": [4], "team": ["B"], "pts": [77]})) \
         .write_parquet(pipeline / "data" / "raw" / "games.parquet")

    out = run_all(pipeline, "--if-needed")
    assert "input moved: raw/games.parquet" in out
    assert "input moved: processed/team_stats.parquet" in out
    assert "skipping" not in out


def test_why_names_the_component_that_moved(pipeline):
    run_all(pipeline, "--if-needed")
    from dagio import state as _state
    _state.reset()
    _config.reset()
    os.environ["DAGIO_PROJECT"] = str(pipeline)
    GAMES.with_columns(pl.col("pts") * 2).write_parquet(
        pipeline / "data" / "raw" / "games.parquet")
    import dagio as dg
    assert dg.why_stale("processed/team_stats.parquet").startswith(
        "input moved: raw/games.parquet")


def test_adding_a_read_to_the_source_makes_it_stale(pipeline):
    """The case the state file alone cannot see. The stored record has no entry for a path
    the last build never read, so only the code can say the input set changed."""
    run_all(pipeline, "--if-needed")
    pl.DataFrame({"team": ["A", "B"], "pace": [98.0, 101.0]}).write_parquet(
        pipeline / "data" / "raw" / "pace.parquet")

    src = pipeline / "stages" / "build_stats.py"
    patched = src.read_text().replace(
        "    out = games.group_by",
        '    pl.read_parquet(dg.reads("raw/pace.parquet", why="team pace"))\n'
        "    out = games.group_by")
    assert "raw/pace.parquet" in patched
    src.write_text(patched)

    out = run(pipeline, "stages/build_stats.py", "--if-needed")
    assert "input added: raw/pace.parquet" in out, out


def test_a_failing_stage_stamps_nothing_and_stays_stale(pipeline):
    src = pipeline / "stages" / "build_stats.py"
    patched = src.read_text().replace(
        "        out.write_parquet(p)",
        "        out.write_parquet(p)\n"
        "        raise RuntimeError('the fit diverged')")
    assert "the fit diverged" in patched
    src.write_text(patched)

    env = {**os.environ, "DAGIO_PROJECT": str(pipeline), "PYTHONPATH": str(pipeline)}
    env.pop("VIRTUAL_ENV", None)
    r = subprocess.run([VENV_PY, "stages/build_stats.py", "--if-needed"],
                       cwd=pipeline, env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "processed/team_stats.parquet" not in state(pipeline)
    assert (pipeline / "data" / "processed" / "team_stats.parquet").exists(), \
        "the partial file is on disk — which is exactly why the stamp must not be"

    src.write_text(textwrap.dedent(STATS).lstrip())          # fix the stage
    out = run(pipeline, "stages/build_stats.py", "--if-needed")
    assert "never stamped" in out


def test_the_trace_records_what_ran(pipeline):
    env_trace = pipeline / ".dagio" / "trace.ndjson"
    os.environ["DAGIO_TRACE"] = str(env_trace)
    try:
        run_all(pipeline, "--if-needed")
    finally:
        os.environ.pop("DAGIO_TRACE")

    events = [json.loads(l) for l in env_trace.read_text().splitlines() if l.strip()]
    pairs = {(e["node"], e["op"], e["rel"]) for e in events if e["kind"] == "io"}
    assert ("stages/build_stats.py", "read", "raw/games.parquet") in pairs
    assert ("stages/build_ratings.py", "write", "processed/ratings.parquet") in pairs
    assert all(e.get("why") for e in events if e["kind"] == "io")


def test_drift_is_empty_when_the_code_and_the_run_agree(pipeline):
    trace = pipeline / ".dagio" / "trace.ndjson"
    os.environ["DAGIO_TRACE"] = str(trace)
    try:
        run_all(pipeline, "--if-needed")
    finally:
        os.environ.pop("DAGIO_TRACE")

    _config.reset()
    _static.reset()
    os.environ["DAGIO_PROJECT"] = str(pipeline)
    from dagio import graph as G, record as _rec
    errors, _ = G.drift(G.build(), _rec.load(trace))
    assert errors == []
