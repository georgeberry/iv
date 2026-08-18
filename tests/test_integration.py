"""The real thing: stage files on disk, run as separate processes.

That is how a pipeline actually works — one process per stage — and it is the only way to
exercise the whole loop: the static scan reads the source, the guard asks it what the
stage reads now, the run stamps, and the next run answers from the state file.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import polars as pl
import pytest

from invalidator.state import read_records

VENV_PY = sys.executable

PIPELINE = '''
    from invalidator import Invalidator

    iv = Invalidator(
        data_root="data",
        data_version="v1",
        source_dirs=["stages"],
        stages=["stages/build_stats.py", "stages/build_ratings.py"],
    )
'''

STATS = '''
    """Season points by team."""
    import polars as pl
    from pipeline import iv

    @iv.step("processed/team_stats.parquet",
             why="season points by team; the rating denominator")
    def build(out):
        games = pl.read_parquet(iv.reads(
            "raw/games.parquet",
            why="one row per team per game; the only source of points"))
        games.group_by("team", maintain_order=True).agg(
            pl.col("pts").sum()).write_parquet(out)

    build()
'''

RATINGS = '''
    """Team ratings."""
    import polars as pl
    from pipeline import iv

    @iv.step("processed/ratings.parquet",
             why="team ratings the app renders", terminal=True)
    def build(out):
        stats = pl.read_parquet(iv.reads(
            "processed/team_stats.parquet", why="season points by team"))
        stats.with_columns((pl.col("pts") / 100).alias("rating")).write_parquet(out)

    build()
'''

GAMES = pl.DataFrame({"game_id": [1, 2, 3], "team": ["A", "B", "A"], "pts": [90, 85, 100]})


@pytest.fixture
def pipeline(project):
    (project / "pipeline.py").write_text(textwrap.dedent(PIPELINE).lstrip())
    (project / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n\n'
        '[tool.invalidator]\ninstance = "pipeline:iv"\n')
    for rel, body in (("stages/build_stats.py", STATS),
                      ("stages/build_ratings.py", RATINGS)):
        (project / rel).write_text(textwrap.dedent(body).lstrip())
    raw = project / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    GAMES.write_parquet(raw / "games.parquet")
    return project


def run(project, stage: str, *args, check: bool = True) -> str:
    env = {**os.environ, "PYTHONPATH": str(project), "NO_COLOR": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("VIRTUAL_ENV", None)
    for var in ("INVALIDATOR_TRACE", "INVALIDATOR_FORCE"):
        if var not in os.environ:
            env.pop(var, None)
    r = subprocess.run([VENV_PY, stage, *args], cwd=project, env=env,
                       capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout + r.stderr


def run_all(project, *args) -> str:
    return (run(project, "stages/build_stats.py", *args)
            + run(project, "stages/build_ratings.py", *args))


def state(project) -> dict:
    return read_records(project / "data" / ".invalidator" / "state")


def test_cold_build_then_a_no_op_run(pipeline):
    run_all(pipeline)
    assert (pipeline / "data" / "processed" / "ratings.parquet").exists()
    st = state(pipeline)
    assert set(st) == {"processed/team_stats.parquet", "processed/ratings.parquet"}
    assert st["processed/ratings.parquet"]["in"]["processed/team_stats.parquet"]["id"] \
        == st["processed/team_stats.parquet"]["id"], \
        "the id is folded in, not the fingerprint"

    out = run_all(pipeline)
    assert out.count("is current — skipping") == 2


def test_a_new_row_at_the_root_rebuilds_the_chain(pipeline):
    run_all(pipeline)
    GAMES.vstack(pl.DataFrame({"game_id": [4], "team": ["B"], "pts": [77]})) \
         .write_parquet(pipeline / "data" / "raw" / "games.parquet")

    out = run_all(pipeline)
    assert "input moved: raw/games.parquet" in out
    assert "input moved: processed/team_stats.parquet" in out
    assert "skipping" not in out


def test_data_version_bump_rebuilds_everything(pipeline):
    run_all(pipeline)
    src = pipeline / "pipeline.py"
    src.write_text(src.read_text().replace('data_version="v1"',
                                           'data_version="v2-later"'))

    out = run_all(pipeline)
    assert out.count("data_version bumped: v1 -> v2-later") == 2, out


def test_adding_a_read_invalidates_because_the_SOURCE_changed(pipeline):
    """The honest way to get "a new input rebuilds it".

    Not by comparing the static read set against the recorded one — those are derived by
    different mechanisms, and a disagreement is not something a rebuild can fix. By the
    step's source hash, which is a DECLARATION: a fact about the code as written,
    computed the same deterministic way at run time and by the CLI. Adding a read IS a
    source change, so the id moves.
    """
    run_all(pipeline)
    pl.DataFrame({"team": ["A", "B"], "pace": [98.0, 101.0]}).write_parquet(
        pipeline / "data" / "raw" / "pace.parquet")

    src = pipeline / "stages" / "build_stats.py"
    patched = src.read_text().replace(
        "    games.group_by",
        '    pl.read_parquet(iv.reads("raw/pace.parquet", why="team pace"))\n'
        "    games.group_by")
    assert "raw/pace.parquet" in patched
    src.write_text(patched)

    out = run(pipeline, "stages/build_stats.py")
    assert "code changed" in out, out

    # ...and now that it has rebuilt, the new input moves it on its own.
    assert "is current — skipping" in run(pipeline, "stages/build_stats.py")
    pl.DataFrame({"team": ["A", "B"], "pace": [1.0, 2.0]}).write_parquet(
        pipeline / "data" / "raw" / "pace.parquet")
    assert "input moved: raw/pace.parquet" in run(pipeline, "stages/build_stats.py")


def test_a_failing_stage_stamps_nothing_and_stays_stale(pipeline):
    src = pipeline / "stages" / "build_stats.py"
    patched = src.read_text().replace(
        '        pl.col("pts").sum()).write_parquet(out)',
        '        pl.col("pts").sum()).write_parquet(out)\n'
        "    raise RuntimeError('the fit diverged')")
    assert "the fit diverged" in patched
    src.write_text(patched)

    out = run(pipeline, "stages/build_stats.py", check=False)
    assert "the fit diverged" in out
    assert "processed/team_stats.parquet" not in state(pipeline)
    assert (pipeline / "data" / "processed" / "team_stats.parquet").exists(), \
        "the partial file is on disk — which is exactly why the stamp must not be"

    src.write_text(textwrap.dedent(STATS).lstrip())          # fix the stage
    assert "never stamped" in run(pipeline, "stages/build_stats.py")


def test_code_true_rebuilds_when_the_function_body_changes(project, pipeline):
    """Opt-in source hashing. Shallow by design — it sees this function, not its helpers."""
    src = pipeline / "stages" / "build_stats.py"
    src.write_text(src.read_text().replace(
        'why="season points by team; the rating denominator")',
        'why="season points by team; the rating denominator", code=True)'))
    run(pipeline, "stages/build_stats.py")
    assert "is current — skipping" in run(pipeline, "stages/build_stats.py")

    # A comment and reformatting are not a change: the hash is over the parsed tree.
    src.write_text(src.read_text().replace(
        "    games = pl.read_parquet(iv.reads(",
        "    # a comment, which changes nothing\n    games = pl.read_parquet(iv.reads("))
    assert "is current — skipping" in run(pipeline, "stages/build_stats.py")

    # Changing what it computes is.
    src.write_text(src.read_text().replace('pl.col("pts").sum()',
                                           'pl.col("pts").max()'))
    assert "code changed" in run(pipeline, "stages/build_stats.py")


def test_the_trace_records_what_ran(pipeline, monkeypatch):
    trace = pipeline / ".invalidator" / "trace.ndjson"
    monkeypatch.setenv("INVALIDATOR_TRACE", str(trace))
    run_all(pipeline)

    events = [json.loads(l) for l in trace.read_text().splitlines() if l.strip()]
    pairs = {(e["node"], e["op"], e["rel"]) for e in events if e["kind"] == "io"}
    assert ("stages/build_stats.py", "read", "raw/games.parquet") in pairs
    assert ("stages/build_ratings.py", "write", "processed/ratings.parquet") in pairs
    assert all(e.get("why") for e in events if e["kind"] == "io")


def test_drift_is_empty_when_the_code_and_the_run_agree(pipeline, monkeypatch):
    trace = pipeline / ".invalidator" / "trace.ndjson"
    monkeypatch.setenv("INVALIDATOR_TRACE", str(trace))
    run_all(pipeline)

    r = subprocess.run([VENV_PY, "-m", "invalidator.cli", "drift"], cwd=pipeline,
                       env={**os.environ, "PYTHONPATH": str(pipeline), "NO_COLOR": "1"},
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "no drift" in r.stdout


def test_a_read_the_scan_cannot_see_does_not_loop(pipeline):
    """A read reached through a dict dispatch is invisible to any static walk.

    If the static set governed staleness, the run would record an input the scan does not
    know about, the comparison would say 'input removed', the rebuild would record it
    again, and the stage would NEVER skip — correct output, no error, and the entire
    point of the cache silently gone. It mirrors, too: a read inside an untaken branch is
    declared and never recorded.

    This is a CONSTRUCTED case, not one observed in the wild. The mismatch that first
    looked like this turned out to be `bookkeeping()` failing to suppress input
    registration, which is fixed. It is kept because the hazard is real whether or not it
    has bitten yet.
    """
    src = pipeline / "stages" / "build_stats.py"
    src.write_text(textwrap.dedent('''
        """Season points by team."""
        import polars as pl
        from pipeline import iv

        def _hidden():
            return iv.reads("raw/extra.parquet", why="reached through a dict")

        _DISPATCH = {"go": _hidden}

        @iv.step("processed/team_stats.parquet", why="season points by team")
        def build(out):
            games = pl.read_parquet(iv.reads("raw/games.parquet", why="one row per game"))
            pl.read_parquet(_DISPATCH["go"]())          # the scan cannot follow this
            games.group_by("team", maintain_order=True).agg(
                pl.col("pts").sum()).write_parquet(out)

        build()
    ''').lstrip())
    pl.DataFrame({"x": [1]}).write_parquet(pipeline / "data" / "raw" / "extra.parquet")

    assert "not on disk" in run(pipeline, "stages/build_stats.py")
    for _ in range(3):
        out = run(pipeline, "stages/build_stats.py")
        assert "is current — skipping" in out, out

    # ...and the input the scan never saw still invalidates, because the RUN recorded it.
    pl.DataFrame({"x": [2]}).write_parquet(pipeline / "data" / "raw" / "extra.parquet")
    assert "input moved: raw/extra.parquet" in run(pipeline, "stages/build_stats.py")
