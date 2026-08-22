"""`iv status` must give the same answer the run would.

The CLI cannot ask a running step what it reads, so it reads each stage's upstreams off the
static scan, while the run reads them off the decorated function. Two ways of arriving at
one fact, and nothing until now checked they arrive at the same place. If they drift, the
CLI says `current` about something the run is about to rebuild — or worse, says `current`
because it could not find the question at all.
"""
from __future__ import annotations

import sys

import polars as pl
import pytest

from iv import graph as _graph
from iv.cli import _staleness
from tests.conftest import write_stage


PIPE = '''
import pathlib
from iv.core import Pipeline
HERE = pathlib.Path(__file__).resolve().parent
iv = Pipeline(root=HERE / "data", source_dirs=["stages"], project_root=HERE,
              stage_dir=HERE / "stage")
'''

MID = '''
import polars as pl
from p import iv

@iv.step(why="the middle")
def build():
    df = pl.read_parquet(iv.reads("raw/feed/", why="the feed"))
    with iv.writes("processed/mid/", why="the middle") as out:
        df.write_parquet(out)
'''

END = '''
import polars as pl
from p import iv

@iv.step(why="the app reads it")
def build():
    df = pl.read_parquet(iv.reads("processed/mid/", why="the middle"))
    with iv.writes("dump/site/", why="the app reads it", terminal=True) as out:
        df.head(1).write_parquet(out)
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    write_stage(tmp_path, "p.py", PIPE)
    write_stage(tmp_path, "stages/mid.py", MID)
    write_stage(tmp_path, "stages/end.py", END)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path / "stages"))
    for m in ("p", "mid", "end"):
        sys.modules.pop(m, None)
    import p
    yield tmp_path, p.iv
    for m in ("p", "mid", "end"):
        sys.modules.pop(m, None)


def steps():
    import end
    import mid
    return {"processed/mid/": mid.build, "dump/site/": end.build}


def seed(iv, n=3):
    with iv.writes("raw/feed/", why="the feed") as out:
        pl.DataFrame({"a": range(n)}).write_parquet(out)


def cli_says(iv):
    """What `iv status` would print: dataset -> stale?

    Produced datasets only. A root nothing declares as an output has no stage to ask, so
    it is not in the report — `raw/feed/` here.
    """
    reasons, _ = _staleness(iv, _graph.build(iv))
    return {k: v is not None for k, v in reasons.items()}


def run_says(iv):
    """What actually happens when you run it: dataset -> did the stage run?"""
    return {ds: fn() is True for ds, fn in steps().items()}


def test_they_agree_when_everything_is_current(project):
    _, iv = project
    seed(iv)
    run_says(iv)                                   # build it all

    cli = cli_says(iv)
    assert cli == {"processed/mid/": False, "dump/site/": False}
    assert run_says(iv) == {"processed/mid/": False, "dump/site/": False}


def test_the_cli_and_the_run_agree_at_every_step_of_a_cascade(project):
    """The CLI answers about the tree AS IT STANDS, and so does the run.

    A moved feed does not make the terminal dump stale — not yet. Its upstream shard has
    not moved, only the thing that builds that shard. So the honest invariant is lockstep:
    at each point, what the CLI calls stale is exactly what runs next. (`iv plan` is what
    reports the not-yet-stale tail, as `maybe — downstream of a rebuild`.)
    """
    _, iv = project
    seed(iv)
    run_says(iv)
    seed(iv, n=9)                                  # the feed moves

    # The middle is stale; the dump is not, because the middle has not been rebuilt yet.
    assert cli_says(iv) == {"processed/mid/": True, "dump/site/": False}

    assert steps()["processed/mid/"]() is True, "the CLI said this one was stale"

    # Rebuilding the middle moved its bytes, and only now is the dump stale.
    assert cli_says(iv) == {"processed/mid/": False, "dump/site/": True}

    assert steps()["dump/site/"]() is True
    assert cli_says(iv) == {"processed/mid/": False, "dump/site/": False}


def test_a_rebuild_that_does_not_move_the_bytes_stops_there(project):
    """The early cutoff, checked through the CLI rather than the run.

    Re-seeding identical data re-commits the same shard, so the middle re-runs and the dump
    must not — and the CLI has to say so, or `iv status` would send you rebuilding a tail
    that is genuinely finished.
    """
    _, iv = project
    seed(iv)
    run_says(iv)
    seed(iv)                                       # same data, same fingerprint

    assert cli_says(iv) == {"processed/mid/": False, "dump/site/": False}
    assert run_says(iv) == {"processed/mid/": False, "dump/site/": False}


def test_the_cli_reports_a_dataset_nothing_has_built(project):
    _, iv = project
    seed(iv)
    cli = cli_says(iv)
    assert cli == {"processed/mid/": True, "dump/site/": True}, \
        "nothing is built yet, so every produced dataset is stale"


def test_a_deleted_output_is_stale_to_the_cli_and_to_the_run(project):
    import shutil
    _, iv = project
    seed(iv)
    run_says(iv)
    shutil.rmtree(iv.resolve_out("processed/mid/"))

    assert cli_says(iv)["processed/mid/"] is True
    assert run_says(iv)["processed/mid/"] is True


def test_the_cli_answer_does_not_change_under_a_snapshot(project):
    """The memoised path and the live path are the same report."""
    _, iv = project
    seed(iv)
    run_says(iv)
    seed(iv, n=9)

    live = cli_says(iv)
    with iv.snapshot():
        cached = cli_says(iv)
    assert cached == live
