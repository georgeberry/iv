

from __future__ import annotations

import sys

import polars as pl
import pytest

from tyke import graph as _graph
from tyke.cli import _downstream_of, _stale_shards, _staleness
from tyke.core import Pipeline

from tests.conftest import write_stage


PIPE = '''
import pathlib
from tyke.core import Pipeline
HERE = pathlib.Path(__file__).resolve().parent
tyke = Pipeline(tree=HERE / "data", code=["stages"], project=HERE,
                 stage_dir=HERE / "stage")
feed = tyke.source("raw/feed/", why="a fetcher drops it here")
'''

MID = '''
from p import feed, tyke

@tyke.data(dataset="processed/mid/", why="the middle")
def build(f=tyke.all_of(feed, why="the feed")):
    return f
'''


END = '''
import mid
from p import tyke

@tyke.data(dataset="dump/site/", why="the app reads it")
def build(m=tyke.all_of(mid.build, why="the middle")):
    return m.head(1)
'''


@pytest.fixture
def project(tmp_path, monkeypatch):
    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
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


    __import__("mid"); __import__("end")
    yield tmp_path, p.tyke
    for m in ("p", "mid", "end"):
        sys.modules.pop(m, None)


def steps():
    import end
    import mid
    return {"processed/mid/": mid.build, "dump/site/": end.build}


def seed(tyke, n=3):
    with tyke.writes("raw/feed/", why="the feed") as out:
        pl.DataFrame({"a": range(n)}).write_parquet(out)


def cli_says(tyke):


    state = _staleness(tyke, _graph.build(tyke))
    return {ds: any(why for why in shards.values()) for ds, shards in state.items()}


def run_says(tyke):

    out = {}
    for ds, asset in steps().items():
        before = asset.is_current()
        asset()
        out[ds] = not before
    return out


def test_they_agree_when_everything_is_current(project):
    _, tyke = project
    seed(tyke)
    run_says(tyke)

    cli = cli_says(tyke)
    assert cli == {"processed/mid/": False, "dump/site/": False}
    assert run_says(tyke) == {"processed/mid/": False, "dump/site/": False}


def test_the_cli_and_the_run_agree_at_every_step_of_a_cascade(project):


    _, tyke = project
    seed(tyke)
    run_says(tyke)
    seed(tyke, n=9)


    assert cli_says(tyke) == {"processed/mid/": True, "dump/site/": False}

    assert not steps()["processed/mid/"].is_current(), "the CLI said this one was stale"
    steps()["processed/mid/"]()


    assert cli_says(tyke) == {"processed/mid/": False, "dump/site/": True}

    assert not steps()["dump/site/"].is_current()
    steps()["dump/site/"]()
    assert cli_says(tyke) == {"processed/mid/": False, "dump/site/": False}


def test_a_rebuild_that_does_not_move_the_bytes_stops_there(project):


    _, tyke = project
    seed(tyke)
    run_says(tyke)
    seed(tyke)

    assert cli_says(tyke) == {"processed/mid/": False, "dump/site/": False}
    assert run_says(tyke) == {"processed/mid/": False, "dump/site/": False}


def test_the_cli_reports_a_dataset_nothing_has_built(project):
    _, tyke = project
    seed(tyke)
    cli = cli_says(tyke)
    assert cli == {"processed/mid/": True, "dump/site/": True}, \
        "nothing is built yet, so every produced dataset is stale"


def test_a_deleted_output_is_stale_to_the_cli_and_to_the_run(project):
    import shutil
    _, tyke = project
    seed(tyke)
    run_says(tyke)
    shutil.rmtree(tyke.resolve_out("processed/mid/"))

    assert cli_says(tyke)["processed/mid/"] is True
    assert run_says(tyke)["processed/mid/"] is True


def test_the_cli_answer_does_not_change_under_a_snapshot(project):

    _, tyke = project
    seed(tyke)
    run_says(tyke)
    seed(tyke, n=9)

    live = cli_says(tyke)
    with tyke.snapshot():
        cached = cli_says(tyke)
    assert cached == live


BRANCHY = '''
from p import feed, tyke

@tyke.data(dataset="processed/mid/", why="the middle")
def build(one=tyke.all_of(feed, why="the upstream, one way"),
          other=tyke.all_of(feed, as_paths=True, why="the upstream, the other way")):
    return one
'''


def test_one_dataset_read_at_two_call_sites_gets_one_answer(tmp_path, monkeypatch):


    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    write_stage(tmp_path, "p.py", PIPE)
    write_stage(tmp_path, "stages/branchy.py", BRANCHY)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path / "stages"))
    for m in ("p", "branchy"):
        sys.modules.pop(m, None)
    import p
    import branchy

    seed(p.tyke)
    branchy.build()
    assert branchy.build.is_current(), "the run considers it current"

    state = _staleness(p.tyke, _graph.build(p.tyke))
    assert not any(state["processed/mid/"].values()), (
        f"the CLI disagrees with the run: {state['processed/mid/']}")

    for m in ("p", "branchy"):
        sys.modules.pop(m, None)


SHARED_A = '''
import polars as pl
from p import feed, tyke

@tyke.data(dataset="processed/preds/", part={"completed": "true"},
         why="one row per game played")
def played(df=tyke.all_of(feed, why="the played upstream")):
    return df
'''

SHARED_B = '''
import polars as pl
from p import tyke

other = tyke.source("raw/other/", why="a second feed, dropped in from outside")

@tyke.data(dataset="processed/preds/", part={"completed": "false"},
         why="one row per game not yet played")
def upcoming(df=tyke.all_of(other, why="the unplayed upstream")):
    return df
'''


def test_each_shard_is_judged_against_the_stage_that_writes_it(tmp_path, monkeypatch):


    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    write_stage(tmp_path, "p.py", PIPE)
    write_stage(tmp_path, "stages/a.py", SHARED_A)
    write_stage(tmp_path, "stages/b.py", SHARED_B)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.syspath_prepend(str(tmp_path / "stages"))
    for m in ("p", "a", "b"):
        sys.modules.pop(m, None)
    import a
    import b
    import p

    seed(p.tyke)
    with p.tyke.writes("raw/other/", why="the other feed") as out:
        pl.DataFrame({"a": [1]}).write_parquet(out)
    a.played()
    b.upcoming()
    assert cli_says(p.tyke)["processed/preds/"] is False


    with p.tyke.writes("raw/other/", why="the other feed") as out:
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(out)

    assert cli_says(p.tyke)["processed/preds/"] is True, "the unplayed half is stale"
    assert a.played.is_current(), "the played half reads nothing that moved"
    assert not b.upcoming.is_current(), "the unplayed half must rebuild"
    b.upcoming()
    assert cli_says(p.tyke)["processed/preds/"] is False

    for m in ("p", "a", "b"):
        sys.modules.pop(m, None)


def test_a_dataset_downstream_of_a_rebuild_is_a_maybe_not_a_red(tmp_path, monkeypatch):


    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    tyke = Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                     project=tmp_path)
    day = ["day1"]

    @tyke.data(dataset="config/today/", why="the clock", ext=".json")
    def today():
        return {"date": day[0]}

    @tyke.data(dataset="raw/feed/", why="a polled feed")
    def feed(clock=tyke.all_of(today, as_paths=True, why="poll once a day")):
        return pl.DataFrame({"a": [1]})

    @tyke.data(dataset="processed/mid/", why="the middle")
    def mid(f=tyke.all_of(feed, why="the feed")):
        return f

    @tyke.data(dataset="dump/site/", why="the app reads it")
    def site(m=tyke.all_of(mid, why="the middle")):
        return m

    today(); feed(); mid(); site()
    g = _graph.build(tyke)
    state = _staleness(tyke, g)
    assert _stale_shards(state) == set() and _downstream_of(g, state) == set()

    day[0] = "day2"
    today()
    state = _staleness(tyke, g)
    maybe = _downstream_of(g, state)
    assert _stale_shards(state) == {("raw/feed/", "")}, \
        "only the thing that reads the clock is stale"
    assert maybe == {("processed/mid/", ""), ("dump/site/", "")}, \
        "the tail is transitive, per shard, and a maybe"
    assert not any(state["processed/mid/"].values()), "a maybe is still current on disk"

    feed()
    state = _staleness(tyke, g)
    assert _stale_shards(state) == set() and _downstream_of(g, state) == set(), \
        "the maybe was right to be a maybe"


def test_only_the_shards_a_selector_reaches_may_follow(tmp_path, monkeypatch):


    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    tyke = Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                     project=tmp_path)
    feed = tyke.source("raw/feed/", why="one season of raw feed, dropped in")

    @tyke.data(dataset="raw/box/", why="one season", part="season")
    def box(season, f=tyke.same_part(feed, why="this season's feed")):
        return f

    @tyke.data(dataset="processed/cohort/", why="a fit on prior seasons only", part="season")
    def cohort(past=tyke.before_part(box, why="strictly earlier seasons")):
        return past

    for s in ("2024", "2025", "2026"):
        with tyke.writes("raw/feed/", why="the feed", part={"season": s}) as out:
            pl.DataFrame({"a": [int(s)]}).write_parquet(out)
    box.for_each(["2024", "2025", "2026"])
    cohort.for_each(["2025", "2026"])

    g = _graph.build(tyke)
    assert _stale_shards(_staleness(tyke, g)) == set()


    with tyke.writes("raw/feed/", why="the feed", part={"season": "2025"}) as out:
        pl.DataFrame({"a": [999]}).write_parquet(out)

    state = _staleness(tyke, g)
    assert _stale_shards(state) == {("raw/box/", "season=2025")}

    maybe = _downstream_of(g, state)
    assert ("processed/cohort/", "season=2026") in maybe, "2026 is fit on 2024 and 2025"
    assert ("processed/cohort/", "season=2025") not in maybe, \
        "2025 is fit on 2024 alone and cannot see 2025 move"


def test_a_dataset_reports_its_stale_and_its_following_shards_separately(tmp_path,
                                                                        monkeypatch):


    from tyke.cli import _line
    shards = {"season=2024": None, "season=2025": None,
              "season=2026": "its inputs moved"}
    maybe = {("d/", "season=2024"), ("d/", "season=2025")}
    tier, note = _line(shards, maybe, "d/")
    assert tier == "stale"
    assert "1/3 (season=2026) stale" in note and "2/3 (season=2024, season=2025) may follow" in note

    tier, note = _line({"": None}, set(), "d/")
    assert (tier, note) == ("current", "1 shard(s)")
