

from __future__ import annotations

import polars as pl
import pytest

from tyke import shards as _sh
from tyke.core import Pipeline


@pytest.fixture
def tyke(tmp_path, monkeypatch):
    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def seed(tyke, dataset, part=None, n=3, why="an upstream feed"):

    with tyke.writes(dataset, why=why, part=part) as out:
        pl.DataFrame({"a": range(n)}).write_parquet(out)
    return tyke._sources.get(dataset) or tyke.source(dataset, why=why)


def counting(monkeypatch):

    calls = []
    real = _sh.list_shards
    monkeypatch.setattr(_sh, "list_shards",
                        lambda d: (calls.append(str(d)), real(d))[1])
    return calls


def seasons(tyke, keys):
    for s in keys:
        seed(tyke, "raw/box/", part={"season": s})

    tyke._assets.clear()

    @tyke.data(dataset="processed/features/", why="per-season features", part="season")
    def features(box=tyke.same_part(tyke._sources["raw/box/"], why="this season")):
        return box

    features.for_each(keys)


def test_a_snapshot_gives_the_same_answer_as_the_live_tree(tyke):
    seasons(tyke, ["2021", "2022", "2023"])
    live = {p: tyke.why_stale("processed/features/", {"season": p})
            for p in ("2021", "2022", "2023")}
    with tyke.snapshot():
        cached = {p: tyke.why_stale("processed/features/", {"season": p})
                  for p in ("2021", "2022", "2023")}
    assert cached == live
    assert set(live.values()) == {None}, "everything was just built"


def test_a_snapshot_sees_staleness_it_was_opened_after(tyke):


    seasons(tyke, ["2021", "2022"])
    seed(tyke, "raw/box/", part={"season": "2021"}, n=99)
    with tyke.snapshot():
        assert tyke.why_stale("processed/features/", {"season": "2021"}) is not None
        assert tyke.why_stale("processed/features/", {"season": "2022"}) is None


def test_a_snapshot_does_not_survive_its_block(tyke):
    seasons(tyke, ["2021"])
    with tyke.snapshot():
        assert tyke.is_current("processed/features/", {"season": "2021"})
    seed(tyke, "raw/box/", part={"season": "2021"}, n=99)
    assert not tyke.is_current("processed/features/", {"season": "2021"}), \
        "the cache outlived the block and is answering about a tree that has changed"


def test_a_snapshot_lists_each_directory_once(tyke, monkeypatch):
    keys = [str(y) for y in range(2000, 2012)]
    seasons(tyke, keys)

    calls = counting(monkeypatch)
    for p in keys:
        tyke.why_stale("processed/features/", {"season": p})
    live = len(calls)

    calls.clear()
    with tyke.snapshot():
        for p in keys:
            tyke.why_stale("processed/features/", {"season": p})
    cached = len(calls)

    assert cached < live, "a snapshot that removes no listings is not doing anything"
    assert cached == len(set(calls)), "each directory is listed at most once"
    assert live > 2 * cached, (
        f"the saving should grow with partitions; {live} live vs {cached} cached "
        f"over {len(keys)} partitions")


def test_the_saving_grows_with_partitions(tyke, monkeypatch):


    def listings(keys):
        pipe = Pipeline(tree=tyke.tree.parent / f"d{len(keys)}",
                           stage_dir=tyke.stage_dir, project=tyke.project)
        seasons(pipe, keys)
        calls = counting(monkeypatch)
        with pipe.snapshot():
            for p in keys:
                pipe.why_stale("processed/features/", {"season": p})
        return len(calls)

    small = listings([str(y) for y in range(2000, 2006)])
    big = listings([str(y) for y in range(2000, 2018)])
    assert small == big, (
        f"under a snapshot the listing count is per-DIRECTORY, not per-partition; "
        f"got {small} for 6 partitions and {big} for 18")


def test_a_commit_is_never_answered_from_a_snapshot(tyke):


    seed(tyke, "raw/feed/", n=1)
    with tyke.snapshot():
        _sh.current_shards(tyke.resolve_out("raw/feed/"))
        seed(tyke, "raw/feed/", n=5)
        seed(tyke, "raw/feed/", n=9)

    live = _sh.list_shards(tyke.resolve_out("raw/feed/"))
    assert sum(len(v) for v in live.values()) == 1, (
        "a commit inside the block read a memoised listing and left a superseded shard "
        f"behind: {sorted(s.name for v in live.values() for s in v)}")
    assert pl.read_parquet(tyke.reads("raw/feed/", why="check")).height == 9


def test_commits_reuse_and_advance_a_snapshot_listing(tyke, monkeypatch):
    seed(tyke, "raw/feed/", n=1)
    calls = counting(monkeypatch)
    with tyke.snapshot():
        _sh.current_shards(tyke.resolve_out("raw/feed/"))
        seed(tyke, "raw/feed/", n=5)
        seed(tyke, "raw/feed/", n=9)
    assert calls == [str(tyke.resolve_out("raw/feed/"))]


def test_a_snapshot_sees_a_commit_made_inside_it(tyke):


    seasons(tyke, ["2021", "2022"])
    with tyke.snapshot():
        assert tyke.is_current("processed/features/", {"season": "2021"})
        seed(tyke, "raw/box/", part={"season": "2021"}, n=99)
        assert not tyke.is_current("processed/features/", {"season": "2021"}), (
            "a stage wrote inside the block and the next staleness check was answered "
            "from the listing taken before it — which is what stops `tyke run` from "
            "holding a snapshot at all")
        assert tyke.is_current("processed/features/", {"season": "2022"})


def test_snapshots_nest_without_taking_a_second_view(tyke):
    seasons(tyke, ["2021"])
    with tyke.snapshot():
        first = tyke.why_stale("processed/features/", {"season": "2021"})
        with tyke.snapshot():
            assert tyke.why_stale("processed/features/", {"season": "2021"}) == first
        assert tyke.why_stale("processed/features/", {"season": "2021"}) == first, \
            "the inner block closed and took the outer cache with it"


def test_an_exception_closes_the_snapshot(tyke):
    seasons(tyke, ["2021"])
    with pytest.raises(RuntimeError):
        with tyke.snapshot():
            raise RuntimeError("boom")
    seed(tyke, "raw/box/", part={"season": "2021"}, n=99)
    assert not tyke.is_current("processed/features/", {"season": "2021"}), \
        "the cache leaked out of a block that raised"
