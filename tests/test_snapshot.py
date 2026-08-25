

from __future__ import annotations

import polars as pl
import pytest

from iv import shards as _sh
from iv.core import Invalidator


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Invalidator(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def seed(iv, dataset, part=None, n=3, why="an upstream feed"):

    with iv.writes(dataset, why=why, part=part) as out:
        pl.DataFrame({"a": range(n)}).write_parquet(out)
    return iv._sources.get(dataset) or iv.source(dataset, why=why)


def counting(monkeypatch):

    calls = []
    real = _sh.list_shards
    monkeypatch.setattr(_sh, "list_shards",
                        lambda d: (calls.append(str(d)), real(d))[1])
    return calls


def seasons(iv, keys):
    for s in keys:
        seed(iv, "raw/box/", part={"season": s})

    iv._assets.clear()

    @iv.data(dataset="processed/features/", why="per-season features", part="season")
    def features(box=iv.same_part(iv._sources["raw/box/"], why="this season")):
        return box

    features.for_each(keys)


def test_a_snapshot_gives_the_same_answer_as_the_live_tree(iv):
    seasons(iv, ["2021", "2022", "2023"])
    live = {p: iv.why_stale("processed/features/", {"season": p})
            for p in ("2021", "2022", "2023")}
    with iv.snapshot():
        cached = {p: iv.why_stale("processed/features/", {"season": p})
                  for p in ("2021", "2022", "2023")}
    assert cached == live
    assert set(live.values()) == {None}, "everything was just built"


def test_a_snapshot_sees_staleness_it_was_opened_after(iv):


    seasons(iv, ["2021", "2022"])
    seed(iv, "raw/box/", part={"season": "2021"}, n=99)
    with iv.snapshot():
        assert iv.why_stale("processed/features/", {"season": "2021"}) is not None
        assert iv.why_stale("processed/features/", {"season": "2022"}) is None


def test_a_snapshot_does_not_survive_its_block(iv):
    seasons(iv, ["2021"])
    with iv.snapshot():
        assert iv.is_current("processed/features/", {"season": "2021"})
    seed(iv, "raw/box/", part={"season": "2021"}, n=99)
    assert not iv.is_current("processed/features/", {"season": "2021"}), \
        "the cache outlived the block and is answering about a tree that has changed"


def test_a_snapshot_lists_each_directory_once(iv, monkeypatch):
    keys = [str(y) for y in range(2000, 2012)]
    seasons(iv, keys)

    calls = counting(monkeypatch)
    for p in keys:
        iv.why_stale("processed/features/", {"season": p})
    live = len(calls)

    calls.clear()
    with iv.snapshot():
        for p in keys:
            iv.why_stale("processed/features/", {"season": p})
    cached = len(calls)

    assert cached < live, "a snapshot that removes no listings is not doing anything"
    assert cached == len(set(calls)), "each directory is listed at most once"
    assert live > 2 * cached, (
        f"the saving should grow with partitions; {live} live vs {cached} cached "
        f"over {len(keys)} partitions")


def test_the_saving_grows_with_partitions(iv, monkeypatch):


    def listings(keys):
        pipe = Invalidator(tree=iv.tree.parent / f"d{len(keys)}",
                           stage_dir=iv.stage_dir, project=iv.project)
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


def test_a_commit_is_never_answered_from_a_snapshot(iv):


    seed(iv, "raw/feed/", n=1)
    with iv.snapshot():
        _sh.current_shards(iv.resolve_out("raw/feed/"))
        seed(iv, "raw/feed/", n=5)
        seed(iv, "raw/feed/", n=9)

    live = _sh.list_shards(iv.resolve_out("raw/feed/"))
    assert sum(len(v) for v in live.values()) == 1, (
        "a commit inside the block read a memoised listing and left a superseded shard "
        f"behind: {sorted(s.name for v in live.values() for s in v)}")
    assert pl.read_parquet(iv.reads("raw/feed/", why="check")).height == 9


def test_snapshots_nest_without_taking_a_second_view(iv):
    seasons(iv, ["2021"])
    with iv.snapshot():
        first = iv.why_stale("processed/features/", {"season": "2021"})
        with iv.snapshot():
            assert iv.why_stale("processed/features/", {"season": "2021"}) == first
        assert iv.why_stale("processed/features/", {"season": "2021"}) == first, \
            "the inner block closed and took the outer cache with it"


def test_an_exception_closes_the_snapshot(iv):
    seasons(iv, ["2021"])
    with pytest.raises(RuntimeError):
        with iv.snapshot():
            raise RuntimeError("boom")
    seed(iv, "raw/box/", part={"season": "2021"}, n=99)
    assert not iv.is_current("processed/features/", {"season": "2021"}), \
        "the cache leaked out of a block that raised"
