from __future__ import annotations

import polars as pl
import pytest

from tyke.core import Pipeline


@pytest.fixture
def tyke(tmp_path, monkeypatch):
    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                    project=tmp_path)


def _pages(tyke, seed, produced, seen):
    @tyke.data(dataset="derived/pages/", part="player", split=True, ext=".html",
             why="one cached page per player", external={"site": "the pages"},
             version=1)
    def pages(s=tyke.all_of(seed, why="an upstream that moves the key"),
              have=tyke.own_last_copy(as_paths=True, why="what a previous run fetched")):
        seen.append(sorted(str(x).rsplit("/", 1)[-1].split(".")[0] for x in (have or ())))
        return dict(produced)

    return pages


def _write_seed(tyke, n):
    with tyke.writes("raw/seed/", why="an upstream feed") as out:
        pl.DataFrame({"n": [n]}).write_parquet(out)


def _seed(tyke, n):
    _write_seed(tyke, n)
    return tyke.source("raw/seed/", why="an upstream feed")


def _parts(tyke):
    return sorted(p.name.split(".")[0] for p in
                  (tyke.resolve_out("derived/pages/")).glob("*.html"))


def test_split_keeps_partitions_it_did_not_return(tyke):
    seed = _seed(tyke, 1)
    produced, seen = {"a": "AAA"}, []
    pages = _pages(tyke, seed, produced, seen)
    pages.build(None)
    assert _parts(tyke) == ["player=a"]

    produced.clear()
    produced["b"] = "BBB"
    _write_seed(tyke, 2)
    pages.build(None)

    assert _parts(tyke) == ["player=a", "player=b"]
    assert seen[-1] == ["player=a"]


def test_an_empty_page_is_a_committed_shard_not_an_absence(tyke):
    seed = _seed(tyke, 1)
    pages = _pages(tyke, seed, {"gone": ""}, [])
    pages.build(None)

    shard = next((tyke.resolve_out("derived/pages/")).glob("*.html"))
    assert shard.stat().st_size == 0
    assert shard.name.split(".")[0] == "player=gone"
    assert tyke.verify("derived/pages/") == []


def test_verify_does_not_read_a_html_dataset_as_parquet(tyke):
    seed = _seed(tyke, 1)
    pages = _pages(tyke, seed, {"a": "AAA", "b": "BB"}, [])
    pages.build(None)

    assert tyke.verify("derived/pages/") == []


def test_a_pipeline_is_usable_by_the_guard_during_its_own_init(tmp_path):
    import tyke.core as core

    seen = []
    real = core._check_declared
    core._check_declared = lambda t: seen.append(t)
    try:
        Pipeline(tree=tmp_path / "data", project=tmp_path)
    finally:
        core._check_declared = real
    assert core._ACTIVE[-1]._in_step is False


def test_split_returning_nothing_new_keeps_every_shard(tyke):
    seed = _seed(tyke, 1)
    produced, seen = {"a": "AAA"}, []
    pages = _pages(tyke, seed, produced, seen)
    pages.build(None)

    produced.clear()
    _write_seed(tyke, 2)
    pages.build(None)

    assert _parts(tyke) == ["player=a"]
    assert tyke.verify("derived/pages/") == []
