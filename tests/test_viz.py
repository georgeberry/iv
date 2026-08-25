

from __future__ import annotations

import polars as pl
import pytest

from iv import Invalidator
from iv import graph as _graph

_viz = pytest.importorskip("iv.viz", reason="needs the viz extra: networkx + matplotlib")


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Invalidator(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def frame():
    return pl.DataFrame({"a": [1, 2]})


def built(iv):

    feed = iv.source("raw/feed/", why="a fetcher drops it here")
    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(feed=iv.all_of(feed, why="arrives out of band")):
        return feed

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(m=iv.all_of(mid, why="the middle")):
        return m

    with iv.writes("raw/feed/", why="out of band") as out:
        frame().write_parquet(out)
    mid(); site()
    return _graph.build(iv)


def test_every_dataset_gets_one_of_the_three_kinds(iv):
    d = _viz.to_networkx(built(iv))
    kinds = {ds: d.nodes[n]["kind"] for n in d for ds, _ in [n]}
    assert kinds == {"raw/feed/": "root",
                     "processed/mid/": "derived",
                     "dump/site/": "terminal"}
    assert set(kinds.values()) <= set(_viz.SHAPE), "a kind with no shape draws as nothing"


def test_two_stages_writing_one_dataset_are_two_nodes(iv):


    feed = iv.source("raw/feed/", why="a fetcher drops it here")

    @iv.data(dataset="processed/preds/", part={"completed": "true"}, why="played")
    def played(f=iv.all_of(feed, why="the feed")):
        return frame()

    @iv.data(dataset="processed/pre/", why="the preseason nets")
    def pre(p=iv.parts(played, completed=["true"], why="played games only")):
        return frame()

    @iv.data(dataset="processed/preds/", part={"completed": "false"}, why="not yet played")
    def upcoming(t=iv.all_of(pre, why="the team ratings")):
        return frame()

    g = _graph.build(iv)
    assert _graph.find_cycle(g) is None, "the checks say this is fine"
    d = _viz.to_networkx(g)
    assert _viz.find_cycle(d) is None, "and so must the picture"
    assert ("processed/preds/", (("completed", "true"),)) in d
    assert ("processed/preds/", (("completed", "false"),)) in d


def test_a_shard_is_labelled_with_the_partition_it_is(iv):
    assert _viz.short(("processed/preds/", (("completed", "true"),))) == \
        "preds [completed=true]"
    assert _viz.short(("processed/mid/", ())) == "mid"


def test_every_kind_has_a_distinct_shape():
    assert len(set(_viz.SHAPE.values())) == len(_viz.SHAPE)


def test_the_states_are_the_ones_status_reports():


    got = _viz.states({"a/": {"": None}, "b/": {"": "its inputs moved"}, "c/": {"": None}},
                      maybe={("c/", "")})
    assert got == {"a/": "current", "b/": "stale", "c/": "maybe"}
    assert set(got.values()) <= set(_viz.STATUS)


def test_a_dataset_with_no_producer_is_not_coloured_as_an_answer(iv):


    g = built(iv)
    status = _viz.states({n: {"": None} for n in g.produced}, maybe=set())
    assert "raw/feed/" not in status
    assert _viz.STATUS["source"] not in {_viz.STATUS[s] for s in ("current", "maybe", "stale")}


def test_every_state_has_a_distinct_colour():
    assert len(set(_viz.STATUS.values())) == len(_viz.STATUS)


def test_draw_writes_a_picture(iv, tmp_path):
    g = built(iv)
    status = _viz.states({n: {"": None} for n in g.produced}, maybe=set())
    out = _viz.draw(g, tmp_path / "dag.png", status=status)
    assert out.exists() and out.stat().st_size > 1000


def test_draw_needs_no_status_at_all(iv, tmp_path):


    out = _viz.draw(built(iv), tmp_path / "plain.png")
    assert out.exists() and out.stat().st_size > 1000


def test_a_cycle_is_cut_to_lay_out_rather_than_crashing(iv, tmp_path, monkeypatch):


    g = built(iv)
    d = _viz.to_networkx(g)
    d.add_edge(("dump/site/", ()), ("processed/mid/", ()), stage="hand-built")
    monkeypatch.setattr(_viz, "to_networkx", lambda _g: d)

    out = _viz.draw(g, tmp_path / "cyclic.png")
    assert out.exists()
