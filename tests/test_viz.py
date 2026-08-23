"""The picture: colour is status, shape is kind.

Two channels because they are two questions. Whether a dataset is current changes every
day; what KIND of thing it is does not. Sharing one channel would mean neither could be
read off the page.
"""
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
    """A source, a derived dataset and a terminal one — one of each shape."""
    @iv.data("processed/mid/", why="the middle")
    def mid(feed=iv.all_of("raw/feed/", why="arrives out of band")):
        return feed

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(m=iv.all_of("processed/mid/", why="the middle")):
        return m

    with iv.writes("raw/feed/", why="out of band") as out:
        frame().write_parquet(out)
    mid(); site()
    return _graph.build(iv)


# ── shape: what kind of dataset it is ─────────────────────────────────────────

def test_every_dataset_gets_one_of_the_three_kinds(iv):
    d = _viz.to_networkx(built(iv))
    kinds = {n: d.nodes[n]["kind"] for n in d}
    assert kinds == {"raw/feed/": "root",
                     "processed/mid/": "derived",
                     "dump/site/": "terminal"}
    assert set(kinds.values()) <= set(_viz.SHAPE), "a kind with no shape draws as nothing"


def test_every_kind_has_a_distinct_shape():
    assert len(set(_viz.SHAPE.values())) == len(_viz.SHAPE)


# ── colour: the same three answers `iv status` gives ──────────────────────────

def test_the_states_are_the_ones_status_reports():
    """`states` is shared with the CLI so the picture and the report cannot drift into
    naming the same three things differently."""
    got = _viz.states({"a/": None, "b/": "its inputs moved", "c/": None}, maybe={"c/"})
    assert got == {"a/": "current", "b/": "stale", "c/": "maybe"}
    assert set(got.values()) <= set(_viz.STATUS)


def test_a_dataset_with_no_producer_is_not_coloured_as_an_answer(iv):
    """A source has no stage to ask, so `iv status` does not report it and the picture
    does not pretend to know — it is grey."""
    g = built(iv)
    reasons = {n: None for n in g.produced}
    status = _viz.states(reasons, maybe=set())
    assert "raw/feed/" not in status
    assert _viz.STATUS["source"] not in {_viz.STATUS[s] for s in ("current", "maybe", "stale")}


def test_every_state_has_a_distinct_colour():
    assert len(set(_viz.STATUS.values())) == len(_viz.STATUS)


# ── it draws ──────────────────────────────────────────────────────────────────

def test_draw_writes_a_picture(iv, tmp_path):
    g = built(iv)
    status = _viz.states({n: None for n in g.produced}, maybe=set())
    out = _viz.draw(g, tmp_path / "dag.png", status=status)
    assert out.exists() and out.stat().st_size > 1000


def test_draw_needs_no_status_at_all(iv, tmp_path):
    """`iv viz --plain` does not read the tree, which is what you want when the data is
    somewhere slow or not there yet."""
    out = _viz.draw(built(iv), tmp_path / "plain.png")
    assert out.exists() and out.stat().st_size > 1000


def test_a_cycle_is_cut_to_lay_out_rather_than_crashing(iv, tmp_path):
    """The picture is most wanted when the graph is wrong, so it has to survive one."""
    @iv.data("processed/a/", why="a")
    def a(b=iv.all_of("processed/b/", why="b")):
        return b

    @iv.data("processed/b/", why="b")
    def b(a=iv.all_of("processed/a/", why="a")):
        return a

    out = _viz.draw(_graph.build(iv), tmp_path / "cyclic.png")
    assert out.exists()
