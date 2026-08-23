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
    kinds = {ds: d.nodes[n]["kind"] for n in d for ds, _ in [n]}
    assert kinds == {"raw/feed/": "root",
                     "processed/mid/": "derived",
                     "dump/site/": "terminal"}
    assert set(kinds.values()) <= set(_viz.SHAPE), "a kind with no shape draws as nothing"


def test_two_stages_writing_one_dataset_are_two_nodes(iv):
    """The played and unplayed halves of a prediction table are different shards. Drawn as
    one node they are the same thing, and a stage that reads one and writes the other reads
    as a cycle it does not have — which is what `iv check` correctly said it did not."""
    @iv.data("processed/preds/", part={"completed": "true"}, why="played")
    def played(feed=iv.all_of("raw/feed/", why="the feed")):
        return frame()

    @iv.data("processed/pre/", why="the preseason nets")
    def pre(p=iv.parts("processed/preds/", completed=["true"], why="played games only")):
        return frame()

    @iv.data("processed/preds/", part={"completed": "false"}, why="not yet played")
    def upcoming(t=iv.all_of("processed/pre/", why="the team ratings")):
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


# ── colour: the same three answers `iv status` gives ──────────────────────────

def test_the_states_are_the_ones_status_reports():
    """`states` is shared with the CLI so the picture and the report cannot drift into
    naming the same three things differently."""
    got = _viz.states({"a/": {"": None}, "b/": {"": "its inputs moved"}, "c/": {"": None}},
                      maybe={("c/", "")})
    assert got == {"a/": "current", "b/": "stale", "c/": "maybe"}
    assert set(got.values()) <= set(_viz.STATUS)


def test_a_dataset_with_no_producer_is_not_coloured_as_an_answer(iv):
    """A source has no stage to ask, so `iv status` does not report it and the picture
    does not pretend to know — it is grey."""
    g = built(iv)
    status = _viz.states({n: {"": None} for n in g.produced}, maybe=set())
    assert "raw/feed/" not in status
    assert _viz.STATUS["source"] not in {_viz.STATUS[s] for s in ("current", "maybe", "stale")}


def test_every_state_has_a_distinct_colour():
    assert len(set(_viz.STATUS.values())) == len(_viz.STATUS)


# ── it draws ──────────────────────────────────────────────────────────────────

def test_draw_writes_a_picture(iv, tmp_path):
    g = built(iv)
    status = _viz.states({n: {"": None} for n in g.produced}, maybe=set())
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


def test_a_shard_takes_the_colour_of_its_dataset(iv, tmp_path):
    """Nodes are (dataset, shard) since a dataset can have several writers, and `iv status`
    answers per dataset — so the lookup has to go through the dataset, or every node is
    grey and the picture says nothing."""
    @iv.data("processed/preds/", part={"completed": "true"}, why="played")
    def played(feed=iv.all_of("raw/feed/", why="the feed")):
        return frame()

    @iv.data("processed/preds/", part={"completed": "false"}, why="not yet played")
    def upcoming(feed=iv.all_of("raw/feed/", why="the feed")):
        return frame()

    g = _graph.build(iv)
    status = _viz.states({"processed/preds/": {"": "its inputs moved"}}, maybe=set())
    colours = {_viz.STATUS[status.get(n[0], "source")] for n in _viz.to_networkx(g)}
    assert _viz.STATUS["stale"] in colours, "both shards take the dataset's answer"
    assert _viz.STATUS["source"] in colours, "and raw/feed/ has no answer to take"
