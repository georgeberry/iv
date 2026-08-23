"""The clickable graph: what the page is handed, checked as a dict.

The page itself is a template with a JSON blob in it, so everything worth asserting is in
the blob. These run against a pipeline built here — never a real tree, which moves while
the assertions are being made.
"""
from __future__ import annotations

import json

import polars as pl
import pytest

from iv import Invalidator
from iv import graph as _graph
from iv import shards as _sh
from iv import web as _web
from iv.cli import _downstream_of, _staleness

_viz = pytest.importorskip("iv.viz", reason="needs the viz extra: networkx")


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Invalidator(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def frame():
    return pl.DataFrame({"a": [1, 2]})


@pytest.fixture
def built(iv):
    """A source, a partitioned dataset, and one dataset TWO stages write — the shape the
    per-partition node exists for."""
    feed = iv.source("raw/feed/", why="a fetcher drops it here")

    @iv.data(dataset="processed/box/", why="one season", part="season")
    def box(season, f=iv.all_of(feed, why="the feed")):
        return frame()

    @iv.data(dataset="processed/preds/", part={"completed": "true"},
             why="games already played")
    def played(b=iv.all_of(box, why="the box")):
        return frame()

    @iv.data(dataset="processed/preds/", part={"completed": "false"},
             why="games not yet played")
    def upcoming(b=iv.all_of(box, why="the box")):
        return frame()

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(p=iv.all_of(played, why="both halves")):
        return frame()

    with iv.writes("raw/feed/", why="out of band") as out:
        frame().write_parquet(out)
    box.for_each(["2023", "2024"])
    played(); upcoming(); site()
    return iv


def load(iv):
    g = _graph.build(iv)
    with _sh.snapshot():
        state = _staleness(iv, g)
        maybe = _downstream_of(g, state)
    return g, state, maybe, _web.payload(g, _viz.states(state, maybe), state, maybe)


# ── the page and `iv status` count the same shards ────────────────────────────

def test_the_page_counts_the_shards_iv_status_counts(built):
    """One number, computed once. Two derivations of one fact is how they drift."""
    g, state, maybe, p = load(built)
    assert sum(len(n["shards"]) for n in p["nodes"]) == sum(len(v) for v in state.values())


def test_a_dataset_two_stages_write_is_two_nodes_with_a_shard_each(built):
    """Each node owns the shard IT writes. Handed the whole dataset's shards, the played
    half would report the unplayed half's staleness as its own — and the total would count
    every shared shard once per writer."""
    _, _, _, p = load(built)
    halves = [n for n in p["nodes"] if n["dataset"] == "processed/preds/"]
    assert len(halves) == 2
    for h in halves:
        assert [s["part"] for s in h["shards"]] == [f"completed={h['part']['completed']}"]


def test_a_partitioned_dataset_is_one_node_holding_every_shard(built):
    """The split is for a dataset several STAGES write, not for one stage's partitions —
    otherwise a twenty-season table would be twenty nodes."""
    _, _, _, p = load(built)
    box = [n for n in p["nodes"] if n["dataset"] == "processed/box/"]
    assert len(box) == 1
    assert sorted(s["part"] for s in box[0]["shards"]) == ["season=2023", "season=2024"]


# ── what a click has to answer ────────────────────────────────────────────────

def test_every_node_carries_what_the_panel_shows(built):
    _, _, _, p = load(built)
    for n in p["nodes"]:
        assert n["why"], f"{n['dataset']} has no why= to show"
        assert n["status"] in _viz.STATUS
        assert n["kind"] in _viz.SHAPE
        assert isinstance(n["writers"], list)


def test_a_source_says_nothing_here_writes_it(built):
    _, _, _, p = load(built)
    feed, = [n for n in p["nodes"] if n["dataset"] == "raw/feed/"]
    assert feed["writers"] == [] and feed["kind"] == "root"
    assert feed["status"] == "source", "no stage to ask, so not an answer"
    assert feed["why"] == "a fetcher drops it here", "the source's own why, not a stage's"


def test_the_reason_a_shard_is_stale_is_carried_verbatim(built):
    """The panel shows what `iv status` prints, not a paraphrase of it."""
    with built.writes("raw/feed/", why="the feed moves") as out:
        pl.DataFrame({"a": [9, 9, 9]}).write_parquet(out)
    g, state, _, p = load(built)
    shown = {(n["dataset"], s["part"]): s["reason"] for n in p["nodes"] for s in n["shards"]}
    assert any(v for v in shown.values()), "something has to be stale for this to mean much"
    for ds, shards in state.items():
        for part, reason in shards.items():
            assert shown[(ds, part or "(one shard)")] == reason


def test_upstream_and_downstream_are_reachable_from_the_edges(built):
    """The page walks edges for the cone, so the edges have to BE the graph."""
    g, _, _, p = load(built)
    ids = {n["id"] for n in p["nodes"]}
    for e in p["edges"]:
        assert e["source"] in ids and e["target"] in ids
    d = _viz.to_networkx(g)
    assert len(p["edges"]) == d.number_of_edges()
    assert len(p["nodes"]) == d.number_of_nodes()


def test_the_counts_add_up_to_the_nodes(built):
    _, _, _, p = load(built)
    assert sum(p["counts"].values()) == len(p["nodes"])


# ── it writes a page ──────────────────────────────────────────────────────────

def test_the_written_page_carries_the_payload(built, tmp_path):
    g, state, maybe, p = load(built)
    out = _web.write(g, tmp_path / "dag.html", status=_viz.states(state, maybe),
                     state=state, maybe=maybe)
    text = out.read_text()
    assert text.startswith("<!doctype html>") and "__DATA__" not in text
    assert "__TITLE__" not in text and "__COLOURS__" not in text
    for n in p["nodes"]:
        assert json.dumps(n["dataset"])[1:-1] in text


def test_the_page_is_one_file(built, tmp_path):
    """No sidecar, no server. `iv viz --html` writes a thing you can open or send."""
    g, _, _, _ = load(built)
    _web.write(g, tmp_path / "dag.html")
    assert [f.name for f in tmp_path.iterdir() if f.is_file()] == ["dag.html"]
