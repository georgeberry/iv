

from __future__ import annotations

import json

import polars as pl
import pytest

from iv import Pipeline
from iv import graph as _graph
from iv import shards as _sh
from iv import web as _web
from iv.cli import _downstream_of, _staleness

_viz = pytest.importorskip("iv.viz", reason="needs the viz extra: networkx")


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def frame():
    return pl.DataFrame({"a": [1, 2]})


@pytest.fixture
def built(iv):


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


def test_the_page_counts_the_shards_iv_status_counts(built):

    g, state, maybe, p = load(built)
    assert sum(len(n["shards"]) for n in p["nodes"]) == sum(len(v) for v in state.values())


def test_a_dataset_two_stages_write_is_two_nodes_with_a_shard_each(built):


    _, _, _, p = load(built)
    halves = [n for n in p["nodes"] if n["dataset"] == "processed/preds/"]
    assert len(halves) == 2
    for h in halves:
        assert [s["part"] for s in h["shards"]] == [f"completed={h['part']['completed']}"]


def test_a_partitioned_dataset_is_one_node_holding_every_shard(built):


    _, _, _, p = load(built)
    box = [n for n in p["nodes"] if n["dataset"] == "processed/box/"]
    assert len(box) == 1
    assert box[0]["partitionKeys"] == ["season"]
    assert sorted(s["part"] for s in box[0]["shards"]) == ["season=2023", "season=2024"]


def test_the_page_lists_every_partition_key_in_use(built):
    _, _, _, p = load(built)
    assert p["partitionKeys"] == ["completed", "season"]


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

    with built.writes("raw/feed/", why="the feed moves") as out:
        pl.DataFrame({"a": [9, 9, 9]}).write_parquet(out)
    g, state, _, p = load(built)
    shown = {(n["dataset"], s["part"]): s["reason"] for n in p["nodes"] for s in n["shards"]}
    assert any(v for v in shown.values()), "something has to be stale for this to mean much"
    for ds, shards in state.items():
        for part, reason in shards.items():
            assert shown[(ds, part or "(one shard)")] == reason


def test_upstream_and_downstream_are_reachable_from_the_edges(built):

    g, _, _, p = load(built)
    ids = {n["id"] for n in p["nodes"]}
    for e in p["edges"]:
        assert e["source"] in ids and e["target"] in ids
    assert len(p["nodes"]) == _viz.to_networkx(g).number_of_nodes()


def test_every_declared_edge_is_drawn(iv):


    feed = iv.source("raw/feed/", why="a fetcher drops it here")

    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(f=iv.all_of(feed, why="the feed")):
        return frame()

    @iv.data(dataset="processed/end/", why="reads both, so feed->end is implied")
    def end(f=iv.all_of(feed, why="the feed"), m=iv.all_of(mid, why="the middle")):
        return frame()

    g = _graph.build(iv)
    assert _viz.to_networkx(g).number_of_edges() == 3
    assert len(_web.payload(g)["edges"]) == 3, "every declared read is drawn"


def test_the_counts_add_up_to_the_nodes(built):
    _, _, _, p = load(built)
    assert sum(p["counts"].values()) == len(p["nodes"])


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

    g, _, _, _ = load(built)
    _web.write(g, tmp_path / "dag.html")
    assert [f.name for f in tmp_path.iterdir() if f.is_file()] == ["dag.html"]


def test_every_edge_points_right(built):


    g, _, _, p = load(built)
    at = {n["id"]: n["position"] for n in p["nodes"]}
    for e in p["edges"]:
        assert at[e["source"]]["x"] < at[e["target"]]["x"], \
            f"{e['source']} -> {e['target']} does not point right"


def test_outputs_of_one_stage_share_a_column(iv):


    feed = iv.source("raw/feed/", why="a fetcher drops it here")

    @iv.step(output={"a": "processed/a/", "b": "processed/b/", "c": "processed/c/"},
             why="one fit, three tables")
    def fit(f=iv.all_of(feed, why="the feed")):
        return {"a": frame(), "b": frame(), "c": frame()}

    p = _web.payload(_graph.build(iv))
    xs = {n["position"]["x"] for n in p["nodes"] if n["dataset"] != "raw/feed/"}
    assert len(xs) == 1, "three outputs of one stage, three columns"


def test_a_column_is_as_wide_as_its_own_longest_label(iv):


    feed = iv.source("raw/x/", why="short name")

    @iv.data(dataset="processed/a_very_long_dataset_name_indeed/", why="long")
    def long_one(f=iv.all_of(feed, why="the feed")):
        return frame()

    @iv.data(dataset="processed/b/", why="short")
    def short_one(f=iv.all_of(long_one, why="the long one")):
        return frame()

    at = {n["dataset"]: n["position"]["x"] for n in _web.payload(_graph.build(iv))["nodes"]}
    wide = at["processed/b/"] - at["processed/a_very_long_dataset_name_indeed/"]
    narrow = at["processed/a_very_long_dataset_name_indeed/"] - at["raw/x/"]
    assert wide > narrow, "the long name's own column pays for it, not the one before"


def test_the_page_needs_only_a_renderer(built, tmp_path):

    g, _, _, _ = load(built)
    text = _web.write(g, tmp_path / "dag.html").read_text()
    assert text.count("<script src=") == 1
    assert "dagre" not in text
    assert "{root:'diamond', terminal:'square'}" in text
    assert "up.difference(upDirect).addClass('up-far')" in text
    assert "down.difference(downDirect).addClass('down-far')" in text
    assert "partitioned by&nbsp;" in text
    assert "data-key=\"${esc(k)}\"" in text
    assert "n.data('partitionKeys').includes(selectedPartition)" in text
    assert "matches.addClass('partition-match')" in text
    assert text.index("immediate upstream") < text.index("shards —")


def test_the_panel_lists_the_reads_the_code_declares(iv):


    feed = iv.source("raw/feed/", why="a fetcher drops it here")

    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(f=iv.all_of(feed, why="the feed")):
        return frame()

    @iv.data(dataset="processed/end/", why="reads both")
    def end(f=iv.all_of(feed, why="the feed"), m=iv.all_of(mid, why="the middle")):
        return frame()

    p = _web.payload(_graph.build(iv))
    node, = [n for n in p["nodes"] if n["dataset"] == "processed/end/"]
    assert node["reads"] == ["processed/mid/", "raw/feed/"]
    src, = [n for n in p["nodes"] if n["dataset"] == "raw/feed/"]
    assert src["readBy"] == ["processed/end/", "processed/mid/"]


def test_the_layout_constants_match_the_css_they_are_meant_to_describe():


    import re
    css = _web._PAGE
    assert f"'font-size':{_web.FONT}," in css
    assert f"'width':{int(_web.MARKER)},'height':{int(_web.MARKER)}," in css
    assert _web.ROW > _web.MARKER, "a row has to be taller than the marker in it"
    assert _web.CHAR > _web.FONT * 0.5, "labels need most of the font size per character"
    assert re.search(r"const FLOOR = 0\.\d+;", css), "the zoom floor keeps the type legible"
