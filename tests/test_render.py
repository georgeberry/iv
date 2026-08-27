from __future__ import annotations

from types import SimpleNamespace

import pytest

from tyke import render
from tyke.static import Node, Site


def test_transitive_reduction_removes_redundant_edges():
    parents = {"a": [], "b": ["a"], "c": ["a", "b"], "outside": ["a"]}
    assert render.transitive_reduction(["a", "b", "c"], parents) == {
        "a": [],
        "b": ["a"],
        "c": ["b"],
        "outside": ["a"],
    }


def test_ancestor_descendant_and_focus_queries():
    parents = {"a": [], "b": ["a"], "c": ["b"], "side": ["a"]}
    assert render.ancestors_of("c", parents) == {"a", "b"}
    assert render.descendants_of("a", parents) == {"b", "c", "side"}
    assert render.focus(["a", "b", "c", "side"], parents, "b") == (
        ["a", "b", "c"],
        {"a": [], "b": ["a"], "c": ["b"]},
    )
    with pytest.raises(KeyError):
        render.focus(["a"], parents, "missing")


def test_render_draws_roots_edges_and_artifacts_without_color():
    order = ["stages/a.py", "stages/b.py", "stages/c.py"]
    parents = {order[0]: [], order[1]: [order[0]], order[2]: [order[0], order[1]]}
    text = render.render(
        order,
        parents,
        edge_artifacts={f"{order[0]}->{order[1]}": ["raw/feed/"]},
        color=False,
        width=20,
    )
    assert "○ a" in text
    assert "● b" in text
    assert "● c" in text
    assert "· raw/feed/" in text
    assert "\033[" not in text


def test_use_color_honors_flags_terminal_and_environment(monkeypatch):
    assert render.use_color(True)
    assert not render.use_color(False)
    monkeypatch.setattr(render.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert render.use_color()
    monkeypatch.setenv("NO_COLOR", "1")
    assert not render.use_color()


def site(kind, dataset, why, **fields):
    return Site(kind, dataset, why, "stages/model.py", 3, **fields)


def test_stage_card_describes_external_reads_writes_and_destinations():
    node = "stages/model.py::train"
    stage = Node(
        node,
        "stages/model.py",
        "train",
        (
            site("external", "external:warehouse", "source system"),
            site("read", "raw/feed/", "training rows", optional=True),
            site("write", "processed/model/", "fitted model", update_file_on_disk=True),
        ),
    )
    graph = SimpleNamespace(
        stages={node: stage},
        producers_of=lambda dataset: ["stages/fetch.py::fetch"],
        consumers_of=lambda dataset: [node, "app.py::serve"],
    )

    card = render.stage_card(node, graph, color=False)

    assert "from  external:warehouse" in card
    assert "raw/feed/ ?  <- fetch.py::fetch" in card
    assert "processed/model/ ~  -> app.py::serve" in card
    with pytest.raises(KeyError):
        render.stage_card("missing", graph, color=False)


def test_stage_card_marks_source_and_terminal_datasets():
    node = "stage.py::only"
    stage = Node(
        node,
        "stage.py",
        "only",
        (
            site("read", "raw/feed/", "input"),
            site("write", "dump/result/", "output"),
        ),
    )
    graph = SimpleNamespace(
        stages={node: stage},
        producers_of=lambda dataset: [],
        consumers_of=lambda dataset: [node],
    )
    card = render.stage_card(node, graph, color=False)
    assert "a source" in card
    assert "terminal" in card
