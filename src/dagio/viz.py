"""Draw the artifact graph to a PNG. Needs the `[viz]` extra.

matplotlib rather than graphviz because it needs no system binary — `dot` is not installed
on most machines and a picture that depends on one is a picture nobody looks at.

Left to right is dependency order, so **an edge pointing LEFT is a bug** — the same
invariant the terminal renderer states vertically. Layers come from longest-path depth
rather than `multipartite_layout`, which keeps a chain on one line instead of fanning it.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display; this runs in CI and over ssh
import matplotlib.pyplot as plt             # noqa: E402
import networkx as nx                       # noqa: E402


def to_networkx(g) -> nx.DiGraph:
    """The ARTIFACT graph: artifact -> artifact, with the stages collapsed onto edges."""
    d = nx.DiGraph()
    for path in g.artifacts:
        d.add_node(path, kind="root" if not g.producers_of(path)
                   else "terminal" if g.is_terminal(path) else "derived")
    for node, stage in g.stages.items():
        for out in stage.outputs(g.scope):
            for inp in stage.inputs(g.scope):
                if inp.path != out.path:    # an updates() self-edge is not a dependency
                    d.add_edge(inp.path, out.path, stage=node)
    return d


def _layers(d: nx.DiGraph) -> dict[str, int]:
    depth: dict[str, int] = {}
    for n in nx.topological_sort(d):
        preds = list(d.predecessors(n))
        depth[n] = 1 + max((depth[p] for p in preds), default=-1)
    return depth


def short(node: str) -> str:
    return node.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def draw(g, out: Path, full: bool = False) -> Path:
    d = to_networkx(g)
    if not full:
        try:
            reduced = nx.transitive_reduction(d)
            reduced.add_nodes_from(d.nodes(data=True))
            d = reduced
        except nx.NetworkXError:
            pass                            # a cycle; `dagio check` reports it properly

    depth = _layers(d)
    by_layer: dict[int, list[str]] = {}
    for n, k in depth.items():
        by_layer.setdefault(k, []).append(n)

    pos = {}
    for k, nodes in by_layer.items():
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (k, -i + (len(nodes) - 1) / 2)

    colors = {"root": "#2a9d8f", "terminal": "#e76f51", "derived": "#4a6fa5"}
    node_colors = [colors[d.nodes[n].get("kind", "derived")] for n in d]

    width = max(8, 2.4 * (max(by_layer) + 1))
    height = max(4, 1.1 * max(len(v) for v in by_layer.values()))
    fig, ax = plt.subplots(figsize=(width, height))
    nx.draw_networkx_edges(d, pos, ax=ax, edge_color="#b8b8b8", arrows=True,
                           arrowsize=11, node_size=140)
    nx.draw_networkx_nodes(d, pos, ax=ax, node_color=node_colors, node_size=140)
    for n, (x, y) in pos.items():
        ax.text(x + 0.055, y, short(n), fontsize=8, va="center", ha="left")

    ax.set_title("left to right is dependency order — an edge pointing LEFT is a bug",
                 fontsize=9, color="#555")
    ax.axis("off")
    ax.set_xlim(-0.4, max(by_layer) + 1.1)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
