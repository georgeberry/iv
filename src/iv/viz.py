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
    """The DATASET graph: dataset -> dataset, with the stages collapsed onto edges.

    Shorter than it used to be because there is less to say. There are no slices to split a
    node on and no self-edges to skip: a dataset is written by exactly one stage, and a
    stage that amends its own output declares that read as `prior=` — the previous run's
    copy, which is lineage and not a dependency.
    """
    d = nx.DiGraph()
    for ds in g.datasets:
        d.add_node(ds, kind="root" if not g.producers_of(ds)
                   else "terminal" if g.is_terminal(ds) else "derived")
    for node, stage in g.stages.items():
        for out in stage.outputs:
            for inp in stage.inputs:
                if inp.dataset == out.dataset or inp.prior:
                    continue
                d.add_edge(inp.dataset, out.dataset, stage=node)
    return d


def find_cycle(d: nx.DiGraph) -> list | None:
    """The artifact cycle, as a list of paths, or None."""
    try:
        c = nx.find_cycle(d)
    except nx.NetworkXNoCycle:
        return None
    return [e[0] for e in c] + [c[-1][1]]


def _layers(d: nx.DiGraph) -> dict[str, int]:
    depth: dict[str, int] = {}
    for n in nx.topological_sort(d):
        preds = list(d.predecessors(n))
        depth[n] = 1 + max((depth[p] for p in preds), default=-1)
    return depth


def short(node: str) -> str:
    return node.rstrip("/").rsplit("/", 1)[-1]


def draw(g, out: Path, full: bool = False) -> Path:
    d = to_networkx(g)
    if not full:
        if find_cycle(d) is None:
            reduced = nx.transitive_reduction(d)
            reduced.add_nodes_from(d.nodes(data=True))
            d = reduced

    # A TEMPORAL loop is legal and has to be drawn as one. `build_preseason` reads
    # `game_predictions` as `predict_games` left it and writes `preseason_team`;
    # `predict_upcoming_games` then reads that and amends `game_predictions`. Over a run
    # that is fine — the amendment comes after — but as a picture it is a ring. Break it
    # at the LATEST stage's edge, which is the amendment, and say so on the figure.
    broken = []
    while (cycle := find_cycle(d)):
        pairs = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
        order = {n: i for i, n in enumerate(g.order())}
        a, b = max(pairs, key=lambda e: order.get(d.edges[e].get("stage"), 0))
        broken.append((a, b, d.edges[a, b].get("stage")))
        d.remove_edge(a, b)

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

    title = "left to right is dependency order — an edge pointing LEFT is a bug"
    if broken:
        # Named, not hidden. A ring the run order resolves is still a ring, and whoever
        # reads this picture should know which edge was cut to draw it.
        title += "\n" + "\n".join(
            f"cut to lay out (amended later by {st}): {short(a)} -> {short(b)}"
            for a, b, st in broken)
    ax.set_title(title, fontsize=9, color="#555")
    ax.axis("off")
    ax.set_xlim(-0.4, max(by_layer) + 1.1)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
