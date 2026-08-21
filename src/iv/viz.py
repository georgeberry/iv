from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
import networkx as nx                       # noqa: E402


def to_networkx(g) -> nx.DiGraph:
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
