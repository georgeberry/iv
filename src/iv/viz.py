from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt             # noqa: E402
import networkx as nx                       # noqa: E402

from .errors import IvError          # noqa: E402
from .graph import _overlaps        # noqa: E402


def to_networkx(g) -> nx.DiGraph:
    """The dataset graph, at the granularity the CHECKS reason about.

    A dataset several stages write, each naming a literal partition, is one node PER
    PARTITION. Collapsed into one, the played and unplayed halves of a prediction table
    become the same thing — and a pair of stages that reads one half and writes the other
    reads as a cycle it does not have. `parent_map` avoids that with `_overlaps`, so this
    uses the same test: an edge only where the read could actually see the write.

    Getting this wrong meant `iv check` said the graph was clean while the picture drew a
    cycle in it and announced cutting an edge to lay it out. Of the two, the picture was
    the one lying.
    """
    d = nx.DiGraph()
    writes: dict[str, list] = {}
    for node, stage in g.stages.items():
        for site in stage.outputs:
            writes.setdefault(site.dataset, []).append(site)

    def ident(site):
        peers = writes.get(site.dataset, ())
        return (site.dataset, site.part if len(peers) > 1 and site.part else ())

    for ds in g.datasets:
        peers = writes.get(ds, [])
        if not peers:
            d.add_node((ds, ()), kind="root")
            continue
        kind = "terminal" if g.is_terminal(ds) else "derived"
        for site in peers:
            d.add_node(ident(site), kind=kind)

    for node, stage in g.stages.items():
        for out in stage.outputs:
            for inp in stage.triggers:
                if inp.dataset == out.dataset:
                    continue
                producers = writes.get(inp.dataset)
                if not producers:
                    d.add_edge((inp.dataset, ()), ident(out), stage=node)
                    continue
                for src in producers:
                    if _overlaps(inp, src):
                        d.add_edge(ident(src), ident(out), stage=node)
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


def short(node) -> str:
    """The dataset's last segment, and the shard it is if the dataset has more than one
    stage writing it."""
    ds, part = node if isinstance(node, tuple) else (node, ())
    name = ds.rstrip("/").rsplit("/", 1)[-1]
    return f"{name} [{','.join(f'{k}={v}' for k, v in part)}]" if part else name


#: COLOUR is status, and these are the colours `iv status` prints — green current, cyan
#: maybe, yellow stale — so the picture and the report say the same thing in the same way.
#: A dataset with no producer has no stage to ask, so it is not coloured, it is grey.
STATUS = {
    "current": "#2a9d8f",
    "maybe": "#3aa8c1",
    "stale": "#e0a458",
    "source": "#9aa0a6",
}

#: SHAPE is what KIND of thing it is, which is a different question from whether it is
#: current — so it gets a different channel rather than fighting for the same one.
SHAPE = {
    "root": "s",          # arrives from outside; nothing here produces it
    "terminal": "D",      # read by something outside this pipeline
    "derived": "o",       # built here and read here
}


def states(reasons: dict, maybe: set) -> dict:
    """`iv status`'s answer per dataset, as the names `STATUS` colours.

    Here rather than in the CLI so the picture and the report cannot drift into naming the
    same three things differently.
    """
    return {name: ("stale" if why else "maybe" if name in maybe else "current")
            for name, why in reasons.items()}


def draw(g, out: Path, full: bool = False, status: dict | None = None) -> Path:
    """The DAG, coloured by status and shaped by kind.

    Two channels, because they are two questions. `iv status` answers "is this current",
    and that is the colour, in the colours it prints. What KIND of dataset it is — arrives
    from outside, built here, read by something else — does not change day to day, and it
    is the shape.
    """
    d = to_networkx(g)
    if not full:
        if find_cycle(d) is None:
            reduced = nx.transitive_reduction(d)
            reduced.add_nodes_from(d.nodes(data=True))
            d = reduced

    # Which edge to cut is decided by run order — and `order()` is a toposort, so it
    # raises on exactly the graph this loop exists to handle. A cycle is when the picture
    # is most wanted, so fall back to declaration order rather than crashing on it.
    try:
        order = {n: i for i, n in enumerate(g.order())}
    except IvError:
        order = {n: i for i, n in enumerate(g.stages)}

    broken = []
    while (cycle := find_cycle(d)):
        pairs = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
        a, b = max(pairs, key=lambda e: order.get(d.edges[e].get("stage"), 0))
        broken.append((a, b, d.edges[a, b].get("stage")))
        d.remove_edge(a, b)

    depth = _layers(d)
    by_layer: dict[int, list[str]] = {}
    for n, k in depth.items():
        by_layer.setdefault(k, []).append(n)

    status = status or {}

    # A COLUMN IS AS WIDE AS ITS OWN LONGEST LABEL. Sizing every column for the longest
    # label in the whole graph is most of a page of white space, because one dataset called
    # possessions_with_lineups should not set the gap between two called xpm and wvorp.
    # One data unit is about one inch, so the figure can be sized from the same numbers.
    CHAR, GAP = 0.092, 0.78
    xs, x = {}, 0.0
    for k in sorted(by_layer):
        xs[k] = x
        x += CHAR * max(len(short(n)) for n in by_layer[k]) + GAP
    total = x

    pos = {}
    for k, nodes in by_layer.items():
        for i, n in enumerate(sorted(nodes)):
            pos[n] = (xs[k], (-i + (len(nodes) - 1) / 2) * 0.34)

    tallest = max(len(v) for v in by_layer.values())
    fig, ax = plt.subplots(figsize=(total + 0.8, max(4.0, 0.34 * tallest + 1.6)))
    nx.draw_networkx_edges(d, pos, ax=ax, edge_color="#c9c9c9", arrows=True,
                           arrowsize=10, node_size=260, width=1.1)
    for kind, marker in SHAPE.items():
        group = [n for n in d if d.nodes[n].get("kind", "derived") == kind]
        if not group:
            continue
        nx.draw_networkx_nodes(
            d, pos, ax=ax, nodelist=group, node_shape=marker, node_size=260,
            edgecolors="#ffffff", linewidths=0.8,
            # A node is (dataset, shard) now, and `iv status` answers per dataset.
            node_color=[STATUS.get(status.get(n[0], "source"), STATUS["source"])
                        for n in group])
    for n, (x, y) in pos.items():
        # Offset in POINTS, not data units: the node is drawn at a fixed size in points, so
        # a gap measured in data units closes up whenever the graph is small enough for the
        # figure to be stretched to fit the legend.
        ax.annotate(short(n), (x, y), textcoords="offset points", xytext=(12, 0),
                    fontsize=11, va="center", ha="left", color="#222")

    from matplotlib.lines import Line2D
    key = [Line2D([], [], color=c, marker="o", linestyle="", markersize=8, label=name)
           for name, c in STATUS.items()]
    key += [Line2D([], [], color="#666", marker=m, linestyle="", markersize=8, label=k)
            for k, m in SHAPE.items()]
    # Capped, so a small graph is not stretched sideways to fit one row of legend — and
    # anchored ABOVE the axes rather than inside them, where it lands on the first column.
    ncol = min(len(key), 4)
    ax.legend(handles=key, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol,
              frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.1)
    rows = -(-len(key) // ncol)

    title = "left to right is dependency order — an edge pointing LEFT is a bug"
    if broken:
        title += "\n" + "\n".join(
            f"cut to lay out (amended later by {st}): {short(a)} -> {short(b)}"
            for a, b, st in broken)
    ax.set_title(title, fontsize=9, color="#555", pad=14 + 15 * rows)
    ax.axis("off")
    ax.set_xlim(-0.35, total + 0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
