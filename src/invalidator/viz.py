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

from .static import slices_meet as _slices_meet  # noqa: E402


def to_networkx(g) -> nx.DiGraph:
    """The ARTIFACT graph: artifact -> artifact, with the stages collapsed onto edges.

    A SLICED artifact is more than one node. `game_predictions.parquet` holds the seasons
    that have been played and the one that has not, and drawn as a single node it is a
    ring: the played rows feed `preseason_team`, which feeds the upcoming rows. Split by
    slice it is a chain, which is what it actually is.
    """
    sliced = {}
    for path, sites in g.sites.items():
        labels = {s.slice for s in sites if s.slice}
        if labels:
            sliced[path] = labels

    def nodes_for(site) -> list[str]:
        """The node(s) a site touches. An UNLABELLED site on a sliced artifact means the
        whole of it, so it touches every slice."""
        labels = sliced.get(site.path)
        if not labels:
            return [site.path]
        return [f"{site.path}#{site.slice}"] if site.slice else [
            f"{site.path}#{l}" for l in sorted(labels)]

    d = nx.DiGraph()
    for path in g.artifacts:
        for n in ([f"{path}#{l}" for l in sorted(sliced[path])] if path in sliced
                  else [path]):
            d.add_node(n, kind="root" if not g.producers_of(path)
                       else "terminal" if g.is_terminal(path) else "derived")
    for node, stage in g.stages.items():
        updated = {s.path for s in stage.sites if s.kind == "update"}
        for out in stage.outputs():
            for inp in stage.inputs():
                if inp.path == out.path:
                    continue                # an updates() self-edge is not a dependency
                if inp.prior:
                    # The PREVIOUS run's copy. `fetch_athletes` reads the draft table and
                    # writes the bios; `build_draft_nba` reads the bios and writes the
                    # draft table — a loop on paper, and neither waits for the other.
                    continue
                if inp.path in updated and out.path in updated:
                    # Two paths the SAME stage rewrites. Either alternative branches of
                    # one declaration — the flat and the nested league spelling of a raw
                    # feed — or simply independent. Neither is built FROM the other: both
                    # already exist when the stage starts.
                    continue
                for a in nodes_for(inp):
                    for b in nodes_for(out):
                        d.add_edge(a, b, stage=node)
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
    return node.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def draw(g, out: Path, full: bool = False) -> Path:
    d = to_networkx(g)
    if not full:
        try:
            reduced = nx.transitive_reduction(d)
            reduced.add_nodes_from(d.nodes(data=True))
            d = reduced
        except nx.NetworkXError:
            pass                            # a cycle; refused just below

    # A TEMPORAL loop is legal and has to be drawn as one. `build_preseason` reads
    # `game_predictions` as `predict_games` left it and writes `preseason_team`;
    # `predict_upcoming_games` then reads that and amends `game_predictions`. Over a run
    # that is fine — the amendment comes after — but as a picture it is a ring. Break it
    # at the LATEST stage's edge, which is the amendment, and say so on the figure.
    broken = []
    while (cycle := find_cycle(d)):
        pairs = [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]
        order = g.order or {}
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
