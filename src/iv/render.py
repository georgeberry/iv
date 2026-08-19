"""The DAG in the terminal: run order downward, dependencies as lanes.

There is no layout search, because there is nothing to search for. A pipeline HAS an
order — the script runs the stages in it — so rows are that order and the only question is
how to route edges between them. That also makes the picture honest: **a line going UP is
a stage reading something written later**, which is a bug you can see rather than one you
have to go and query.

Transitive reduction is on by default. Nearly every stage reads the same root feed, so the
full graph is mostly edges that constrain nothing; drawing them all hides the handful that
actually force the ordering. `--full` shows everything.

Reads any `{nodes, parent_map}` JSON, which is dbt's `target/manifest.json` shape too.
"""
from __future__ import annotations

import os
import shutil
import sys

V, TL, BL, H = "│", "╮", "╰", "─"


def _c(code: str, s: str, on: bool) -> str:
    return f"\033[{code}m{s}\033[0m" if on else s


def use_color(flag: bool | None = None) -> bool:
    if flag is not None:
        return flag
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


# ── shaping ───────────────────────────────────────────────────────────────────

def transitive_reduction(order: list[str], parents: dict[str, list[str]]
                         ) -> dict[str, list[str]]:
    """Drop every edge implied by a longer path. Reachability is preserved exactly."""
    rank = {n: i for i, n in enumerate(order)}
    ancestors: dict[str, set[str]] = {}
    out: dict[str, list[str]] = {}
    for n in sorted(parents, key=lambda x: rank.get(x, 10 ** 9)):
        direct = [p for p in parents.get(n, ()) if p in rank]
        keep = []
        for p in direct:
            if not any(p in ancestors.get(q, set()) for q in direct if q != p):
                keep.append(p)
        out[n] = sorted(keep)
        reach = set(direct)
        for p in direct:
            reach |= ancestors.get(p, set())
        ancestors[n] = reach
    return out


def ancestors_of(node: str, parents: dict[str, list[str]]) -> set[str]:
    seen, stack = set(), list(parents.get(node, ()))
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(parents.get(n, ()))
    return seen


def descendants_of(node: str, parents: dict[str, list[str]]) -> set[str]:
    children: dict[str, list[str]] = {}
    for n, ps in parents.items():
        for p in ps:
            children.setdefault(p, []).append(n)
    seen, stack = set(), list(children.get(node, ()))
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(children.get(n, ()))
    return seen


def focus(order: list[str], parents: dict[str, list[str]], needle: str
          ) -> tuple[list[str], dict[str, list[str]]]:
    """One node's whole cone: everything that feeds it, everything it breaks."""
    hits = [n for n in order if needle in n]
    if not hits:
        raise KeyError(needle)
    keep: set[str] = set()
    for h in hits:
        keep |= {h} | ancestors_of(h, parents) | descendants_of(h, parents)
    kept = [n for n in order if n in keep]
    return kept, {n: [p for p in parents.get(n, ()) if p in keep] for n in kept}


# ── drawing ───────────────────────────────────────────────────────────────────

def label(node: str) -> str:
    """A stage is a file; show it without its directory and extension."""
    return node.rsplit("/", 1)[-1].removesuffix(".py")


def render(order: list[str], parents: dict[str, list[str]],
           edge_artifacts: dict[str, list[str]] | None = None,
           color: bool | None = None, width: int | None = None) -> str:
    """Rows in run order, one lane per open edge."""
    on = use_color(color)
    rank = {n: i for i, n in enumerate(order)}
    term = width or shutil.get_terminal_size((100, 24)).columns

    # An edge occupies a lane from just after its parent's row through its child's row.
    edges = []
    for n in order:
        for p in parents.get(n, ()):
            if p in rank:
                edges.append((rank[p], rank[n], p, n))
    edges.sort(key=lambda e: (e[0], e[1]))

    # A lane is free at `start` only if its last edge ENDED BEFORE that row. Reusing it on
    # the same row puts a `╰` and a `╮` in one cell, and one silently overwrites the other.
    lanes: list[tuple[int, int, str, str]] = []
    placed: dict[tuple[str, str], int] = {}
    for e in edges:
        start = e[0]
        slot = next((i for i, occ in enumerate(lanes) if occ[1] < start), None)
        if slot is None:
            lanes.append(e)
            slot = len(lanes) - 1
        else:
            lanes[slot] = e
        placed[(e[2], e[3])] = slot

    nlanes = len(lanes)
    out: list[str] = []
    for i, node in enumerate(order):
        cells = [" "] * nlanes
        ends = []
        for (p, c), slot in placed.items():
            s, e = rank[p], rank[c]
            if s < i < e:
                cells[slot] = _c("90", V, on)
            elif s == i:
                cells[slot] = _c("90", TL, on)
            elif e == i:
                cells[slot] = _c("36", BL, on)
                ends.append(slot)

        gutter = "".join(cells)
        if ends:
            right = max(ends)
            plain = [c for c in cells]
            for k in range(right + 1, nlanes):
                if plain[k] == " ":
                    plain[k] = _c("36", H, on)
            gutter = "".join(plain)

        has_parents = bool([p for p in parents.get(node, ()) if p in rank])
        marker = _c("36", "●", on) if has_parents else _c("32", "○", on)
        line = f"{gutter} {marker} {_c('1', label(node), on)}"
        out.append(line[:term + 40])       # +40 for the invisible colour bytes

        if edge_artifacts:
            for p in sorted(parents.get(node, ())):
                for art in edge_artifacts.get(f"{p}->{node}", []):
                    out.append(" " * nlanes + f"   {_c('90', '· ' + art, on)}")
    return "\n".join(out)


def stage_card(node: str, g, color: bool | None = None) -> str:
    """One stage: what it reads, what it writes, and who is on the far end of each."""
    on = use_color(color)
    stage = g.stages.get(node)
    if stage is None:
        raise KeyError(node)
    lines = [_c("1", node, on)]
    for s in stage.externals():
        lines.append(f"  {_c('1', 'from', on)}  {s.path}")
        lines.append(f"      {_c('90', s.why, on)}")
    for kind, sites in (("reads", stage.inputs()),
                        ("writes", stage.outputs())):
        if not sites:
            continue
        lines.append(f"  {_c('1', kind, on)}")
        for s in sorted(sites, key=lambda x: x.path):
            flags = "".join(f" {f}" for f, ok in
                            (("?", s.optional), ("!", s.terminal), ("~", s.prior)) if ok)
            if kind == "reads":
                far = g.producers_of(s.path)
                far_s = ("<- " + ", ".join(label(f) for f in far)) if far \
                    else _c("90", "(a root — fetched outside this pipeline)", on)
            else:
                far = [c for c in g.consumers_of(s.path) if c != node]
                far_s = ("-> " + ", ".join(label(f) for f in far)) if far \
                    else _c("90", "(terminal — the app or a human)", on)
            lines.append(f"    {s.path}{flags}  {far_s}")
            lines.append(f"      {_c('90', s.why, on)}")
    return "\n".join(lines)
