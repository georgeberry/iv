"""The declared graph, and every structural check over it.

Stages are the nodes; a dataset written by one and read by another is the edge. Built from
the static scan alone, so all of this works on a fresh checkout with no data and nothing
ever run.

WHAT THE CHECKS ARE FOR. Each one is a way a pipeline can be wrong that no single stage can
notice about itself, which is exactly the class of bug that survives review: a read nothing
produces, a write nothing consumes, two stages that both think they own a dataset, an
ordering that runs a consumer before its producer, and a cycle.

ONE WRITER PER DATASET, with no exception. A dataset is a directory, so a second producer
that has something legitimate to add writes its own SHARD — a different partition. Two
stages writing the same partition is unambiguously a bug, and an exemption for "the second
one only amends the first" is how a stage that fully overwrites someone else's output hides
indefinitely.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import static as _static
from .errors import IvError


@dataclass(frozen=True)
class Graph:
    pipe: object
    stages: dict[str, _static.Stage]

    # ── the shape ─────────────────────────────────────────────────────────────

    @property
    def datasets(self) -> list[str]:
        seen = {s.dataset for st in self.stages.values() for s in st.sites
                if s.kind != "external"}
        return sorted(seen)

    @property
    def produced(self) -> list[str]:
        return sorted({s.dataset for st in self.stages.values() for s in st.outputs})

    def producers_of(self, dataset: str) -> list[str]:
        return sorted(n for n, st in self.stages.items()
                      if any(s.dataset == dataset for s in st.outputs))

    def consumers_of(self, dataset: str) -> list[str]:
        return sorted(n for n, st in self.stages.items()
                      if any(s.dataset == dataset for s in st.inputs))

    def is_terminal(self, dataset: str) -> bool:
        return any(s.dataset == dataset and s.terminal
                   for st in self.stages.values() for s in st.outputs)

    def parent_map(self) -> dict[str, list[str]]:
        """stage -> the stages that must run before it."""
        out: dict[str, list[str]] = {n: [] for n in self.stages}
        for node, st in self.stages.items():
            for s in st.inputs:
                out[node].extend(p for p in self.producers_of(s.dataset) if p != node)
        return {k: sorted(set(v)) for k, v in out.items()}

    def order(self) -> list[str]:
        """A run order: declared if the pipeline names one, else a topological sort."""
        declared = declared_order(self.pipe)
        if declared:
            return [n for n in declared if n in self.stages] + \
                   sorted(n for n in self.stages if n not in declared)
        return [n for layer in toposort(self.parent_map()) for n in layer]

    def export(self) -> dict:
        return {"schema": "iv/2", "nodes": sorted(self.stages),
                "parent_map": self.parent_map(),
                "datasets": {d: {"writers": self.producers_of(d),
                                 "readers": self.consumers_of(d)} for d in self.produced}}


def build(pipe) -> Graph:
    return Graph(pipe=pipe, stages=_static.scan(pipe))


def declared_order(pipe) -> list[str]:
    """Run order scraped from the shell script the pipeline names, if it names one."""
    import re
    if not pipe.order_from:
        return []
    p = pipe.order_from
    from pathlib import Path
    path = Path(p) if Path(p).is_absolute() else Path(pipe.project_root or ".") / p
    if not path.exists():
        return []
    out, seen = [], set()
    pat = re.compile(r"(?:python3?|uv run(?:\s+\S+)*?)\s+(\S+\.py)")
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue                       # a commented-out stage is not in the order
        m = pat.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            out.append(m.group(1))
    return out


def toposort(parents: dict[str, list[str]]) -> list[list[str]]:
    """Layers, each depending only on earlier ones. Raises on a cycle."""
    remaining = {k: set(v) & set(parents) for k, v in parents.items()}
    out = []
    while remaining:
        layer = sorted(k for k, v in remaining.items() if not v)
        if not layer:
            raise IvError(f"cycle among {sorted(remaining)}")
        out.append(layer)
        remaining = {k: v - set(layer) for k, v in remaining.items() if k not in layer}
    return out


def find_cycle(g: Graph) -> str | None:
    try:
        toposort(g.parent_map())
    except IvError as e:
        return str(e)
    return None


# ── the checks ────────────────────────────────────────────────────────────────

def check(g: Graph) -> tuple[list[str], list[str]]:
    """Every structural check. Returns (errors, warnings)."""
    errors: list[str] = []
    warns: list[str] = []
    roots = tuple(g.pipe.roots)

    for ds in sorted({s.dataset for st in g.stages.values() for s in st.inputs}):
        if g.producers_of(ds) or ds.startswith(roots):
            continue
        site = next(s for st in g.stages.values() for s in st.inputs if s.dataset == ds)
        msg = (f"READ WITH NO PRODUCER  {ds}\n"
               f"    read at {site.location} but nothing writes it. Either a stage is "
               f"missing, or it arrives out of band — put it under one of {list(roots)} "
               f"or add that prefix to the Pipeline's roots=.")
        (warns if site.optional else errors).append(msg)

    for ds in g.produced:
        if g.consumers_of(ds) or g.is_terminal(ds):
            continue
        site = next(s for st in g.stages.values() for s in st.outputs if s.dataset == ds)
        errors.append(
            f"WRITE WITH NO CONSUMER  {ds}\n"
            f"    written at {site.location} and read by nothing. Delete the stage, or "
            f"pass terminal=True if it is consumed outside this pipeline.")

    for ds in g.produced:
        writers = g.producers_of(ds)
        if len(writers) > 1:
            errors.append(
                f"TWO WRITERS  {ds}\n"
                f"    written by {writers}. One writer per dataset — a second producer "
                f"should write its own PARTITION of it, or its own dataset.")

    cyc = find_cycle(g)
    if cyc:
        errors.append(f"CYCLE  {cyc}")

    # RUNS ONCE, NEVER AGAIN. A stage whose output has no id-bearing input has nothing that
    # can move, so it is current forever the moment it is stamped — indistinguishable, from
    # the outside, from the cache working perfectly. Correct for fetch-once history; a bug
    # for anything polled, and the fix is to read the clock.
    for node, st in g.stages.items():
        real = [s for s in st.outputs if s.kind != "constant"]
        if real and not st.inputs:
            warns.append(
                f"RUNS ONCE  {node} writes {sorted({s.dataset for s in real})} and "
                f"reads no dataset, so nothing can ever make it stale. Right for a "
                f"fetch-once archive. If it should re-run, give it something that moves — "
                f'read a constants file such as "config/today/".')

    declared = declared_order(g.pipe)
    if declared:
        pos = {n: i for i, n in enumerate(declared)}
        for node, ps in g.parent_map().items():
            for p in ps:
                if node in pos and p in pos and pos[p] > pos[node]:
                    errors.append(
                        f"ORDER  {node} runs before {p}, but reads something {p} writes.")
    elif g.pipe.order_from:
        warns.append(
            f"no stage invocations found in {g.pipe.order_from}, so run ORDER was not "
            f"checked. A stage launched via a shell function or a variable is invisible "
            f"to the scrape.")

    return errors, warns


def drift(g: Graph, events: list[dict]) -> tuple[list[str], list[str]]:
    """What a run actually touched, against what the code declares.

    `recorded - declared` is an ERROR: the process really did open that dataset, so the
    scan is wrong about the graph. `declared - recorded` is a WARN: an absent optional
    input or a branch not taken produces it legitimately.
    """
    errors, warns = [], []
    seen: dict[str, set[tuple[str, str]]] = {}
    for e in events:
        if e.get("kind") != "io" or e.get("op") not in ("read", "write"):
            continue
        seen.setdefault(e.get("node", "?"), set()).add((e["op"], e.get("rel", "")))

    for node, pairs in sorted(seen.items()):
        st = g.stages.get(node)
        declared = {("read", s.dataset) for s in (st.inputs if st else ())} | \
                   {("write", s.dataset) for s in (st.outputs if st else ())}
        for op, ds in sorted(pairs - declared):
            errors.append(f"UNDECLARED {op.upper()}  {node} -> {ds}")
        for op, ds in sorted(declared - pairs):
            warns.append(f"declared but not seen  {node} -> {ds} ({op})")
    return errors, warns
