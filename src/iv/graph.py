from __future__ import annotations

from dataclasses import dataclass

from . import static as _static
from .errors import IvError


@dataclass(frozen=True)
class Graph:
    iv: object
    stages: dict[str, _static.Stage]


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
        """Who must run before whom — per PARTITION, not per dataset.

        Two stages may write different partitions of one dataset, and a reader that names
        the partitions it wants depends only on whoever writes those. Comparing dataset
        names alone merges them, and a pair that writes past and future games while reading
        each other's half reads as a cycle it does not have.
        """
        out: dict[str, list[str]] = {n: [] for n in self.stages}
        for node, st in self.stages.items():
            for s in st.triggers:
                for p in self.producers_of(s.dataset):
                    if p != node and any(_overlaps(s, w) for w in self.stages[p].outputs
                                         if w.dataset == s.dataset):
                        out[node].append(p)
        return {k: sorted(set(v)) for k, v in out.items()}

    def order(self) -> list[str]:
        """Run order: dependencies first, ties broken by where the step is written.

        Two steps in the SAME file have a source order and it is meaningful, so the tiebreak
        preserves it and the ORDER check below holds it to being topological. Two steps in
        different files do not — file order is alphabetical, which means nothing — so there
        the toposort is the only answer, and there is nothing to check.
        """
        at = {n: i for i, n in enumerate(self.stages)}
        return [n for layer in toposort(self.parent_map()) for n in sorted(layer, key=at.get)]

    def export(self) -> dict:
        return {"schema": "iv/2", "nodes": sorted(self.stages),
                "parent_map": self.parent_map(),
                "datasets": {d: {"writers": self.producers_of(d),
                                 "readers": self.consumers_of(d)} for d in self.produced}}


def build(iv) -> Graph:
    """Every stage this pipeline has, however it was declared.

    Two ways in, and the graph is their UNION rather than whichever it finds first. A
    declared stage registers itself when its module imports; a stage written with
    `iv.reads`/`iv.writes` is found by scanning `source_dirs`. Preferring one would drop the
    other's nodes silently — and a dataset whose producer is missing from the graph looks
    like a root, which reads as `current` for anything with a file on disk. A green that
    means "I could not find the question" is worse than a red.

    The two cannot collide: a node is named `<file>::<function>` from the function's own
    code object either way, and a declared stage has no `iv.reads(...)` call for the scan
    to find in the first place.
    """
    nodes: dict[str, _static.Node] = {}
    for _, st in _static.scan(iv).items():
        for node in _static.nodes_of(st):
            nodes[node.name] = node
    for node in declared_nodes(iv):
        nodes[node.name] = node
    return Graph(iv=iv, stages=nodes)


def declared_nodes(iv) -> list[_static.Node]:
    """The registered assets, as the same `Node`s the scan produces.

    Everything downstream — `check`, `parent_map`, `iv status` — reads a Node and does not
    care which route it arrived by, so a declaration has to arrive in that shape rather
    than in one of its own.
    """
    out = []
    for asset in getattr(iv, "_assets", {}).values():
        fn_name = getattr(asset.fn, "__name__", "")
        rel = iv._rel_source(getattr(asset.fn, "__code__").co_filename)
        sites = [
            _static.Site(kind="read", dataset=r.dataset, why=r.why, file=rel,
                         line=_line_of(asset.fn), optional=r.optional,
                         update_file_on_disk=r.is_own,
                         where=r.where(), sel=r.sel(), owner=fn_name)
            for r in asset.reads
        ]
        # A partitioned asset writes a shard per key, and which keys is a runtime list —
        # so the write names no literal part=, exactly as a for_each does.
        sites.append(
            _static.Site(kind="write", dataset=asset.dataset, why=asset.why, file=rel,
                         line=_line_of(asset.fn), terminal=asset.terminal, owner=fn_name))
        # `guarded` is "has a skip check that could be fooled". A root asset has no
        # upstream to be stale against and runs every time, which is how anything outside
        # the tree gets in — so it is not guarded, and the RUNS ONCE warning below is not
        # about it.
        out.append(_static.Node(name=f"{rel}::{fn_name}", file=rel, fn=fn_name,
                                sites=tuple(sites),
                                guarded=asset.if_needed and asset.may_skip))
    return out


def _line_of(fn) -> int:
    return getattr(getattr(fn, "__code__", None), "co_firstlineno", 0)


def _overlaps(read, write) -> bool:
    """Could this read see anything this write produces?

    Only a value stated on BOTH sides can rule an edge out. Anything unstated means the
    read may well cover the write, so the edge stays: an edge too many is a redundant
    ordering constraint, an edge too few is a stage running before its input exists.
    """
    wanted = dict(read.where)
    for key, val in write.part:
        if key in wanted and val not in wanted[key]:
            return False
    return True


def toposort(parents: dict[str, list[str]]) -> list[list[str]]:
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


def check(g: Graph) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []
    roots = tuple(g.iv.roots)

    for name in sorted({s.dataset for st in g.stages.values() for s in st.inputs}):
        if g.producers_of(name) or name.startswith(roots):
            continue
        site = next(s for st in g.stages.values() for s in st.inputs if s.dataset == name)
        msg = (f"READ WITH NO PRODUCER  {name}\n"
               f"    read at {site.location} but nothing writes it. Either a stage is "
               f"missing, or it arrives out of band — put it under one of {list(roots)} "
               f"or add that prefix to the Pipeline's roots=.")
        (warns if site.optional else errors).append(msg)

    for name in g.produced:
        if g.consumers_of(name) or g.is_terminal(name):
            continue
        site = next(s for st in g.stages.values() for s in st.outputs if s.dataset == name)
        errors.append(
            f"WRITE WITH NO CONSUMER  {name}\n"
            f"    written at {site.location} and read by nothing. Delete the stage, or "
            f"pass terminal=True if it is consumed outside this pipeline.")

    for name in g.produced:
        writers = g.producers_of(name)
        if len(writers) < 2:
            continue
        sites = [s for st in g.stages.values() for s in st.outputs if s.dataset == name]
        parts = [s.part for s in sites]
        if all(parts) and len(set(parts)) == len(parts):
            continue
        why = ("they do not say which partition each writes, so whichever runs last wins"
               if not all(parts) else
               "they declare the same partition, so whichever runs last wins")
        errors.append(
            f"TWO WRITERS  {name}\n"
            f"    written by {writers} and {why}. Two stages may share a dataset only by "
            f"writing DIFFERENT partitions of it, each declared with a literal part=.")

    cyc = find_cycle(g)
    if cyc:
        errors.append(f"CYCLE  {cyc}")

    for node, st in g.stages.items():
        real = [s for s in st.outputs if s.kind != "constant"]
        if real and not st.triggers and st.guarded:
            warns.append(
                f"RUNS ONCE  {node} writes {sorted({s.dataset for s in real})} and "
                f"reads nothing that can trigger it, so it runs once and never again. "
                f"Right for a fetch-once archive. An update_file_on_disk= read does not "
                f"count: it is the copy this stage is about to overwrite, excluded from "
                f"the comparison by design. If this should re-run, read something that "
                f'moves — "config/today/".')

    for node, st in g.stages.items():
        # if_needed=False means the step runs no skip check, so there is nothing that could
        # be fooled by a write it cannot see. `for_each` decides per shard instead.
        if st.fn and st.guarded:
            seen_by_scan = {s.dataset for s in st.outputs}
            in_own_body = {s.dataset for s in st.outputs if s.owner == st.fn}
            hidden = seen_by_scan - in_own_body
            if hidden:
                errors.append(
                    f"WRITE OUTSIDE THE STEP  {node}\n"
                    f"    writes {sorted(hidden)} from a helper, not from its own body. "
                    f"The skip check reads the decorated function's source, so it cannot "
                    f"see those and will skip the stage while they are missing. Move the "
                    f"iv.writes(...) into the step body.")

    for node, st in g.stages.items():
        mine = {s.dataset for s in st.outputs}
        for site in st.inputs:
            if not site.update_file_on_disk or site.dataset in mine:
                continue
            errors.append(
                f"UPDATES SOMEONE ELSE  {node}\n"
                f"    reads {site.dataset} with update_file_on_disk=True at "
                f"{site.location}, but writes {sorted(mine) or 'nothing'}. That flag "
                f"excludes a dataset from the staleness comparison, which is only right "
                f"for the copy this stage is about to overwrite. On another stage's "
                f"dataset it hides a real dependency: run that producer first and read "
                f"it normally.")

    at = {n: i for i, n in enumerate(g.stages)}
    for node, ps in g.parent_map().items():
        for p in ps:
            if node.split("::")[0] == p.split("::")[0] and at[p] > at[node]:
                errors.append(
                    f"ORDER  {node} is defined before {p}, but reads something {p} writes. "
                    f"Within one file, definition order is run order.")

    return errors, warns


def drift(g: Graph, events: list[dict]) -> tuple[list[str], list[str]]:
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
        for op, name in sorted(pairs - declared):
            errors.append(f"UNDECLARED {op.upper()}  {node} -> {name}")
        for op, name in sorted(declared - pairs):
            warns.append(f"declared but not seen  {node} -> {name} ({op})")
    return errors, warns
