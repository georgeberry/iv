from __future__ import annotations

from dataclasses import dataclass

from . import static as _static
from .errors import IvError


@dataclass(frozen=True)
class Graph:
    iv: object
    stages: dict[str, _static.Node]


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


        return not self.consumers_of(dataset)

    def parent_map(self) -> dict[str, list[str]]:


        out: dict[str, list[str]] = {n: [] for n in self.stages}
        for node, st in self.stages.items():
            for s in st.triggers:
                for p in self.producers_of(s.dataset):
                    if p != node and any(_overlaps(s, w) for w in self.stages[p].outputs
                                         if w.dataset == s.dataset):
                        out[node].append(p)
        return {k: sorted(set(v)) for k, v in out.items()}

    def order(self) -> list[str]:


        at = {n: i for i, n in enumerate(self.stages)}
        return [n for layer in toposort(self.parent_map()) for n in sorted(layer, key=at.get)]


def build(iv) -> Graph:


    nodes = sorted(declared_nodes(iv), key=lambda n: (n.file, _first_line(n), n.fn))
    return Graph(iv=iv, stages={n.name: n for n in nodes})


def _first_line(node) -> int:
    return min((s.line for s in node.sites), default=0)


def declared_nodes(iv) -> list[_static.Node]:


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


        for name, why in asset.externals:
            sites.append(
                _static.Site(kind="external", dataset=_static.EXTERNAL_PREFIX + name,
                             why=why, file=rel, line=_line_of(asset.fn), owner=fn_name))
        fixed = tuple(sorted(asset.fixed_part.items())) if asset.fixed_part else ()
        for o in asset.outputs.values():
            sites.append(
                _static.Site(kind="write", dataset=o.dataset, why=asset.why, file=rel,
                             line=_line_of(asset.fn), part=(o.part or fixed),
                             owner=fn_name))


        out.append(_static.Node(name=f"{rel}::{fn_name}", file=rel, fn=fn_name,
                                sites=tuple(sites),
                                guarded=asset.may_skip))
    return out


def _line_of(fn) -> int:
    return getattr(getattr(fn, "__code__", None), "co_firstlineno", 0)


def _overlaps(read, write) -> bool:


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

    for name, src in sorted(getattr(g.iv, "_sources", {}).items()):
        if not g.consumers_of(name):
            warns.append(
                f"SOURCE NOBODY READS  {name}\n"
                f"    declared as arriving from outside — {src.why} — and read by no "
                f"stage. Delete the declaration, or wire it up.")

    produced = set(g.produced)
    for name, d in sorted(getattr(g.iv, "_datasets", {}).items()):
        if name in produced:
            continue
        readers = g.consumers_of(name)
        required = [n for n in readers
                    if any(site.dataset == name and not site.optional
                           for site in g.stages[n].inputs)]
        if required:


            errors.append(
                f"READ, NOBODY WRITES  {name}\n"
                f"    declared with iv.dataset(...) — {d.why} — and read by "
                f"{', '.join(required)}, but named in no stage's output=. Nothing puts it "
                f"there, so the read cannot succeed. Write it with @iv.data or @iv.step, "
                f"or declare it a source if it arrives from outside.")
        elif readers:
            warns.append(
                f"OPTIONAL, NOBODY WRITES  {name}\n"
                f"    declared with iv.dataset(...) — {d.why} — and named in no stage's "
                f"output=, but every reader ({', '.join(readers)}) declares optional=True, "
                f"which is what a dataset one configuration produces and another does not "
                f"looks like. If this configuration should produce it, wire it up.")
        else:
            warns.append(
                f"DECLARED, NOBODY WRITES  {name}\n"
                f"    declared with iv.dataset(...) — {d.why} — and named in no stage's "
                f"output=, so nothing puts it there. Nothing reads it either, so this is a "
                f"name a rename left behind. Wire it up, or delete it.")

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
