"""The graph the call sites describe, and every question worth asking of it.

Nodes are of two kinds and both matter. A STAGE is a file. An ARTIFACT is a path. An edge
runs producer -> artifact -> consumer, so the artifact is on the edge rather than hidden
inside it, and `iv graph --artifacts` can name it.

The checks are the point. Each one is a mistake that is invisible while you are making it
and expensive once made:

  READ WITH NO PRODUCER    something is read that nothing writes and no one fetches
  WRITE WITH NO CONSUMER   a stage still runs for an artifact nobody reads any more
  ORDER                    a stage reads an artifact written later in the run
  TWO WRITERS              two stages overwrite one artifact; the winner is whoever ran last
  POLICY CONFLICT          two writers of one artifact disagree about what it IS
  CYCLE                    A feeds B feeds A
  UNREACHABLE              nothing upstream is a root, so new data can never make it stale

The last one is the guard against this package's own silent-failure mode. An artifact
whose chain reaches no root is permanently, quietly current, and nothing else would ever
say so.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from dataclasses import replace as _replace

from . import static as _static
from .errors import InvalidatorError
from .static import Site, Stage, matches
from .static import slices_meet as _slices_meet


def _same(a: str, b: str) -> bool:
    """Do two declared paths name the same artifact? Templates match their instances."""
    return a == b or matches(a, b) or matches(b, a)


@dataclass
class Graph:
    iv: object = None
    stages: dict[str, Stage] = field(default_factory=dict)
    order: dict[str, int] = field(default_factory=dict)
    # Files that declare I/O and no entry point reaches. In a two-pipeline repo these are
    # usually the other pipeline's stages, so they are reported and not graphed.
    unreached: list[str] = field(default_factory=list)
    writers: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    updaters: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    readers: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    sites: dict[str, list[Site]] = field(default_factory=lambda: defaultdict(list))

    @property
    def artifacts(self) -> list[str]:
        return sorted(set(self.writers) | set(self.updaters) | set(self.readers))

    @property
    def produced(self) -> list[str]:
        return sorted(set(self.writers) | set(self.updaters))

    def is_terminal(self, path: str) -> bool:
        return any(s.terminal for s in self.sites[path])

    def producers_of(self, path: str, slice: str = "") -> list[str]:
        """The stages that write `path` — narrowed to those whose ROWS this could see.

        `slice` is how an artifact says it is not one population. A writer of a disjoint
        slice is not a producer of what you are reading, however much it writes the file.
        """
        out = {n for p, ns in self.writers.items() if _same(p, path) for n in ns}
        out |= {n for p, ns in self.updaters.items() if _same(p, path) for n in ns}
        if slice:
            out = {n for n in out
                   if any(_same(w.path, path) and _slices_meet(w.slice, slice)
                          for w in self.stages[n].sites
                          if w.kind in ("write", "update"))}
        return sorted(out)

    def consumers_of(self, path: str) -> list[str]:
        return sorted({n for p, ns in self.readers.items() if _same(p, path) for n in ns})

    def stage_parents(self) -> dict[str, list[str]]:
        """stage -> the stages that produce something it reads."""
        parents: dict[str, list[str]] = {n: [] for n in self.stages}
        for node, stage in self.stages.items():
            seen = set()
            for s in stage.inputs():
                if s.prior:
                    continue            # deliberately the previous run's copy
                for p in self.producers_of(s.path):
                    if p != node:
                        seen.add(p)
            parents[node] = sorted(seen)
        return parents

    def parent_map(self) -> dict[str, list[str]]:
        """The bipartite map: artifacts are nodes too. dbt's `parent_map` shape."""
        parents: dict[str, list[str]] = {}
        for node, stage in self.stages.items():
            parents[node] = sorted({s.path for s in stage.inputs()
                                    if not s.prior})
        for path in self.artifacts:
            parents.setdefault(path, [])
            parents[path] = sorted(set(parents[path]) | set(self.producers_of(path)))
        return parents


def build(iv, stages: dict[str, Stage] | None = None,
          order: list[str] | None = None) -> Graph:
    stages = _static.scan(iv) if stages is None else stages
    if order is None:
        order = iv.declared_order()

    # A NODE IS AN ENTRY POINT — a file the pipeline actually invokes. A library is not
    # one, however much it declares: `wvorp.data` reads the panels on behalf of every
    # stage and writes the raw feeds on behalf of every fetcher, so as a node it both
    # precedes and follows everything, and the graph is one cycle through it.
    #
    # When the project says what it runs, that list IS the node set and each entry point
    # owns its libraries' declarations. Without a declared order there is nothing better
    # to go on than "it produces something", which is the old rule.
    if order:
        entries = [n for n in order if n in stages]
        sites = {n: _static.sites_of_entry(iv, n) for n in entries}
        # A file that declares I/O and is reached by NO entry point is NOT a node. It may
        # be a stage nothing runs, which is worth saying — but in a repo running two
        # pipelines out of one tree it is usually just the other pipeline's, and folding
        # those in reports every shared artifact name as two writers in conflict. So they
        # are reported (`g.unreached`) rather than graphed.
        reached = {n for e in entries for n in _static.entry_owners(iv, e)}
        unreached = sorted(n for n, st in stages.items()
                           if n not in reached and st.outputs())
        stages = {n: _replace(stages[n], sites=sites[n]) for n in entries}
    else:
        unreached = []
        stages = {n: st for n, st in stages.items() if st.outputs()}

    g = Graph(iv=iv, stages=dict(stages))
    g.unreached = unreached
    ranks = {n: i for i, n in enumerate(order)} if order else {}
    # A stage the order does not mention sorts after everything it does, rather than
    # silently at the front where it would look like a legal producer for everyone.
    g.order = {n: ranks.get(n, 10 ** 6 + i) for i, n in enumerate(sorted(stages))}

    for node, stage in stages.items():
        for s in stage.sites:
            g.sites[s.path].append(s)
            if s.kind == "external":
                continue                # provenance, not a dependency — carries no id
            if s.kind in ("read", "collection"):
                g.readers[s.path].append(node)
            elif s.kind == "write":
                g.writers[s.path].append(node)
            else:                       # update: reads AND writes, so it is both
                g.updaters[s.path].append(node)
                g.readers[s.path].append(node)
    return g


# ── toposort, and the cycle detection that falls out of it ────────────────────

def toposort(parents: dict[str, list[str]]) -> list[list[str]]:
    """Kahn's algorithm, grouped by depth. Raises on a cycle, naming what is in it."""
    nodes = set(parents) | {p for ps in parents.values() for p in ps}
    indeg = {n: len(set(parents.get(n, ()))) for n in nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for n, ps in parents.items():
        for p in set(ps):
            children[p].append(n)

    gens, frontier, seen = [], sorted(n for n, d in indeg.items() if d == 0), 0
    while frontier:
        gens.append(frontier)
        seen += len(frontier)
        nxt = []
        for n in frontier:
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    nxt.append(c)
        frontier = sorted(nxt)
    if seen != len(indeg):
        raise ValueError("cycle among: "
                         + ", ".join(sorted(n for n, d in indeg.items() if d > 0)))
    return gens


# ── the checks ────────────────────────────────────────────────────────────────

def stage_parents(g) -> dict[str, list[str]]:
    """Stage -> the stages that must run before it, over WRITE edges only.

    An `updates()` self-edge is excluded by construction, and so is a `prior=True` read:
    both are a stage naming a file it does not wait for.
    """
    order = g.order or {}
    parents = {}
    for node, stage in g.stages.items():
        updated = {s.path for s in stage.sites if s.kind == "update"}
        deps = set()
        for s in stage.inputs():
            if s.prior or any(_same(s.path, u) for u in updated):
                continue
            prod = {p for p in g.producers_of(s.path, s.slice) if p != node}
            # A producer that runs LATER is not what this stage read — it is a downstream
            # amendment to the same file. `build_preseason` reads `game_predictions`,
            # which `predict_games` writes before it and `predict_upcoming_games` amends
            # after; counting the amendment as a dependency makes the pair a cycle that
            # the run order plainly does not have.
            earlier = {p for p in prod if order.get(p, -1) < order.get(node, 0)}
            deps |= earlier or prod
        parents[node] = sorted(deps)
    return parents


def find_cycle(g) -> str | None:
    """The cycle in the stage graph, as `toposort` describes it, or None."""
    try:
        toposort(stage_parents(g))
    except ValueError as e:
        return str(e)
    return None


def require_acyclic(g) -> None:
    """Refuse to render or export a graph that has a cycle.

    A cycle means the run order is not a fact — every downstream question (what does this
    depend on, what breaks if it changes, in what order does this run) has no answer, and
    drawing a picture anyway invites someone to trust the picture. `iv check` is where you
    find out WHAT the cycle is; everything that produces a graph artifact stops here.
    """
    cycle = find_cycle(g)
    if cycle:
        raise InvalidatorError(
            f"the stage graph has a CYCLE, so its order is not defined and nothing built "
            f"from it would be true:\n    {cycle}\n"
            f"Run `iv check` for the full picture. A cycle is usually one read that "
            f"should be prior=True, or two stages that should be one.")


def check(g: Graph) -> tuple[list[str], list[str]]:
    """Every structural check. Returns (errors, warnings)."""
    roots = tuple(g.iv.roots)
    errors: list[str] = []
    warns: list[str] = []

    # UNDEFINED NAMES in a stage. A build function is not executed until the pipeline
    # runs it, so a name left behind by a refactor imports perfectly and raises an hour
    # in. Moving that to the preflight is the same job as everything else here.
    names = _static.undefined_names(g.iv)
    if names is None:
        warns.append("pyflakes is not installed, so undefined names were NOT checked")
    for line in names or ():
        errors.append(f"UNDEFINED NAME  {line}")
    for line in _static.missing_imports(g.iv):
        errors.append(f"MISSING MODULE  {line}")

    # A file that declares I/O and nothing runs. Worth saying; not worth graphing, since
    # in a two-pipeline repo it is usually the other pipeline's stage.
    for node in g.unreached:
        warns.append(
            f"NOT RUN  {node} declares I/O, but nothing this pipeline runs reaches it. "
            f"Another pipeline's stage, or one that has fallen out of the refresh.")

    # READ WITH NO PRODUCER — a root is fine, anything else is a missing stage.
    for path in sorted(set(g.readers)):
        if g.producers_of(path):
            continue
        site = g.sites[path][0]
        if path.startswith(roots):
            continue                    # a root arrives out of band, by definition
        msg = (f"READ WITH NO PRODUCER  {path}\n"
               f"    read at {site.location} but nothing writes it. Either a stage is "
               f"missing, or it is a root — put it under one of {list(roots)} or add "
               f"that prefix to the Invalidator's roots=.")
        (warns if site.optional else errors).append(msg)

    # WRITE WITH NO CONSUMER — legal only if something outside the pipeline reads it.
    for path in g.produced:
        if g.consumers_of(path) or g.is_terminal(path):
            continue
        site = next(s for s in g.sites[path] if s.kind in ("write", "update"))
        errors.append(
            f"WRITE WITH NO CONSUMER  {path}\n"
            f"    written at {site.location} and read by nothing. Delete the stage, or "
            f"pass terminal=True if it is consumed outside this pipeline.")

    # TWO WRITERS — one writer plus N updaters is the legal patch chain.
    for path, nodes in g.writers.items():
        if len(set(nodes)) > 1:
            errors.append(
                f"TWO WRITERS  {path}\n"
                f"    fully written by {sorted(set(nodes))}; whichever runs last wins. "
                f"Use updates() if the second one amends the first.")

    # POLICY CONFLICT — the artifact cannot be two things at once.
    #
    # `code` is NOT in the list, and cannot be: it is a digest of the writing function,
    # so two writers of one artifact disagree about it by construction. A chain would
    # report a conflict for simply being a chain.
    for path in g.produced:
        outs = [s for s in g.sites[path] if s.kind in ("write", "update")]
        for attr in ("policy", "fp", "terminal", "version"):
            values = {getattr(s, attr) for s in outs}
            if len(values) > 1:
                errors.append(
                    f"POLICY CONFLICT  {path}\n"
                    f"    its writers disagree about {attr}=: {sorted(map(str, values))}\n"
                    f"    at {', '.join(s.location for s in outs)}")

    # ORDER — per ARTIFACT: satisfied if SOME producer runs earlier.
    if g.iv.declared_order():
        for node, stage in g.stages.items():
            for s in stage.inputs():
                if s.prior:
                    continue
                producers = [p for p in g.producers_of(s.path) if p != node]
                if not producers:
                    continue
                if not any(g.order.get(p, 10 ** 9) < g.order.get(node, 0)
                           for p in producers):
                    errors.append(
                        f"ORDER  {node} reads {s.path} at {s.location}, but its only "
                        f"producer(s) {producers} run later. Move the stage, or pass "
                        f"prior=True if reading the previous run's copy is the intent.")
    else:
        warns.append(
            "ORDER  not checked: the pipeline declares no run order. Pass "
            "stages=[...] or order_from=\"refresh.sh\" to Invalidator().")

    # CYCLE — over WRITE edges only, so an updates() self-edge is excluded by construction.
    cycle = find_cycle(g)
    if cycle:
        errors.append(f"CYCLE  {cycle}")

    # GUARDED FETCH — the guard against this package's own silent failure.
    #
    # An artifact's id is H(its own fingerprint, its metadata, its inputs' ids), and the
    # check recomputes it from the STORED fingerprint. So an artifact with no id-bearing
    # inputs has nothing in its id that can move on its own: once stamped it is current
    # forever. Guard a fetcher on its own output and it fetches once and never again —
    # which is indistinguishable, from the outside, from the cache working perfectly.
    #
    # Three policies say "yes, that is the intent": `settled` (fetch-once history, where
    # the question is coverage rather than staleness), `manual` (rebuilt on demand), and
    # `clock` (a new day is the input, checked against the record's `fetched`).
    for path in g.produced:
        site = next(s for s in g.sites[path] if s.kind in ("write", "update"))
        if site.policy in ("settled", "manual", "clock"):
            continue
        stage = g.stages[g.producers_of(path)[0]]
        if any(not _same(s.path, path) for s in stage.inputs()):
            continue                    # it has real inputs; they move it
        if not (site.guarded or any(_same(p, path) for p in stage.guards)):
            # Unguarded, so it runs every time. Correct for a fetcher — but say where it
            # comes from, or the graph shows an artifact that appeared out of nothing.
            if not stage.externals():
                warns.append(
                    f"NO PROVENANCE  {path}\n"
                    f"    written at {site.location} from nothing this pipeline declares. "
                    f"If it is fetched, say from where with iv.external(...).")
            continue
        errors.append(
            f"GUARDED FETCH  {path}\n"
            f"    {stage.node} guards on {path}, which is built from no declared "
            f"input. "
            f"Nothing in its id can move, so after the first run this stage never runs "
            f"again.\n"
            f"    Drop the guard if it fetches; or say the staleness question does not "
            f"apply with policy=\"settled\" (fetch-once history), policy=\"manual\" "
            f"(rebuilt by hand), or policy=\"clock\" (once a day).")

    return errors, warns


# ── declared vs recorded ──────────────────────────────────────────────────────

def drift(g: Graph, events: list[dict]) -> tuple[list[str], list[str]]:
    """What a run DECLARED, against what the static scan says it declares.

    Be precise about what this can and cannot see. The trace records every `iv.reads` /
    `iv.writes` the run executed — declarations, not file opens. So this compares two
    views of the same declarations: the one the scan derives by walking calls from each
    entry point, and the one the process actually reached.

      recorded - declared  is an ERROR. The run declared it and the scan does not see it,
                           so the artifact's stored input map is missing an edge and it
                           can go stale without noticing. A module-level alias breaking
                           the call walk is the shape this catches.
      declared - recorded  is a WARN.  An absent optional=True input, or a branch this
                           run did not take, produces it legitimately.

    What it CANNOT catch is a bare `pl.read_parquet` that no declaration covers — it is
    absent from both sides. Only wrapping the I/O primitive sees that, which is a
    different mechanism and not this one.
    """
    errors, warns = [], []
    rec_by_node: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"read": set(), "write": set()})
    # Stages whose work the cache skipped. `declared - recorded` says nothing about them:
    # of course a stage that did not run did not read its inputs. Two ways to skip — the
    # whole step, and every partition of one — and both have to be visible here or the
    # report is dominated by the cache doing its job.
    idle: set[str] = set()
    for ev in events:
        if ev.get("kind") == "skip":
            idle.add(ev["node"])
            continue
        if ev.get("kind") != "io" or ev.get("op") == "external":
            continue                    # an external carries no id and no path to diff
        op = "write" if ev.get("op") == "write" else "read"
        rec_by_node[ev["node"]][op].add(ev["rel"])
        if op == "write" and ev.get("built") == []:
            idle.add(ev["node"])        # a partitioned write that reused every partition

    for node in sorted(set(rec_by_node) | set(g.stages)):
        stage = g.stages.get(node)
        sites_r = stage.inputs() if stage else ()
        sites_w = stage.outputs() if stage else ()
        declared_r = {s.path for s in sites_r}
        declared_w = {s.path for s in sites_w}
        # Arms of one if/elif/else are alternatives: no run takes more than one, so a
        # recorded arm answers for its siblings. Without this the untaken half of
        # `data.load`'s parent-vs-sub-league fork is reported unseen on every run.
        alts: dict[str, set[str]] = defaultdict(set)
        for site in (*sites_r, *sites_w):
            if site.branch:
                alts[site.branch].add(site.path)
        rec = rec_by_node.get(node, {"read": set(), "write": set()})

        def _sibling_seen(d: str, seen: set[str]) -> bool:
            return any(d in group and any(_same(a, r) for a in group for r in seen)
                       for group in alts.values())

        for op, declared in (("read", declared_r), ("write", declared_w)):
            extra = {r for r in rec[op]
                     if not any(_same(d, r) for d in declared)}
            for r in sorted(extra):
                errors.append(f"UNDECLARED {op.upper()}  {node} -> {r}")
            missing = {d for d in declared
                       if not any(_same(d, r) for r in rec[op])
                       and not _sibling_seen(d, rec[op])}
            for d in sorted(missing):
                if node in rec_by_node and node not in idle:
                    warns.append(f"declared but not seen  {node} -> {d} ({op})")
    if idle:
        # Said out loud rather than silently dropped: a quiet report and a report with
        # nothing to say should not look the same.
        warns.append(f"{len(idle)} stage(s) skipped their work this run, so their "
                     f"declared inputs are not diffed: {', '.join(sorted(idle))}")
    return errors, warns


# ── export ────────────────────────────────────────────────────────────────────

def export(g: Graph) -> dict:
    """`{nodes, parent_map}` — dbt's `target/manifest.json` shape, so anything that reads
    a dbt manifest reads this too."""
    nodes: dict[str, dict] = {}
    for node, stage in g.stages.items():
        nodes[node] = {"kind": "stage", "order": g.order.get(node),
                       "reads": len(stage.inputs()),
                       "writes": len(stage.outputs())}
    for path in g.artifacts:
        outs = [s for s in g.sites[path] if s.kind in ("write", "update")]
        nodes[path] = {
            "kind": "artifact",
            "terminal": g.is_terminal(path),
            "producers": g.producers_of(path),
            "why": outs[0].why if outs else g.sites[path][0].why,
            "policy": outs[0].policy if outs else None,
            "code": bool(outs and outs[0].code),
        }
    return {"schema": 1, "generated_by": "iv",
            "data_version": g.iv.data_version,
            "nodes": nodes, "parent_map": g.parent_map(),
            "stage_parent_map": g.stage_parents()}
