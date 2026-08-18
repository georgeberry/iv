"""The DAG, read out of the source without running anything.

`iv.reads(...)` and `iv.writes(...)` are ordinary calls, so the graph is already in the
code — this reads it back out with `ast`, which means `iv graph` and
`iv check` work on a fresh checkout with no data and nothing ever run. The trace
(`iv drift`) is the cross-check, not the source.

THE RULE THAT MAKES THIS POSSIBLE, and it is enforced here rather than asked for politely:
**the path and `why=` must be literals.** A computed path cannot be read statically, which
is exactly the rot this exists to prevent. If you want to compute one, the stage is doing
too much and wants splitting. The one thing that varies at runtime is the PARTITION:

    iv.reads("raw/box/{season}.parquet", why="...", part={"season": season})

The template is the literal; the value is not. That is also how the graph knows twenty-one
files are one node.

HOW A CALL IS RECOGNISED. The receiver is an `Invalidator` instance whose name is whatever
the project chose and which is usually imported from another module, so there is nothing
reliable to match on there. Instead a call counts if the METHOD name is one of ours AND it
carries a `why=` string literal — which every one of them requires. Two conditions, one of
them a keyword nothing else in Python uses this way, makes a false positive essentially
impossible while working whatever you call the instance.

A stage is a FILE. One process is one stage in the normal case, and a file is what the
pipeline script names, so the file path is the node id.
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import DeclError
from .fingerprint import DIGEST_LEN
from .paths import fields as _template_fields

# method name -> (kind, positional index or None, keyword spelling). Each method names
# its artifact differently — `reads(path)`, `step(output)`, `for_each(..., output=)` — and
# a scanner that knew only one spelling would report every stage using another as
# unproduced.
METHODS: dict[str, tuple[str, object, str]] = {
    "reads": ("read", 0, "path"),
    "collection": ("collection", 0, "path"),
    "writes": ("write", 0, "path"),
    "updates": ("update", 0, "path"),
    "external": ("external", 0, "name"),
    "step": ("write", 0, "output"),
    "for_each": ("write", None, "output"),
    "partitions": ("write", 0, "output"),
}

EXTERNAL_PREFIX = "external:"

_cache: dict[tuple, dict] = {}


@dataclass(frozen=True)
class Site:
    """One call site, as written."""
    kind: str                                   # read | write | update | external
    path: str                                   # the template literal
    why: str
    file: str                                   # project-relative
    line: int
    optional: bool = False
    prior: bool = False
    terminal: bool = False
    fp: str = "data"
    policy: str = "tracked"
    version: str = ""                           # the NAME of an extra version, if any
    code: str = ""                              # source hash, when step(code=True)
    owner: str = ""                             # the @step function this sits inside
    guarded: bool = False                       # @step guards; a bare writes() does not

    @property
    def partitioned(self) -> bool:
        return bool(_template_fields(self.path))

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Stage:
    """One file's call sites, plus the within-file call graph that scopes them.

    A step's reads are rarely all lexically inside it — the first stage migrated in anger
    factored its one read into a `_box_features()` helper shared by two steps, which left
    both steps owning no reads at all and therefore permanently stale. So a step's inputs
    are the reads inside it PLUS the reads inside any same-file function it calls,
    transitively. That is still purely static, and it is what a reader would assume.

    A read in a function from ANOTHER module is still invisible; `iv drift`
    against a real trace is what catches those.
    """
    node: str
    module: str = ""                            # dotted name, for cross-file resolution
    sites: tuple[Site, ...] = ()
    guards: tuple[str, ...] = ()                # artifacts named in a build_if_needed(...)
    calls: tuple[tuple[str, tuple[str, ...]], ...] = ()     # fn -> names it calls
    defines: tuple[str, ...] = ()               # functions defined in this file
    imports: tuple[tuple[str, str, str], ...] = ()          # local name -> (module, attr)

    def _of(self, *kinds: str) -> tuple[Site, ...]:
        return tuple(s for s in self.sites if s.kind in kinds)

    def called_by(self, fn: str) -> tuple[str, ...]:
        return dict(self.calls).get(fn, ())

    def inputs_of(self, owner: str) -> tuple[Site, ...]:
        """Deprecated in favour of the project-wide walk; kept for one-file cases."""
        scope, calls, stack = {owner}, dict(self.calls), [owner]
        while stack:
            for c in calls.get(stack.pop(), ()):
                if c not in scope:
                    scope.add(c)
                    stack.append(c)
        return tuple(s for s in self.inputs() if s.owner in scope)

    def inputs(self) -> tuple[Site, ...]:
        """Everything that must exist BEFORE this stage runs. Id-bearing only."""
        return self._of("read", "collection", "update")

    def outputs(self) -> tuple[Site, ...]:
        """Everything this stage leaves behind."""
        return self._of("write", "update")

    def externals(self) -> tuple[Site, ...]:
        """Sources outside the pipeline. Provenance, not dependencies."""
        return self._of("external")


# ── extracting one call ───────────────────────────────────────────────────────

def _lit_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) \
        else None


def _lit_strs(node: ast.AST | None) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.Tuple, ast.List)):
        out = [_lit_str(e) for e in node.elts]
        return tuple(out) if all(o is not None for o in out) else None
    return None


def _lit_bool(node: ast.AST | None, default: bool) -> bool:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, bool) \
        else default


def _sites(call: ast.Call, kind: str, at: tuple, rel_file: str,
           code: str = "", guarded: bool = False, owner: str = "") -> list[Site]:
    """One call -> one Site per artifact it names. `step([a, b])` names two."""
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    where = f"{rel_file}:{call.lineno}"
    index, keyword = at

    raw = call.args[index] if index is not None and len(call.args) > index \
        else kw.get(keyword)

    paths = _lit_strs(raw)
    if paths is None:
        shown = ast.dump(raw, annotate_fields=False)[:60] if raw is not None else "missing"
        raise DeclError(
            f"{where}: the artifact path must be a string LITERAL, not {shown}. A "
            f"computed path cannot be read without running the code, which is what this "
            f"exists to avoid. For a per-partition path use a template plus part=: "
            f'"raw/box/{{season}}.parquet", part={{"season": season}}.')

    why = _lit_str(kw.get("why"))
    if not why:
        raise DeclError(
            f"{where}: why= is required and must be a string literal saying what "
            f"{paths[0]!r} is for. It is what `iv stage` and `iv graph` "
            f"print; there is nowhere else for it to live.")

    if kind == "external":
        # No path, no fingerprint, no id — provenance only. The graph shows it; the
        # staleness rule cannot see it, which is the honest position.
        return [Site(kind=kind, path=EXTERNAL_PREFIX + paths[0], why=why,
                     file=rel_file, line=call.lineno, owner=owner)]

    # The default has to match the method's own, or the scan prices a check differently
    # from the run that made the record — a collection defaults to a footer read.
    _fp_default = "rows" if kind == "collection" else "data"
    fp = _lit_str(kw.get("fp")) or ("<callable>" if "fp" in kw else _fp_default)
    policy = _lit_str(kw.get("policy")) or "tracked"
    if "policy" in kw and _lit_str(kw["policy"]) is None:
        raise DeclError(f"{where}: policy= must be a string literal")
    version = _lit_str(kw.get("version")) or ""
    if "version" in kw and not version:
        raise DeclError(
            f"{where}: version= must be a string literal naming one of the "
            f"Invalidator's versions, e.g. version=\"model\". Passing the value itself "
            f"would be a name this scan cannot resolve, and then `iv status` "
            f"could never see a bump that a run would see.")

    for p in paths:
        if "part" in kw:
            _check_part_keys(kw["part"], p, where, partial=kind == "collection")
        elif _template_fields(p) and kind not in ("write", "collection"):
            # A partitioned WRITE is one artifact with a partition column, not one file
            # per partition, so its path carries no placeholder.
            raise DeclError(
                f"{where}: {p!r} has placeholder(s) {list(_template_fields(p))} "
                f"but no part=")

    return [Site(kind=kind, path=p, why=why, file=rel_file, line=call.lineno,
                 optional=_lit_bool(kw.get("optional"), False),
                 prior=_lit_bool(kw.get("prior"), False),
                 terminal=_lit_bool(kw.get("terminal"), False),
                 fp=fp, policy=policy, version=version, code=code, guarded=guarded,
                 owner=owner)
            for p in paths]


def _check_part_keys(node: ast.AST, path: str, where: str, *,
                     partial: bool = False) -> None:
    """The part KEYS must be literals even though the values need not be.

    The keys are structure — they say which placeholders this call fills. The values are
    the only genuinely runtime thing in a declaration.

    `partial` for a COLLECTION, whose whole point is the field it leaves free: the seasons
    on disk are discovered, so demanding a value for `{season}` would be demanding the
    answer to the question being asked. An unknown key is still an error either way.
    """
    if not isinstance(node, ast.Dict):
        return                                  # a variable dict; the runtime check catches it
    keys = [_lit_str(k) for k in node.keys]
    if any(k is None for k in keys):
        raise DeclError(f"{where}: part= keys must be string literals")
    needed, given = set(_template_fields(path)), set(keys)
    if given - needed or (needed != given and not partial):
        raise DeclError(
            f"{where}: {path!r} names placeholder(s) {sorted(needed) or 'none'} but "
            f"part= supplies {sorted(given)}")


# ── the code hash ─────────────────────────────────────────────────────────────

def function_digest(node: ast.AST) -> str:
    """A hash of what a function DOES, insensitive to how it is spelled.

    MUST match `invalidator.core.source_digest`, which computes the same thing from a live
    function object — the decorator and `iv status` have to agree or one of them
    is lying. `ast.unparse` normalises whitespace, comments and formatting away, and the
    decorators are stripped, so reformatting or editing a `why=` does not invalidate data
    while a real change to the logic does.
    """
    import copy
    node = copy.deepcopy(node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.decorator_list = []
    return hashlib.sha256(ast.unparse(node).encode()).hexdigest()[:DIGEST_LEN]


# ── walking one file ──────────────────────────────────────────────────────────

class _Visitor(ast.NodeVisitor):
    def __init__(self, rel_file: str) -> None:
        self.rel_file = rel_file
        self.module = _module_name(rel_file)
        self.sites: list[Site] = []
        self.guards: list[str] = []
        self._seen: set[int] = set()            # step calls already handled as decorators
        self._owner = ""                        # the function currently being walked
        self._defined: set[str] = set()         # every function defined in this file
        self.calls: dict[str, set[str]] = {}    # caller -> callee names
        self.imports: dict[str, tuple[str, str]] = {}   # local name -> (module, attr)

    # A `@iv.step(...)` decorator is handled here rather than in visit_Call, because only
    # from the FunctionDef can we see the body it guards and hash it.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._defined.add(node.name)
        step_owner = ""
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            name = _method_name(dec.func)
            if name != "step" or not _has_why(dec):
                continue
            self._seen.add(id(dec))
            # Default TRUE, to match `step`'s runtime default. If the two disagree the
            # run stamps a code hash the CLI cannot see, and the change is invisible to
            # `status` while a run would rebuild — the worst of both.
            tracks_code = _lit_bool(
                next((k.value for k in dec.keywords if k.arg == "code"), None), True)
            digest = function_digest(node) if tracks_code else ""
            # `step` guards by default, which is the whole point of it — so a step over an
            # artifact with no inputs is the guarded-fetch bug, exactly as an explicit
            # build_if_needed over one would be.
            guards = _lit_bool(
                next((k.value for k in dec.keywords if k.arg == "if_needed"), None), True)
            self.sites.extend(_sites(dec, "write", (0, "output"), self.rel_file,
                                     code=digest, guarded=guards, owner=node.name))
            step_owner = node.name
        # A step's reads belong to THAT step, not to the file. The decorator clears the
        # read set on entry, so at runtime a step's inputs are exactly the reads inside
        # it — and the scan has to say the same thing or `why_stale` reports an input the
        # record has never seen. Reads in a separately-defined helper are NOT attributed;
        # `iv drift` is what catches those.
        # Every function is an owner, not just a step one: a helper's reads have to be
        # attributed to the helper so a step that calls it can pick them up.
        outer, self._owner = self._owner, node.name
        try:
            self.generic_visit(node)
        finally:
            self._owner = outer

    visit_AsyncFunctionDef = visit_FunctionDef

    # Imports are recorded so a call into another module of the same project can be
    # followed. A real pipeline reads through its own library — wvorp's stages call
    # `wvorp.data.load` — and a scan that stopped at the file boundary would report every
    # one of them as having no inputs at all.
    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            #  `import wvorp.data`        -> "wvorp" is what gets used, but the module we
            #  `import wvorp.data as d`   -> want is the full dotted name either way.
            self.imports[a.asname or a.name.split(".")[0]] = (a.name, "")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # RELATIVE imports too. A library reaches its siblings with `from .data import x`,
        # and skipping those stopped the walk at the package boundary — wvorp's panel
        # builder then declared NO inputs at all, so new raw data could never make it
        # stale. Resolve against this module's own package: `wvorp.panels` at level 1 is
        # `wvorp`, at level 2 its parent.
        mod = node.module or ""
        if node.level:
            pkg = self.module.rsplit(".", node.level)[0] if self.module else ""
            mod = f"{pkg}.{mod}" if mod else pkg
            if not mod:
                self.generic_visit(node)
                return
        if mod:
            for a in node.names:
                self.imports[a.asname or a.name] = (mod, a.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if id(node) not in self._seen:
            name = _method_name(node.func)
            if name in METHODS and _why_is_computed(node):
                raise DeclError(
                    f"{self.rel_file}:{node.lineno}: {name}(why=...) is not a string "
                    f"literal, so this declaration is invisible to the scan and the "
                    f"artifact silently loses the input. A helper that takes `why` as a "
                    f"parameter declares nothing — spell it at the call site.")
            if name in METHODS and _has_why(node):
                kind, index, keyword = METHODS[name]
                self.sites.extend(_sites(node, kind, (index, keyword), self.rel_file,
                                         owner=self._owner))
            elif name == "build_if_needed":
                self.guards.extend(_guarded_paths(node))
            elif isinstance(node.func, ast.Name):
                # A plain call to something this file might define. Resolved after the
                # walk, since a helper may be defined below its caller.
                self.calls.setdefault(self._owner, set()).add(node.func.id)
            elif isinstance(node.func, ast.Attribute) and \
                    isinstance(node.func.value, ast.Name):
                # `lib.load(...)` — recorded dotted and split when resolving, because the
                # module and the function are in different halves of it. Anything that is
                # not an imported module (`df.select(...)`) simply resolves to nothing.
                self.calls.setdefault(self._owner, set()).add(
                    f"{node.func.value.id}.{node.func.attr}")
        # A function passed BY NAME is reached too, even though nothing here calls it
        # syntactically — `iv.for_each(SEASONS, build_one, ...)` and
        # `iv.build_if_needed(rel, build)` are exactly this, and their reads live in the
        # callee.
        for arg in [*node.args, *(k.value for k in node.keywords)]:
            if isinstance(arg, ast.Name):
                self.calls.setdefault(self._owner, set()).add(arg.id)
        self.generic_visit(node)


def _method_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _has_why(call: ast.Call) -> bool:
    """`why=` as a string literal. Every one of our methods requires it, and nothing else
    in ordinary Python is spelled this way — which is what makes name-matching safe."""
    return any(k.arg == "why" and _lit_str(k.value) for k in call.keywords)


def _why_is_computed(call: ast.Call) -> bool:
    """A `why=` that is present but not a literal.

    Worth its own error rather than a silent skip: the call looks exactly like a
    declaration, the scan cannot read it, and the artifact quietly loses an input. Found
    the hard way — a four-line helper taking `why` as a parameter declared nothing, and
    the only symptom was an artifact with one input where it should have had five.
    """
    return any(k.arg == "why" and _lit_str(k.value) is None for k in call.keywords)


def _guarded_paths(call: ast.Call) -> list[str]:
    """Which artifacts a `build_if_needed(...)` covers. Literals only, like everything
    else here — a computed guard target cannot be checked."""
    arg = call.args[0] if call.args else \
        next((k.value for k in call.keywords if k.arg == "paths"), None)
    return list(_lit_strs(arg) or ())


def scan_file(path: Path, root: Path, keep_empty: bool = False) -> Stage | None:
    rel = str(path.relative_to(root))
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return None
    v = _Visitor(rel)
    v.visit(tree)
    if not v.sites and not keep_empty:
        return None
    # Keep every callee name, not just the same-file ones — an imported name is resolved
    # against the project index after every file has been scanned.
    calls = tuple((fn, tuple(sorted(c))) for fn, c in sorted(v.calls.items()))
    return Stage(node=rel, module=_module_name(rel), sites=tuple(v.sites),
                 guards=tuple(v.guards), calls=calls,
                 defines=tuple(sorted(v._defined)),
                 imports=tuple(sorted((k, m, a) for k, (m, a) in v.imports.items())))


def _module_name(rel: str) -> str:
    """`src/wvorp/data.py` -> `wvorp.data`; `scripts/build_bpm.py` -> `build_bpm`.

    A leading `src/` is stripped because that is the packaging convention, not part of the
    import path. A file that is not importable as a module still gets a name; it simply
    never matches an import.
    """
    parts = rel[:-3].split("/")
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def scan(iv, force: bool = False) -> dict[str, Stage]:
    """Every stage in the configured source dirs, keyed by project-relative file path.

    A file with no declarations of its own is kept IF the project names it as something
    it runs. That is not a corner case: the natural shape is a thin entry point calling a
    library that does the I/O, and dropping the entry point leaves the LIBRARY as the
    node — which cannot be ordered, because the refresh never names it.
    """
    key = (str(iv.project_root), iv.source_dirs)
    if key in _cache and not force:
        return _cache[key]
    entries = set(iv.declared_order() or ())
    out: dict[str, Stage] = {}
    for d in iv.source_dirs:
        base = iv.project_root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            rel = str(p.relative_to(iv.project_root))
            stage = scan_file(p, iv.project_root, keep_empty=rel in entries)
            if stage is not None:
                out[stage.node] = stage
    _cache[key] = out
    return out


def reset() -> None:
    _cache.clear()


# ── matching a concrete artifact to a template ────────────────────────────────

def path_pattern(template: str) -> re.Pattern:
    parts, last = [], 0
    for m in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template):
        parts.append(re.escape(template[last:m.start()]))
        parts.append(r"[^/]+")
        last = m.end()
    parts.append(re.escape(template[last:]))
    return re.compile("^" + "".join(parts) + "$")


def matches(template: str, rel: str) -> bool:
    """Does a concrete rel path come from this template? Exact match when unpartitioned."""
    return template == rel or bool(path_pattern(template).match(rel))


def writers_of(iv, rel: str) -> set[str]:
    """Every stage that DECLARES a write to `rel`.

    More than one is a chain, not a conflict: an artifact several stages build in
    sequence is a normal shape, and the set is what tells a stamp whose contributions
    still belong in the record.
    """
    return {st.node for st in scan(iv).values() for site in st.outputs()
            if matches(site.path, rel)}


def _write_site(iv, rel: str) -> tuple[Stage, Site] | None:
    """The one stage and the one write site that produce `rel`.

    The scan covers every source dir, which in a repo running two pipelines out of one
    tree means two stages can declare the same artifact — wvorp's `xpm.parquet` is
    written by `scripts/build_xpm.py` on one league and `mnba/build_ui_parquets.py` on
    the other. `declared_order()` says which stages THIS configuration actually runs, so
    it breaks the tie; without it, or if it does not resolve to one, there is no single
    producer and saying so is the honest answer.
    """
    hits = [(st, s) for st in scan(iv).values() for s in st.outputs()
            if matches(s.path, rel)]
    if len(hits) > 1:
        order = iv.declared_order()
        if order:
            hits = [h for h in hits if h[0].node in set(order)] or hits
    return hits[0] if len(hits) == 1 else None


def _reach(iv, start_node: str, start_fn: str) -> set[tuple[str, str]]:
    """Every (file, function) reachable from one step, across modules.

    A stage's inputs are the reads it makes plus the reads made by everything it calls —
    including a helper in another module of the same project. wvorp's stages read through
    `wvorp.data.load`, so a walk that stopped at the file boundary would report them as
    having no inputs, and every artifact would be permanently stale.

    Only modules INSIDE the scanned source dirs are followed. A read inside polars, or
    inside a dependency, is not this pipeline's declaration to make.
    """
    stages = scan(iv)
    by_module = {st.module: st for st in stages.values() if st.module}

    seen: set[tuple[str, str]] = set()
    # Module scope too: a module-level `SRC = iv.reads(...)` executes before any step and
    # feeds all of them, so it is every step's input and the scan has to say so.
    stack = [(start_node, start_fn), (start_node, "")]
    while stack:
        node, fn = stack.pop()
        if (node, fn) in seen or node not in stages:
            continue
        seen.add((node, fn))
        st = stages[node]
        imports = {k: (m, a) for k, m, a in st.imports}
        for name in st.called_by(fn):
            if name in st.defines:
                stack.append((node, name))          # a helper in this file
                continue
            base, _, attr_call = name.partition(".")
            target = imports.get(base if attr_call else name)
            if target is None:
                continue
            mod, attr = target
            # Three spellings reach the same place:
            #   from wvorp.data import load   -> ("wvorp.data", "load")
            #   from wvorp import data        -> ("wvorp", "data"), used as data.load(...)
            #   import wvorp.data as d        -> ("wvorp.data", ""),  used as d.load(...)
            for cand in (f"{mod}.{attr}" if attr else mod, mod):
                other = by_module.get(cand)
                if other is None:
                    continue
                # The call site recorded the ATTRIBUTE name (`load`), which is the
                # function to enter; `name` is right when the import named it directly.
                stack.append((other.node, attr_call or attr or name))
                stack.append((other.node, ""))       # module-level reads in that file
                break
    return seen


def sites_of_entry(iv, node: str) -> tuple[Site, ...]:
    """Every declaration an ENTRY POINT is responsible for, its libraries included.

    A stage's declarations are not all in the file the refresh names. wvorp reads through
    `wvorp.data.load` and writes raw feeds through `wvorp.data.fetch_one`, so attributing
    those to `src/wvorp/data.py` makes the LIBRARY a node in the pipeline — one that
    reads the panels every stage reads and writes the feeds every fetcher writes. Ordering
    it is meaningless, and the graph collapses into a single cycle through it.

    So a library is not a node. Its sites belong to whoever reaches it, which is what the
    cross-module walk already computes for inputs; this is the same walk, kept for every
    kind of site rather than just reads.
    """
    stages = scan(iv)
    reach = entry_reach(iv, node)
    # By (file, FUNCTION), not by file. `wvorp.data` holds both `load`, which every stage
    # calls, and `fetch_one`, which only the fetchers do — attributing the whole file
    # would make every stage a writer of every raw feed, and the graph one cycle again.
    return tuple(s for (n, fn) in sorted(reach)
                 for s in stages[n].sites if s.owner == fn)


def entry_reach(iv, node: str) -> set[tuple[str, str]]:
    """Every (file, function) whose declarations belong to entry point `node`."""
    stages = scan(iv)
    if node not in stages:
        return set()
    reach = set()
    for fn in ("", *stages[node].defines):
        reach |= _reach(iv, node, fn)
    return reach


def entry_owners(iv, node: str) -> set[str]:
    """Every FILE reached from entry point `node`, itself included."""
    return {n for n, _ in entry_reach(iv, node)}


def inputs_for_artifact(iv, rel: str) -> dict[str, object] | None:
    """`{input rel: fp strategy}` for whatever stage writes `rel`, from the code.

    This is what lets a staleness check notice an input that was ADDED: the stored record
    has no entry for a path the last build never read, so without the code there is
    nothing to compare against.

    None when the answer would be a guess — no producer, or several, which is a condition
    `iv check` reports rather than one this should silently resolve. A template
    input keeps its `{placeholder}`, since the concrete partitions are runtime.
    """
    hit = _write_site(iv, rel)
    if hit is None:
        return None
    stage, site = hit
    stages = scan(iv)
    scope = _reach(iv, stage.node, site.owner)
    out: dict[str, object] = {}
    for node, fn in scope:
        for s in stages[node].inputs():
            if s.owner == fn and not matches(s.path, rel):
                out[s.path] = s.fp
    return out


def version_for_artifact(iv, rel: str) -> str | None:
    """The extra version NAME its write site declares, or None if it declares none."""
    sites = [s for st in scan(iv).values() for s in st.outputs() if matches(s.path, rel)]
    if len(sites) != 1:
        return None
    return sites[0].version or ""


def code_hash_for_artifact(iv, rel: str) -> str | None:
    """The current source hash of the `step(code=True)` function producing `rel`."""
    sites = [s for st in scan(iv).values() for s in st.outputs() if matches(s.path, rel)]
    return sites[0].code if len(sites) == 1 and sites[0].code else None


def spec_for_artifact(iv, rel: str) -> Site | None:
    """The write site that declares `rel`, or None if it is a root or has several."""
    sites = [s for st in scan(iv).values() for s in st.outputs() if matches(s.path, rel)]
    return sites[0] if len(sites) == 1 else None
