from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from .errors import DeclError

METHODS: dict[str, tuple[str, object, str]] = {
    "reads": ("read", 0, "dataset"),
    "writes": ("write", 0, "dataset"),
    "constants": ("constant", 0, "dataset"),
    "for_each": ("write", None, "dataset"),
    "external": ("external", 0, "name"),
}

EXTERNAL_PREFIX = "external:"

SKIP_DIRS = frozenset({"__pycache__", "node_modules", "site-packages",
                       "build", "dist", "venv"})

_cache: dict[tuple, dict] = {}


def reset() -> None:
    _cache.clear()


@dataclass(frozen=True)
class Site:
    kind: str
    dataset: str
    why: str
    file: str
    line: int
    optional: bool = False
    update_file_on_disk: bool = False
    terminal: bool = False
    part: tuple = ()
    where: tuple = ()
    sel: tuple | None = ()
    owner: str = ""

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Stage:
    node: str
    sites: tuple[Site, ...] = ()
    calls: tuple[tuple[str, tuple[str, ...]], ...] = ()
    defines: tuple[str, ...] = ()
    imports: tuple[tuple[str, str, str], ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()
    steps: tuple[str, ...] = ()
    unguarded: tuple[str, ...] = ()

    def of(self, *kinds: str) -> tuple[Site, ...]:
        return tuple(s for s in self.sites if s.kind in kinds)

    @property
    def inputs(self) -> tuple[Site, ...]:
        return self.of("read")

    @property
    def triggers(self) -> tuple[Site, ...]:
        return tuple(s for s in self.of("read") if not s.update_file_on_disk)

    @property
    def outputs(self) -> tuple[Site, ...]:
        return self.of("write", "constant")

    def reachable_from(self, fn: str) -> set[str]:
        calls = dict(self.calls)
        seen, stack = {fn}, [fn]
        while stack:
            here = stack.pop()
            for c in calls.get(here, ()):
                # A bare name inside a function means that function's own nested def before
                # it means a module-level one of the same name.
                for cand in (f"{here}.{c}", c):
                    if cand in self.defines:
                        if cand not in seen:
                            seen.add(cand)
                            stack.append(cand)
                        break
        return seen

    def outputs_of(self, fn: str) -> tuple[Site, ...]:
        scope = self.reachable_from(fn)
        return tuple(s for s in self.outputs if s.owner in scope)

    @property
    def constants(self) -> tuple[Site, ...]:
        return self.of("constant")

    @property
    def externals(self) -> tuple[Site, ...]:
        return self.of("external")


@dataclass(frozen=True)
class Node:
    """One STAGE: a step function, and the I/O reachable from it.

    A stage used to be a file, which was only ever a proxy — `sys.argv[0]` at runtime and a
    relative path in the scan. Put every stage in one file and that proxy collapses to a
    single node. The unit of work is the step.
    """
    name: str
    file: str
    fn: str
    sites: tuple[Site, ...] = ()
    guarded: bool = False

    def of(self, *kinds: str) -> tuple[Site, ...]:
        return tuple(s for s in self.sites if s.kind in kinds)

    @property
    def inputs(self) -> tuple[Site, ...]:
        return self.of("read")

    @property
    def triggers(self) -> tuple[Site, ...]:
        return tuple(s for s in self.of("read") if not s.update_file_on_disk)

    @property
    def outputs(self) -> tuple[Site, ...]:
        return self.of("write", "constant")

    @property
    def constants(self) -> tuple[Site, ...]:
        return self.of("constant")

    @property
    def externals(self) -> tuple[Site, ...]:
        return self.of("external")

    @property
    def steps(self) -> tuple[str, ...]:
        return (self.fn,) if self.fn else ()

    def outputs_of(self, fn: str) -> tuple[Site, ...]:
        return self.outputs

    def reachable_from(self, fn: str) -> set[str]:
        return {s.owner for s in self.sites}


def nodes_of(stage: Stage) -> list[Node]:
    """A file's stages: one per step function, plus module scope if anything is left.

    A helper called by two steps belongs to both — they really do both read it.
    """
    out: list[Node] = []
    claimed: set[int] = set()
    for fn in stage.steps:
        scope = stage.reachable_from(fn)
        mine = tuple(s for s in stage.sites if s.owner in scope)
        for s in mine:
            claimed.add(id(s))
        out.append(Node(name=f"{stage.node}::{fn}", file=stage.node, fn=fn, sites=mine,
                        guarded=fn not in stage.unguarded))
    # Sites no step reaches still belong to the function they are written in — a helper
    # that drives `for_each`, say. Grouping every leftover into one module-scope node
    # merges producers that have nothing to do with each other, and a merged node both
    # reads and writes the whole chain, which reads as a cycle that is not there.
    loose = [s for s in stage.sites if id(s) not in claimed]
    for fn in stage.defines:
        mine = tuple(s for s in loose if s.owner in stage.reachable_from(fn))
        if mine and not any(s.owner != fn for s in mine if s.owner in stage.defines):
            out.append(Node(name=f"{stage.node}::{fn}", file=stage.node, fn=fn, sites=mine))
            claimed.update(id(s) for s in mine)
    rest = tuple(s for s in stage.sites if id(s) not in claimed)
    if rest or not out:
        out.append(Node(name=stage.node, file=stage.node, fn="", sites=rest))
    return out


def _lit(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _lit_bool(node: ast.AST | None, default: bool = False) -> bool:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, bool) else default


def _method_name(func: ast.AST) -> str | None:
    return func.attr if isinstance(func, ast.Attribute) else None


def _canon(ds: str) -> str:
    return ds.strip().strip("/") + "/"


def _site(call: ast.Call, kind: str, at: tuple, rel_file: str, owner: str) -> Site:
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    where = f"{rel_file}:{call.lineno}"
    index, keyword = at
    raw = call.args[index] if index is not None and len(call.args) > index else kw.get(keyword)

    name = _lit(raw)
    if name is None:
        shown = ast.dump(raw, annotate_fields=False)[:60] if raw is not None else "missing"
        raise DeclError(
            f"{where}: the dataset must be a string LITERAL, not {shown}. A computed name "
            f"cannot be read without running the code, which is what this exists to avoid. "
            f"A partition is a shard NAME inside the dataset, so it belongs in part=, not "
            f"in the path: writes('raw/box/', part={{'season': season}}).")

    why = _lit(kw.get("why"))
    if not why:
        raise DeclError(
            f"{where}: why= is required and must be a string literal saying what {name!r} "
            f"is for. It is what `iv stage` and `iv graph` print; there is nowhere else "
            f"for it to live.")

    if kind == "external":
        return Site(kind=kind, dataset=EXTERNAL_PREFIX + name, why=why,
                    file=rel_file, line=call.lineno, owner=owner)
    return Site(kind=kind, dataset=_canon(name), why=why, file=rel_file, line=call.lineno,
                optional=_lit_bool(kw.get("optional")),
                update_file_on_disk=_lit_bool(kw.get("update_file_on_disk")),
                terminal=_lit_bool(kw.get("terminal")),
                part=_lit_part(kw.get("part")),
                where=_lit_where(kw.get("where")), sel=_lit_sel(kw.get("where")),
                owner=owner)


def _lit_where(node) -> tuple:
    """The subset of a `where=` rule that names partition values outright.

    Only the explicit-list form is read here: it is the one that says which shards are in
    play precisely enough to decide whether two stages touch the same rows. Comparisons
    (`{"lt": ...}`) are left out, so an edge is kept — a missing edge is a wrong DAG.
    """
    if not isinstance(node, ast.Dict):
        return ()
    out = []
    for k, v in zip(node.keys, node.values):
        key = _lit(k)
        if key is None or not isinstance(v, (ast.List, ast.Tuple)):
            continue
        vals = tuple(_lit(e) for e in v.elts)
        if any(x is None for x in vals):
            continue
        out.append((key, tuple(sorted(vals))))
    return tuple(sorted(out))


PART = "\x00PART"


def _lit_sel(node) -> tuple | None:
    """The WHOLE `where=` rule, read off the source — or None if it cannot be.

    This is what makes a shard's key computable before its body runs, which is what removes
    the need to write the inputs down anywhere. `_lit_where` above reads only the subset the
    DAG needs; this has to read all of it, because a selector left out is a dependency
    silently widened to the whole dataset.

    `iv.PART` stands for the partition being built. Every runtime selector in practice is a
    function of it — `where={"season": {"lt": iv.PART}}` — and spelling it as a marker is
    what keeps it readable without executing the closure that would otherwise supply it.

    None means the rule is a value this cannot see: a variable, a call, a comprehension.
    That is not a warning, it is the end of the road — an input that cannot be resolved
    cannot be hashed, so `why_stale` has to say so rather than guess.
    """
    if node is None:
        return ()
    if not isinstance(node, ast.Dict):
        return None
    out = []
    for k, v in zip(node.keys, node.values):
        key = _lit(k)
        if key is None:
            return None
        if isinstance(v, (ast.List, ast.Tuple, ast.Set)):
            vals = tuple(_lit_val(e) for e in v.elts)
            if any(x is None for x in vals):
                return None
            out.append((key, ("in", tuple(sorted(vals)))))
        elif isinstance(v, ast.Dict):
            ops = []
            for ok, ov in zip(v.keys, v.values):
                op, bound = _lit(ok), _lit_val(ov)
                if op is None or bound is None:
                    return None
                ops.append((op, bound))
            out.append((key, ("range", tuple(sorted(ops)))))
        else:
            one = _lit_val(v)
            if one is None:
                return None
            out.append((key, ("in", (one,))))
    return tuple(sorted(out))


def _lit_val(node) -> str | None:
    """A selector value: a string/number literal, or `iv.PART`."""
    if isinstance(node, ast.Attribute) and node.attr == "PART":
        return PART
    if isinstance(node, ast.Constant) and node.value is not None \
            and not isinstance(node.value, bool):
        return str(node.value)
    return None


def _lit_part(node) -> tuple:
    if not isinstance(node, ast.Dict):
        return ()
    out = []
    for k, v in zip(node.keys, node.values):
        key, val = _lit(k), _lit(v)
        if key is None or val is None:
            return ()
        out.append((key, val))
    return tuple(sorted(out))


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel_file: str) -> None:
        self.rel_file = rel_file
        self.sites: list[Site] = []
        self.calls: dict[str, set[str]] = {}
        self.defines: set[str] = set()
        self.imports: dict[str, tuple[str, str]] = {}
        self.aliases: dict[str, str] = {}
        self._owner = ""
        self._seen: set[int] = set()
        # Names BOUND inside a function. A bare `Name` passed as an argument is treated as a
        # function reference — that is how `for_each(seasons, build_one)` is seen — but a
        # local variable that happens to share a step's name is a value, not a reference.
        self.bound: dict[str, set[str]] = {}
        self.steps: list[str] = []
        self.unguarded: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # A nested def is qualified by the function it sits in. Three `for_each` callbacks
        # all spelled `one` are three functions, and a flat name space merges them into a
        # single node that appears to read and write everything all three touch.
        name = f"{self._owner}.{node.name}" if self._owner else node.name
        self.defines.add(name)
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and _method_name(dec.func) == "step":
                self._seen.add(id(dec))
                if not any(k.arg == "why" and isinstance(k.value, ast.Constant)
                           for k in dec.keywords):
                    raise DeclError(
                        f"{self.rel_file}:{dec.lineno}: why= is required on @iv.step and "
                        f"must be a string literal saying what the stage is for.")
                self.steps.append(name)
                if any(k.arg == "if_needed" and _lit_bool(k.value) is False
                       for k in dec.keywords):
                    self.unguarded.append(name)
        prev, self._owner = self._owner, name
        self.generic_visit(node)
        self._owner = prev

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            self.imports[a.asname or a.name.split(".")[0]] = (a.name, "")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if node.level:
            base = self.rel_file.rsplit("/", node.level)[0].replace("/", ".")
            mod = f"{base}.{mod}" if mod else base
        for a in node.names:
            self.imports[a.asname or a.name] = (mod, a.name)

    def _bind(self, target) -> None:
        for n in ast.walk(target):
            if isinstance(n, ast.Name):
                self.bound.setdefault(self._owner, set()).add(n.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            self._bind(tgt)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Name):
            self.aliases[node.targets[0].id] = node.value.id
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._bind(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._bind(item.optional_vars)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _method_name(node.func)
        if id(node) not in self._seen and name in METHODS \
                and any(k.arg == "why" for k in node.keywords):
            kind, index, keyword = METHODS[name]
            self.sites.append(_site(node, kind, (index, keyword), self.rel_file, self._owner))
        called = (name or (node.func.id if isinstance(node.func, ast.Name) else None))
        if called:
            self.calls.setdefault(self._owner, set()).add(called)
        for arg in list(node.args) + [k.value for k in node.keywords]:
            if isinstance(arg, ast.Name) and \
                    arg.id not in self.bound.get(self._owner, ()):
                self.calls.setdefault(self._owner, set()).add(arg.id)
        self.generic_visit(node)


def scan_file(path: Path, root: Path) -> Stage | None:
    rel = str(path.relative_to(root))
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError) as e:
        raise DeclError(f"{rel} does not parse, so its I/O cannot be read: {e}") from e
    v = _Visitor(rel)
    v.visit(tree)
    if not v.sites:
        return None
    return Stage(node=rel, sites=tuple(v.sites), steps=tuple(v.steps),
                 unguarded=tuple(v.unguarded),
                 calls=tuple((k, tuple(sorted(vs))) for k, vs in sorted(v.calls.items())),
                 defines=tuple(sorted(v.defines)),
                 imports=tuple((k, m, a) for k, (m, a) in sorted(v.imports.items())),
                 aliases=tuple(sorted(v.aliases.items())))


def scan(iv) -> dict[str, Stage]:
    root = Path(iv.project_root or Path.cwd())
    key = (str(root), tuple(iv.source_dirs))
    if key in _cache:
        return _cache[key]
    out: dict[str, Stage] = {}
    for d in iv.source_dirs:
        base = root / d
        if not base.exists():
            continue
        # A source_dir may name a FILE. With every declaration in one pipeline file, that is
        # the exact answer to "what should be scanned" — naming its directory would drag in
        # whatever else happens to sit beside it.
        if base.is_file():
            st = scan_file(base, root)
            if st is not None:
                out[st.node] = st
            continue
        for f in sorted(base.rglob("*.py")):
            # Vendored code is not this project's source. Scanning it means parsing several
            # thousand files to find no declarations, and one of them will be a fixture with
            # a deliberately broken encoding — which, correctly, is a hard error.
            if any(part in SKIP_DIRS or part.startswith(".") for part in f.parts):
                continue
            st = scan_file(f, root)
            if st is not None:
                out[st.node] = st
    _cache[key] = out
    return out


def writers_of(iv, dataset: str) -> list[str]:
    ds = _canon(dataset)
    return sorted({n for n, st in scan(iv).items()
                   if any(s.dataset == ds for s in st.outputs)})


def undefined_names(iv) -> list[str] | None:
    try:
        from pyflakes.api import check as _pf_check
        from pyflakes.reporter import Reporter
    except ImportError:
        return None
    import io
    root = Path(iv.project_root or Path.cwd())
    out, err = io.StringIO(), io.StringIO()
    for d in iv.source_dirs:
        base = root / d
        if not base.exists():
            continue
        # A source_dir may name a FILE. With every declaration in one pipeline file, that is
        # the exact answer to "what should be scanned" — naming its directory would drag in
        # whatever else happens to sit beside it. This used to try to file the result in a
        # dict, copied from `scan` where there is one; here `out` is the report pyflakes
        # writes into, so `iv preflight` raised TypeError for every project whose
        # source_dirs names a file — which is the shape the docstring recommends.
        files = [base] if base.is_file() else [
            f for f in sorted(base.rglob("*.py"))
            # Vendored code is not this project's source. Scanning it means parsing several
            # thousand files to find no declarations, and one of them will be a fixture with
            # a deliberately broken encoding — which, correctly, is a hard error.
            if not any(part in SKIP_DIRS or part.startswith(".") for part in f.parts)]
        for f in files:
            _pf_check(f.read_text(), str(f.relative_to(root)), Reporter(out, err))
    return [l for l in out.getvalue().splitlines() if "undefined name" in l]


def missing_imports(iv) -> list[str]:
    root = Path(iv.project_root or Path.cwd())
    tops = {d.split("/")[0] for d in iv.source_dirs}
    bad = []
    for node, st in scan(iv).items():
        for local, mod, attr in st.imports:
            head = mod.split(".")[0]
            if head not in tops:
                continue
            rel = mod.replace(".", "/")
            if (root / f"{rel}.py").exists() or (root / rel / "__init__.py").exists():
                continue
            bad.append(f"{node}: imports {mod!r}, which is not in this project")
    return bad


def imports_of(path: Path, root: Path) -> tuple[tuple[str, str, str], ...]:
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return ()
    v = _Visitor(str(path.relative_to(root)))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            v.visit_Import(n)
        elif isinstance(n, ast.ImportFrom):
            v.visit_ImportFrom(n)
    return tuple((k, m, a) for k, (m, a) in sorted(v.imports.items()))
