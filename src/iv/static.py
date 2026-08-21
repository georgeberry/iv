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
    prior: bool = False
    terminal: bool = False
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

    def of(self, *kinds: str) -> tuple[Site, ...]:
        return tuple(s for s in self.sites if s.kind in kinds)

    @property
    def inputs(self) -> tuple[Site, ...]:
        return self.of("read")

    @property
    def triggers(self) -> tuple[Site, ...]:
        return tuple(s for s in self.of("read") if not s.prior)

    @property
    def outputs(self) -> tuple[Site, ...]:
        return self.of("write", "constant")

    def reachable_from(self, fn: str) -> set[str]:
        calls = dict(self.calls)
        seen, stack = {fn}, [fn]
        while stack:
            for c in calls.get(stack.pop(), ()):
                if c not in seen and c in self.defines:
                    seen.add(c)
                    stack.append(c)
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
                optional=_lit_bool(kw.get("optional")), prior=_lit_bool(kw.get("prior")),
                terminal=_lit_bool(kw.get("terminal")), owner=owner)


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
        self.steps: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.defines.add(node.name)
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and _method_name(dec.func) == "step":
                self._seen.add(id(dec))
                if not any(k.arg == "why" and isinstance(k.value, ast.Constant)
                           for k in dec.keywords):
                    raise DeclError(
                        f"{self.rel_file}:{dec.lineno}: why= is required on @iv.step and "
                        f"must be a string literal saying what the stage is for.")
                self.steps.add(node.name)
        prev, self._owner = self._owner, node.name
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

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Name):
            self.aliases[node.targets[0].id] = node.value.id
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
            if isinstance(arg, ast.Name):
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
    return Stage(node=rel, sites=tuple(v.sites), steps=tuple(sorted(v.steps)),
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
        for f in sorted(base.rglob("*.py")):
            if "__pycache__" in f.parts:
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
