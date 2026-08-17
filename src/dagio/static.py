"""The DAG, read out of the source without running anything.

`dagio.reads(...)` and `dagio.writes(...)` are ordinary calls, so the graph is already in
the code — this reads it back out with `ast`, which means `dagio graph` and `dagio check`
work on a fresh checkout with no data and no run. The trace (`dagio drift`) is the
cross-check, not the source.

THE RULE THAT MAKES THIS POSSIBLE, and it is enforced here rather than asked for politely:
**the path and `why=` must be literals.** A computed path cannot be read statically, which
is exactly the rot this exists to prevent. If you want to compute one, the stage is doing
too much and wants splitting. The one thing that varies at runtime is the PARTITION:

    dagio.reads("raw/player_box/{season}.parquet", why="...", part={"season": season})

The template is the literal; the value is not. That is also how the graph knows twenty-one
files are one node.

A stage is a FILE. One process is one stage in the normal case, and a file is what the
pipeline script names, so the file path is the node id.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import DeclError
from .paths import fields as _template_fields

FUNCS = {"reads": "read", "writes": "write", "updates": "update",
         "external": "external"}

# The partition helpers declare a write too, they just spell the path differently:
# `for_each(..., artifact="...")` and `PartitionCache("...", key)`. A scanner that only
# knew the two plain spellings would report every partitioned artifact as unproduced.
PARTITION_FUNCS = {"for_each": "artifact", "PartitionCache": 0}

EXTERNAL_PREFIX = "external:"

_cache: dict | None = None


@dataclass(frozen=True)
class Site:
    """One `reads`/`writes`/`updates` call, as written."""
    kind: str                                   # read | write | update
    path: str                                   # the template literal
    why: str
    file: str                                   # project-relative
    line: int
    optional: bool = False
    prior: bool = False
    terminal: bool = False
    fp: str = "data"
    versions: tuple[str, ...] = ("data",)
    policy: str = "tracked"
    scope: tuple[str, ...] = ()

    @property
    def partitioned(self) -> bool:
        return bool(_template_fields(self.path))

    def applies_to(self, scope: str | None) -> bool:
        return not self.scope or scope is None or scope in self.scope

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Stage:
    """One file's call sites."""
    node: str
    sites: tuple[Site, ...] = ()
    guards: tuple[str, ...] = ()        # artifacts named in a build_if_needed(...)

    def _of(self, *kinds: str, scope: str | None = None) -> tuple[Site, ...]:
        return tuple(s for s in self.sites if s.kind in kinds and s.applies_to(scope))

    def inputs(self, scope: str | None = None) -> tuple[Site, ...]:
        """Everything that must exist BEFORE this stage runs. Id-bearing only."""
        return self._of("read", "update", scope=scope)

    def externals(self, scope: str | None = None) -> tuple[Site, ...]:
        """Sources outside the pipeline. Provenance, not dependencies."""
        return self._of("external", scope=scope)

    def outputs(self, scope: str | None = None) -> tuple[Site, ...]:
        """Everything this stage leaves behind."""
        return self._of("write", "update", scope=scope)


# ── extracting one call ───────────────────────────────────────────────────────

def _literal_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) \
        else None


def _literal_strs(node: ast.AST | None) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.Tuple, ast.List)):
        out = [_literal_str(e) for e in node.elts]
        return tuple(out) if all(o is not None for o in out) else None
    return None


def _literal_bool(node: ast.AST | None, default: bool) -> bool:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, bool) \
        else default


def _site(call: ast.Call, kind: str, rel_file: str,
          path_arg: str | int = 0) -> Site:
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    where = f"{rel_file}:{call.lineno}"

    if isinstance(path_arg, str):
        raw_path = kw.get(path_arg)
    else:
        raw_path = call.args[path_arg] if len(call.args) > path_arg else kw.get("path")
    path = _literal_str(raw_path)
    if path is None:
        raise DeclError(
            f"{where}: the artifact path must be a string LITERAL, not "
            f"{ast.dump(raw_path, annotate_fields=False)[:60] if raw_path else 'missing'}. "
            f"A computed path cannot be read without running the code, which is what this "
            f"exists to avoid. For a per-partition path use a template plus part=: "
            f'"raw/box/{{season}}.parquet", part={{"season": season}}.')

    why = _literal_str(kw.get("why"))
    if not why:
        raise DeclError(
            f"{where}: why= is required and must be a string literal saying what "
            f"{path!r} is for. It is what `dagio stage` and `dagio graph` print; there is "
            f"nowhere else for it to live.")

    fp = _literal_str(kw.get("fp")) or ("<callable>" if "fp" in kw else "data")
    versions = _literal_strs(kw.get("versions")) if "versions" in kw else ("data",)
    if versions is None:
        raise DeclError(f"{where}: versions= must be a literal tuple of axis names")
    policy = _literal_str(kw.get("policy")) or "tracked"
    if "policy" in kw and _literal_str(kw["policy"]) is None:
        raise DeclError(f"{where}: policy= must be a string literal")
    scope = _literal_strs(kw.get("scope")) or () if "scope" in kw else ()

    if kind == "external":
        # No path, no fingerprint, no id — provenance only. The graph shows it; the
        # staleness rule cannot see it, which is the honest position.
        return Site(kind=kind, path=EXTERNAL_PREFIX + path, why=why,
                    file=rel_file, line=call.lineno, scope=tuple(scope))

    # A partitioned WRITE is one artifact with a partition column, not one file per
    # partition, so its path carries no placeholder — the key names the column.
    if "part" in kw:
        _check_part_keys(kw["part"], path, where)
    elif _template_fields(path) and kind != "write":
        raise DeclError(
            f"{where}: {path!r} has placeholder(s) {list(_template_fields(path))} "
            f"but no part=")

    return Site(kind=kind, path=path, why=why, file=rel_file, line=call.lineno,
                optional=_literal_bool(kw.get("optional"), False),
                prior=_literal_bool(kw.get("prior"), False),
                terminal=_literal_bool(kw.get("terminal"), False),
                fp=fp, versions=tuple(versions), policy=policy, scope=tuple(scope))


def _check_part_keys(node: ast.AST, path: str, where: str) -> None:
    """The part KEYS must be literals even though the values need not be.

    The keys are structure — they say which placeholders this call fills. The values are
    the only genuinely runtime thing in a declaration.
    """
    if not isinstance(node, ast.Dict):
        return                                  # a variable dict; the runtime check catches it
    keys = [_literal_str(k) for k in node.keys]
    if any(k is None for k in keys):
        raise DeclError(f"{where}: part= keys must be string literals")
    needed, given = set(_template_fields(path)), set(keys)
    if needed != given:
        raise DeclError(
            f"{where}: {path!r} names placeholder(s) {sorted(needed) or 'none'} but "
            f"part= supplies {sorted(given)}")


# ── walking one file ──────────────────────────────────────────────────────────

class _Visitor(ast.NodeVisitor):
    """Finds dagio calls however the module imported them.

    `import dagio as dg` -> dg.reads(...)   ·   `from dagio import reads` -> reads(...)
    Both are ordinary Python, so both have to be recognised; a package that only saw one
    spelling would silently miss half a codebase's edges.
    """

    def __init__(self, rel_file: str) -> None:
        self.rel_file = rel_file
        self.modules: set[str] = set()          # names bound to the dagio module
        self.direct: dict[str, str] = {}        # local name -> kind
        self.partition_direct: dict[str, str] = {}   # local name -> for_each|PartitionCache
        self.guard_names: set[str] = set()      # local names bound to build_if_needed
        self.sites: list[Site] = []
        self.guards: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for a in node.names:
            if a.name == "dagio" or a.name.startswith("dagio."):
                self.modules.add(a.asname or a.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module and (node.module == "dagio" or node.module.startswith("dagio.")):
            for a in node.names:
                if a.name in FUNCS:
                    self.direct[a.asname or a.name] = FUNCS[a.name]
                elif a.name in PARTITION_FUNCS:
                    self.partition_direct[a.asname or a.name] = a.name
                elif a.name == "build_if_needed":
                    self.guard_names.add(a.asname or a.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else \
            (f.id if isinstance(f, ast.Name) else None)
        via_module = isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) \
            and f.value.id in self.modules

        kind = path_arg = None
        if name in FUNCS and (via_module or (isinstance(f, ast.Name)
                                             and name in self.direct)):
            kind, path_arg = FUNCS[name], 0
        elif name in PARTITION_FUNCS and (via_module or (
                isinstance(f, ast.Name) and name in self.partition_direct)):
            kind, path_arg = "write", PARTITION_FUNCS[name]

        if kind is not None:
            self.sites.append(_site(node, kind, self.rel_file, path_arg))
        elif self._is_guard(f):
            self.guards.extend(self._guarded_paths(node))
        self.generic_visit(node)

    def _is_guard(self, f: ast.AST) -> bool:
        if isinstance(f, ast.Attribute) and f.attr == "build_if_needed":
            return isinstance(f.value, ast.Name) and f.value.id in self.modules
        return isinstance(f, ast.Name) and f.id in self.guard_names

    @staticmethod
    def _guarded_paths(call: ast.Call) -> list[str]:
        """Which artifacts a `build_if_needed(...)` covers. Literals only, like everything
        else here — a computed guard target cannot be checked."""
        arg = call.args[0] if call.args else \
            next((k.value for k in call.keywords if k.arg == "paths"), None)
        one = _literal_str(arg)
        if one:
            return [one]
        many = _literal_strs(arg)
        return list(many) if many else []


def scan_file(path: Path, root: Path) -> Stage | None:
    rel = str(path.relative_to(root))
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return None
    v = _Visitor(rel)
    v.visit(tree)
    return Stage(node=rel, sites=tuple(v.sites), guards=tuple(v.guards)) \
        if v.sites else None


def scan(force: bool = False) -> dict[str, Stage]:
    """Every stage in the configured source dirs, keyed by project-relative file path."""
    global _cache
    if _cache is not None and not force:
        return _cache
    from .config import get
    cfg = get()
    out: dict[str, Stage] = {}
    for d in cfg.source_dirs:
        base = cfg.project_root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            stage = scan_file(p, cfg.project_root)
            if stage is not None:
                out[stage.node] = stage
    _cache = out
    return out


def reset() -> None:
    global _cache
    _cache = None


# ── matching a concrete artifact to a template ────────────────────────────────

def _pattern(template: str) -> re.Pattern:
    parts, last = [], 0
    for m in re.finditer(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", template):
        parts.append(re.escape(template[last:m.start()]))
        parts.append(r"[^/]+")
        last = m.end()
    parts.append(re.escape(template[last:]))
    return re.compile("^" + "".join(parts) + "$")


def matches(template: str, rel: str) -> bool:
    """Does a concrete rel path come from this template? Exact match when unpartitioned."""
    return template == rel or bool(_pattern(template).match(rel))


def producers_of(rel: str, scope: str | None = None) -> list[str]:
    return sorted(st.node for st in scan().values()
                  for s in st.outputs(scope) if matches(s.path, rel))


def consumers_of(rel: str, scope: str | None = None) -> list[str]:
    return sorted(st.node for st in scan().values()
                  for s in st.inputs(scope) if matches(s.path, rel))


def inputs_for_artifact(rel: str) -> dict[str, object] | None:
    """`{input rel: fp strategy}` for whatever stage writes `rel`, from the code.

    This is what lets a staleness check notice an input that was ADDED: the stored record
    has no entry for a path the last build never read, so without the code there is
    nothing to compare against.

    Returns None when the answer would be a guess — no producer, or several, which is a
    condition `dagio check` reports rather than one this should silently resolve. A
    template input keeps its `{placeholder}`, since the concrete partitions are runtime.
    """
    from .config import get
    scope = get().scope
    owners = [st for st in scan().values()
              if any(matches(s.path, rel) for s in st.outputs(scope))]
    if len(owners) != 1:
        return None
    stage = owners[0]
    return {s.path: s.fp for s in stage.inputs(scope) if not matches(s.path, rel)}


def spec_for_artifact(rel: str) -> Site | None:
    """The write site that declares `rel`, or None if it is a root or has several."""
    sites = [s for st in scan().values() for s in st.outputs()
             if matches(s.path, rel)]
    return sites[0] if len(sites) == 1 else None


# ── run order ─────────────────────────────────────────────────────────────────

_INVOKE = re.compile(r"(?:python[0-9.]*|uv run python|python -m)\s+(?:-\S+\s+)*"
                     r"([\w./-]+\.py)")


def declared_order() -> list[str] | None:
    """The order the pipeline actually runs its stages in, if the project says.

        [tool.dagio]
        stages    = ["stages/fetch.py", "stages/build.py"]   # explicit
        order_from = "refresh.sh"                            # or scanned from the script

    Without either, `dagio check` falls back to a topological order — which can still find
    a cycle, but cannot find a stage that runs before its producer, because there is
    nothing to compare the graph against.
    """
    from .config import get
    cfg = get()
    raw = _raw_config()
    if raw.get("stages"):
        return list(raw["stages"])
    src = raw.get("order_from")
    if not src:
        return None
    script = cfg.project_root / src
    if not script.exists():
        raise DeclError(f"order_from names {src}, which does not exist")
    order, seen = [], set()
    for line in script.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue                            # a commented-out stage does not run
        for m in _INVOKE.finditer(stripped):
            node = m.group(1)
            if node not in seen:
                seen.add(node)
                order.append(node)              # first invocation wins
    return order


def _raw_config() -> dict:
    from .config import _read_toml, get
    return _read_toml(get().project_root)
