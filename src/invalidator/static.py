"""The DAG, read out of the source without running anything.

`iv.reads(...)` and `iv.writes(...)` are ordinary calls, so the graph is already in the
code — this reads it back out with `ast`, which means `invalidator graph` and
`invalidator check` work on a fresh checkout with no data and nothing ever run. The trace
(`invalidator drift`) is the cross-check, not the source.

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
    code: str = ""                              # source hash, when step(code=True)
    guarded: bool = False                       # @step guards; a bare writes() does not

    @property
    def partitioned(self) -> bool:
        return bool(_template_fields(self.path))

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Stage:
    """One file's call sites."""
    node: str
    sites: tuple[Site, ...] = ()
    guards: tuple[str, ...] = ()                # artifacts named in a build_if_needed(...)

    def _of(self, *kinds: str) -> tuple[Site, ...]:
        return tuple(s for s in self.sites if s.kind in kinds)

    def inputs(self) -> tuple[Site, ...]:
        """Everything that must exist BEFORE this stage runs. Id-bearing only."""
        return self._of("read", "update")

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
           code: str = "", guarded: bool = False) -> list[Site]:
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
            f"{paths[0]!r} is for. It is what `invalidator stage` and `invalidator graph` "
            f"print; there is nowhere else for it to live.")

    if kind == "external":
        # No path, no fingerprint, no id — provenance only. The graph shows it; the
        # staleness rule cannot see it, which is the honest position.
        return [Site(kind=kind, path=EXTERNAL_PREFIX + paths[0], why=why,
                     file=rel_file, line=call.lineno)]

    fp = _lit_str(kw.get("fp")) or ("<callable>" if "fp" in kw else "data")
    policy = _lit_str(kw.get("policy")) or "tracked"
    if "policy" in kw and _lit_str(kw["policy"]) is None:
        raise DeclError(f"{where}: policy= must be a string literal")

    for p in paths:
        if "part" in kw:
            _check_part_keys(kw["part"], p, where)
        elif _template_fields(p) and kind != "write":
            # A partitioned WRITE is one artifact with a partition column, not one file
            # per partition, so its path carries no placeholder.
            raise DeclError(
                f"{where}: {p!r} has placeholder(s) {list(_template_fields(p))} "
                f"but no part=")

    return [Site(kind=kind, path=p, why=why, file=rel_file, line=call.lineno,
                 optional=_lit_bool(kw.get("optional"), False),
                 prior=_lit_bool(kw.get("prior"), False),
                 terminal=_lit_bool(kw.get("terminal"), False),
                 fp=fp, policy=policy, code=code, guarded=guarded)
            for p in paths]


def _check_part_keys(node: ast.AST, path: str, where: str) -> None:
    """The part KEYS must be literals even though the values need not be.

    The keys are structure — they say which placeholders this call fills. The values are
    the only genuinely runtime thing in a declaration.
    """
    if not isinstance(node, ast.Dict):
        return                                  # a variable dict; the runtime check catches it
    keys = [_lit_str(k) for k in node.keys]
    if any(k is None for k in keys):
        raise DeclError(f"{where}: part= keys must be string literals")
    needed, given = set(_template_fields(path)), set(keys)
    if needed != given:
        raise DeclError(
            f"{where}: {path!r} names placeholder(s) {sorted(needed) or 'none'} but "
            f"part= supplies {sorted(given)}")


# ── the code hash ─────────────────────────────────────────────────────────────

def function_digest(node: ast.AST) -> str:
    """A hash of what a function DOES, insensitive to how it is spelled.

    MUST match `invalidator.core.source_digest`, which computes the same thing from a live
    function object — the decorator and `invalidator status` have to agree or one of them
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
        self.sites: list[Site] = []
        self.guards: list[str] = []
        self._seen: set[int] = set()            # step calls already handled as decorators

    # A `@iv.step(...)` decorator is handled here rather than in visit_Call, because only
    # from the FunctionDef can we see the body it guards and hash it.
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            name = _method_name(dec.func)
            if name != "step" or not _has_why(dec):
                continue
            self._seen.add(id(dec))
            tracks_code = _lit_bool(
                next((k.value for k in dec.keywords if k.arg == "code"), None), False)
            digest = function_digest(node) if tracks_code else ""
            # `step` guards by default, which is the whole point of it — so a step over an
            # artifact with no inputs is the guarded-fetch bug, exactly as an explicit
            # build_if_needed over one would be.
            guards = _lit_bool(
                next((k.value for k in dec.keywords if k.arg == "if_needed"), None), True)
            self.sites.extend(_sites(dec, "write", (0, "output"), self.rel_file,
                                     code=digest, guarded=guards))
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        if id(node) not in self._seen:
            name = _method_name(node.func)
            if name in METHODS and _has_why(node):
                kind, index, keyword = METHODS[name]
                self.sites.extend(_sites(node, kind, (index, keyword), self.rel_file))
            elif name == "build_if_needed":
                self.guards.extend(_guarded_paths(node))
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


def _guarded_paths(call: ast.Call) -> list[str]:
    """Which artifacts a `build_if_needed(...)` covers. Literals only, like everything
    else here — a computed guard target cannot be checked."""
    arg = call.args[0] if call.args else \
        next((k.value for k in call.keywords if k.arg == "paths"), None)
    return list(_lit_strs(arg) or ())


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


def scan(iv, force: bool = False) -> dict[str, Stage]:
    """Every stage in the configured source dirs, keyed by project-relative file path."""
    key = (str(iv.project_root), iv.source_dirs)
    if key in _cache and not force:
        return _cache[key]
    out: dict[str, Stage] = {}
    for d in iv.source_dirs:
        base = iv.project_root / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.py")):
            stage = scan_file(p, iv.project_root)
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


def _owner(iv, rel: str) -> Stage | None:
    """The single stage that writes `rel`, or None if there is not exactly one."""
    owners = [st for st in scan(iv).values()
              if any(matches(s.path, rel) for s in st.outputs())]
    return owners[0] if len(owners) == 1 else None


def inputs_for_artifact(iv, rel: str) -> dict[str, object] | None:
    """`{input rel: fp strategy}` for whatever stage writes `rel`, from the code.

    This is what lets a staleness check notice an input that was ADDED: the stored record
    has no entry for a path the last build never read, so without the code there is
    nothing to compare against.

    None when the answer would be a guess — no producer, or several, which is a condition
    `invalidator check` reports rather than one this should silently resolve. A template
    input keeps its `{placeholder}`, since the concrete partitions are runtime.
    """
    stage = _owner(iv, rel)
    if stage is None:
        return None
    return {s.path: s.fp for s in stage.inputs() if not matches(s.path, rel)}


def code_hash_for_artifact(iv, rel: str) -> str | None:
    """The current source hash of the `step(code=True)` function producing `rel`."""
    sites = [s for st in scan(iv).values() for s in st.outputs() if matches(s.path, rel)]
    return sites[0].code if len(sites) == 1 and sites[0].code else None


def spec_for_artifact(iv, rel: str) -> Site | None:
    """The write site that declares `rel`, or None if it is a root or has several."""
    sites = [s for st in scan(iv).values() for s in st.outputs() if matches(s.path, rel)]
    return sites[0] if len(sites) == 1 else None
