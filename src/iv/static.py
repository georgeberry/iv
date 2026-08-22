from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


EXTERNAL_PREFIX = "external:"

SKIP_DIRS = frozenset({"__pycache__", "node_modules", "site-packages",
                       "build", "dist", "venv"})


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
    def externals(self) -> tuple[Site, ...]:
        return self.of("external")


def undefined_names(iv) -> list[str] | None:
    try:
        from pyflakes.api import check as _pf_check
        from pyflakes.reporter import Reporter
    except ImportError:
        return None
    import io
    out, err = io.StringIO(), io.StringIO()
    for root, f in _sources(iv):
        _pf_check(f.read_text(), str(f.relative_to(root)), Reporter(out, err))
    return [l for l in out.getvalue().splitlines() if "undefined name" in l]


def _sources(iv):
    """Every project file to check. A source_dir may name a FILE, which is the exact answer
    when every declaration lives in one pipeline module."""
    root = Path(iv.project_root or Path.cwd())
    for d in iv.source_dirs:
        base = root / d
        if not base.exists():
            continue
        if base.is_file():
            yield root, base
            continue
        for f in sorted(base.rglob("*.py")):
            # Vendored code is not this project's source, and one of its fixtures will have
            # a deliberately broken encoding.
            if any(part in SKIP_DIRS or part.startswith(".") for part in f.parts):
                continue
            yield root, f


def _imported_modules(path: Path) -> set[str]:
    """The dotted module names a file imports. Nothing else about it is read.

    This used to come off the full declaration scan, which had to parse every call in the
    project to answer a question about its import lines.
    """
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            out.add(n.module)
    return out


def missing_imports(iv) -> list[str]:
    """A stage importing a module of this project that is not there.

    Only this project's own modules: an absent third-party package is pip's problem and
    shows up the moment anything runs, but a local module a refactor renamed is a name that
    looks fine and fails at the one moment the stage is finally reached.
    """
    tops = {d.split("/")[0].removesuffix(".py") for d in iv.source_dirs}
    bad = []
    for root, f in _sources(iv):
        node = str(f.relative_to(root))
        for mod in sorted(_imported_modules(f)):
            if mod.split(".")[0] not in tops:
                continue
            rel = mod.replace(".", "/")
            if (root / f"{rel}.py").exists() or (root / rel / "__init__.py").exists():
                continue
            bad.append(f"{node}: imports {mod!r}, which is not in this project")
    return bad
