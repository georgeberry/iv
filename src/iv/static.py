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
    part: tuple = ()
    where: tuple = ()
    sel: tuple | None = ()
    owner: str = ""
    rule: str = ""

    @property
    def location(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class Node:


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
    for project, f in _sources(iv):
        _pf_check(f.read_text(), str(f.relative_to(project)), Reporter(out, err))
    return [l for l in out.getvalue().splitlines() if "undefined name" in l]


def _sources(iv):


    project = Path(iv.project or Path.cwd())
    for d in iv.code:
        base = project / d
        if not base.exists():
            continue
        if base.is_file():
            yield project, base
            continue
        for f in sorted(base.rglob("*.py")):


            if any(part in SKIP_DIRS or part.startswith(".") for part in f.parts):
                continue
            yield project, f


def _imported_modules(path: Path) -> set[str]:


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


    tops = {d.split("/")[0].removesuffix(".py") for d in iv.code}
    bad = []
    for project, f in _sources(iv):
        node = str(f.relative_to(project))
        for mod in sorted(_imported_modules(f)):
            if mod.split(".")[0] not in tops:
                continue
            rel = mod.replace(".", "/")
            if (project / f"{rel}.py").exists() or (project / rel / "__init__.py").exists():
                continue
            bad.append(f"{node}: imports {mod!r}, which is not in this project")
    return bad
