"""Rel path in, concrete path out.

Artifacts are named by their path relative to the data root — `processed/xpm.parquet`,
never an absolute path or a bucket URI. That is what makes a trace from one tree and a
trace from another talk about the same artifact, and what lets an id survive moving the
data somewhere else.

A partitioned artifact is named by a TEMPLATE:

    iv.reads("raw/box/{season}.parquet", why="...", part={"season": "2026"})

The template is the literal, so the static scan can still read it without running
anything; the value is runtime. `{season}` is the partition key, and it is also how the
graph knows twenty-one files are one node.
"""
from __future__ import annotations

import re
from pathlib import Path

from .errors import ConfigError, DeclError

_FIELD = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def mkpath(spec, project_root: Path):
    """A data_root spec -> Path, or a CloudPath when it names a bucket.

    cloudpathlib is not a dependency. A URI without it is a clear error rather than a Path
    that silently means something else on disk.
    """
    if not isinstance(spec, str):
        return spec
    if "://" not in spec:
        p = Path(spec)
        return p if p.is_absolute() else project_root / p
    try:
        from cloudpathlib import AnyPath
    except ImportError as e:
        raise ConfigError(
            f"data_root {spec!r} is a URI, which needs cloudpathlib installed "
            f"(pip install 'cloudpathlib[gs]')") from e
    return AnyPath(spec)


def fields(template: str) -> tuple[str, ...]:
    """The partition keys a template names, in order."""
    return tuple(_FIELD.findall(template))


def render(template: str, part: dict[str, str] | None) -> str:
    """Substitute the partition values. Every mismatch is an error, never a silent pass.

    A missing key would produce a path with a literal `{season}` in it, which then reads
    or writes a file nobody meant; an extra key means the caller believes it is
    partitioning something that is not partitioned.
    """
    needed = set(fields(template))
    given = dict(part or {})
    missing = sorted(needed - set(given))
    extra = sorted(set(given) - needed)
    if missing:
        raise DeclError(
            f"{template!r} needs part={{{', '.join(repr(m) for m in missing)}: ...}}")
    if extra:
        raise DeclError(
            f"{template!r} has no placeholder for part key(s) {extra}; "
            f"it names {sorted(needed) or 'none'}")
    return template.format(**{k: str(v) for k, v in given.items()})


def resolve(iv, rel: str):
    """A rendered rel path -> the concrete path under the data root."""
    if _FIELD.search(rel):
        raise DeclError(
            f"{rel!r} still has an unrendered placeholder; pass part= at the call site")
    return iv.data_root / rel


def to_rel(iv, path) -> str | None:
    """A concrete path -> its rel string, or None if it is outside the data tree."""
    root = str(iv.data_root).rstrip("/")
    s = str(path)
    if s == root:
        return ""
    if s.startswith(root + "/"):
        return s[len(root) + 1:]
    return None
