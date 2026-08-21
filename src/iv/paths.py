"""A data-root spec in, a concrete path out.

Datasets are named by their path relative to the root — `processed/box_features/`, never
an absolute path and never a bucket URI. That is what lets a trace from one tree and a
trace from another talk about the same dataset, and what lets an id survive moving the
data somewhere else.

There is nothing here about templates or partitions. A partition is a segment of a SHARD
name inside a dataset directory, never a placeholder in the dataset's path, so nothing here
ever renders a path from a dict. See `iv.shards`.
"""
from __future__ import annotations

from pathlib import Path

from .errors import ConfigError


def mkpath(spec, project_root: Path | None):
    """A root spec -> Path, or a CloudPath when it names a bucket.

    cloudpathlib is not a dependency. A URI without it is a clear error rather than a Path
    that silently means something else on local disk.
    """
    if not isinstance(spec, str):
        return spec
    if "://" not in spec:
        p = Path(spec)
        return p if p.is_absolute() or project_root is None else project_root / p
    try:
        from cloudpathlib import AnyPath
    except ImportError as e:
        raise ConfigError(
            f"root {spec!r} is a URI, which needs cloudpathlib installed "
            f"(pip install 'cloudpathlib[gs]')") from e
    return AnyPath(spec)
