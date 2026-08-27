from __future__ import annotations

from pathlib import Path

from .errors import ConfigError


def mkpath(spec, project: Path | None):
    if not isinstance(spec, str):
        return spec
    if "://" not in spec:
        p = Path(spec)
        return p if p.is_absolute() or project is None else project / p
    try:
        from cloudpathlib import AnyPath
    except ImportError as e:
        raise ConfigError(
            f"root {spec!r} is a URI, which needs cloudpathlib installed "
            f"(pip install 'cloudpathlib[gs]')") from e
    return AnyPath(spec)
