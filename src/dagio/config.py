"""Where the project is, where the data is, and what the version axes currently say.

Config comes from `[tool.dagio]` in `pyproject.toml` (or a standalone `dagio.toml`),
overridable by environment variables, overridable again by an explicit `configure()` call
in a test. Discovery walks up from the working directory, so a stage deep in `scripts/`
finds the same root the CLI does.

    [tool.dagio]
    data_root   = "data"                  # relative to the project root, or absolute, or a URI
    source_dirs = ["scripts", "src"]      # what the AST scan walks
    scopes      = ["W", "M"]              # optional; the `scope=` vocabulary

    [tool.dagio.versions]
    data  = "0.1.0"
    model = "0.1.0"

Version axes are named by the project, not by dagio. `data` and `model` are only a
convention — a project with one axis, or four, configures exactly those. `versions=` at a
write site selects which of them enter that artifact's id.

A project whose versions live in code rather than in TOML points at them instead:

    [tool.dagio]
    versions_from = "mypkg.version:VERSIONS"     # a dict[str, str]

which is the shape a repo that already keeps `DATA_VERSION` on a hyperparameter class
needs, so the two cannot drift.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import ConfigError

# A directory holding any of these is a project root. `pyproject.toml` first because it is
# the one that can also CARRY the config; the others only mark the boundary.
_ROOT_MARKERS = ("pyproject.toml", "dagio.toml", ".git")


@dataclass(frozen=True)
class Config:
    project_root: Path
    data_root: object                      # Path, or a CloudPath for a URI data_root
    source_dirs: tuple[str, ...] = ("src", "scripts")
    versions: dict[str, str] = field(default_factory=dict)
    scopes: tuple[str, ...] = ()
    scope: str | None = None               # the ACTIVE scope, from $DAGIO_SCOPE
    state_rel: str = ".dagio/state.json"
    trace_path: Path | None = None         # None = tracing off

    @property
    def state_path(self):
        return self.data_root / self.state_rel


_config: Config | None = None


def _find_root(start: Path) -> Path:
    for d in (start, *start.parents):
        if any((d / m).exists() for m in _ROOT_MARKERS):
            return d
    raise ConfigError(
        f"no project root above {start} — expected one of {_ROOT_MARKERS}. "
        f"Create a pyproject.toml with a [tool.dagio] table, or call "
        f"dagio.configure(project_root=..., data_root=...)."
    )


def _read_toml(root: Path) -> dict:
    """`[tool.dagio]` out of pyproject.toml, or the whole of a standalone dagio.toml."""
    standalone = root / "dagio.toml"
    if standalone.exists():
        return tomllib.loads(standalone.read_text())
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        return tomllib.loads(pyproject.read_text()).get("tool", {}).get("dagio", {})
    return {}


def _mkpath(spec: str):
    """A path string -> Path, or a CloudPath when it names a bucket.

    cloudpathlib is not a dependency. A URI data_root without it is a clear error rather
    than a Path that silently means something else on disk.
    """
    if "://" not in spec:
        return Path(spec)
    try:
        from cloudpathlib import AnyPath
    except ImportError as e:
        raise ConfigError(
            f"data_root {spec!r} is a URI, which needs cloudpathlib installed "
            f"(pip install 'cloudpathlib[gs]')"
        ) from e
    return AnyPath(spec)


def _resolve_versions(raw: dict) -> dict[str, str]:
    """The version axes, either literal in TOML or imported from the project's own code."""
    if "versions_from" in raw:
        target = raw["versions_from"]
        if ":" not in target:
            raise ConfigError(
                f"versions_from must be 'module:attribute', got {target!r}")
        modname, attr = target.split(":", 1)
        import importlib
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            raise ConfigError(f"versions_from {target!r}: cannot import {modname}") from e
        try:
            value = getattr(mod, attr)
        except AttributeError as e:
            raise ConfigError(f"versions_from {target!r}: {modname} has no {attr}") from e
        if callable(value):
            value = value()
        if not isinstance(value, dict):
            raise ConfigError(
                f"versions_from {target!r} must resolve to a dict[str, str], "
                f"got {type(value).__name__}")
        return {str(k): str(v) for k, v in value.items()}
    return {str(k): str(v) for k, v in (raw.get("versions") or {}).items()}


def load(start: Path | None = None) -> Config:
    """Discover and build the config. Called lazily by `get()`; explicit in tests."""
    root = Path(os.environ["DAGIO_PROJECT"]) if "DAGIO_PROJECT" in os.environ \
        else _find_root(Path(start or Path.cwd()).resolve())
    raw = _read_toml(root)

    data_spec = os.environ.get("DAGIO_DATA_ROOT") or raw.get("data_root")
    if not data_spec:
        raise ConfigError(
            f"no data_root — set [tool.dagio] data_root in {root}/pyproject.toml "
            f"or export DAGIO_DATA_ROOT"
        )
    data_root = _mkpath(data_spec)
    if isinstance(data_root, Path) and not data_root.is_absolute():
        data_root = root / data_root

    scopes = tuple(raw.get("scopes") or ())
    scope = os.environ.get("DAGIO_SCOPE") or raw.get("scope")
    if scope and scopes and scope not in scopes:
        raise ConfigError(f"scope {scope!r} is not one of the configured scopes {scopes}")

    trace = os.environ.get("DAGIO_TRACE")
    return Config(
        project_root=root,
        data_root=data_root,
        source_dirs=tuple(raw.get("source_dirs") or ("src", "scripts")),
        versions=_resolve_versions(raw),
        scopes=scopes,
        scope=scope,
        state_rel=os.environ.get("DAGIO_STATE") or raw.get("state") or ".dagio/state.json",
        trace_path=Path(trace) if trace else None,
    )


def get() -> Config:
    global _config
    if _config is None:
        _config = load()
    return _config


def configure(**overrides) -> Config:
    """Override config in-process. For tests and for embedding.

    Every keyword must name a real `Config` field — an unknown one raises rather than
    being ignored, because a silently dropped setting is indistinguishable from a setting
    that did not work.
    """
    global _config
    base = _config
    if base is None:
        allowed = {"project_root", "data_root"}
        if allowed <= set(overrides):
            base = Config(
                project_root=Path(overrides["project_root"]),
                data_root=_mkpath(str(overrides["data_root"]))
                if isinstance(overrides["data_root"], str) else overrides["data_root"],
            )
            overrides = {k: v for k, v in overrides.items() if k not in allowed}
        else:
            base = load()
    unknown = set(overrides) - {f for f in Config.__dataclass_fields__}
    if unknown:
        raise ConfigError(
            f"unknown config field(s): {sorted(unknown)}. "
            f"Known: {sorted(Config.__dataclass_fields__)}")
    _config = replace(base, **overrides)
    return _config


def reset() -> None:
    """Forget the loaded config. Tests only."""
    global _config
    _config = None
