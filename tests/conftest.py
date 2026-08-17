from __future__ import annotations

import pytest

import dagio
from dagio import config as _config
from dagio import io as _io
from dagio import record as _rec
from dagio import state as _state


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project: a source tree, a data root, and one version axis."""
    root = tmp_path / "proj"
    (root / "stages").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[tool.dagio]\n'
        'data_root = "data"\n'
        'source_dirs = ["stages"]\n'
        '\n[tool.dagio.versions]\n'
        'data = "0.1.0"\n'
        'model = "0.1.0"\n'
    )
    (root / "data").mkdir()

    for var in ("DAGIO_TRACE", "DAGIO_SCOPE", "DAGIO_STATE", "DAGIO_DATA_ROOT",
                "DAGIO_STAGE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DAGIO_PROJECT", str(root))
    monkeypatch.chdir(root)

    _config.reset()
    _state.reset()
    _io.reset()
    _rec.reset()
    yield root
    _config.reset()
    _state.reset()
    _io.reset()
    _rec.reset()


@pytest.fixture
def bump(project):
    """Change a version axis, the way editing pyproject.toml would."""
    def _bump(axis: str, value: str):
        cfg = dagio.get_config()
        versions = dict(cfg.versions)
        versions[axis] = value
        _config.configure(versions=versions)
        _state.reset()
    return _bump


def fresh_process():
    """Simulate the next stage starting: a new process reads nothing yet."""
    _io.reset()
    _state.reset()
