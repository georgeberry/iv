from __future__ import annotations

import textwrap

import pytest

from iv import Invalidator
from iv import static as _static


@pytest.fixture(autouse=True)
def _clear_scan_cache():
    """The static scan is memoised per (project_root, source_dirs), and every test builds
    a fresh temp project — so a stale entry from a previous test would be invisible."""
    _static.reset()
    yield
    _static.reset()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A throwaway project directory with a source tree and a data root."""
    root = tmp_path / "proj"
    (root / "stages").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "demo"\nversion = "0"\n')
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE",
                "IV_INSTANCE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def iv(project):
    """An Invalidator over that project. Constructor-only config, so this is all of it."""
    return Invalidator(
        data_root=project / "data",
        data_version="v1",
        source_dirs=["stages"],
        project_root=project,
    )


def write_stage(project, rel: str, body: str) -> None:
    """Drop a stage file into the project, dedented."""
    p = project / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip())
    _static.reset()
