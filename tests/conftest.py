from __future__ import annotations

import textwrap

import pytest

from iv import static as _static


@pytest.fixture(autouse=True)
def _clear_scan_cache():
    """The scan is memoised per (project_root, source_dirs), and every test builds a fresh
    temp project — so a stale entry from a previous test would be invisible."""
    _static.reset()
    yield
    _static.reset()


def write_stage(project, rel: str, body: str) -> None:
    """Drop a stage file into a project, dedented."""
    p = project / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip())
    _static.reset()
