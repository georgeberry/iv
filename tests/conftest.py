from __future__ import annotations

import textwrap


def write_stage(project, rel: str, body: str) -> None:
    """Drop a source file into a project, dedented. Only `iv preflight` reads files."""
    p = project / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip())
