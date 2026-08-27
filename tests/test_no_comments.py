from __future__ import annotations

import io
import tokenize
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_source_and_test_code_contains_no_comments():
    comments = []
    paths = [ROOT / "example.py", *sorted((ROOT / "src").rglob("*.py")),
             *sorted((ROOT / "tests").rglob("*.py"))]
    for path in paths:
        tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        comments.extend(
            f"{path.relative_to(ROOT)}:{token.start[0]}"
            for token in tokens
            if token.type == tokenize.COMMENT
        )
    for path in ROOT.glob("*.toml"):
        comments.extend(
            f"{path.relative_to(ROOT)}:{line_number}"
            for line_number, line in enumerate(path.read_text().splitlines(), 1)
            if line.lstrip().startswith("#")
        )
    assert comments == []
