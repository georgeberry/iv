from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .errors import StateError

RECORDER_VERSION = 2


def emit(iv, kind: str, **fields) -> None:
    if iv.trace_path is None or iv._depth:
        return
    if iv._trace_fh is None:
        iv.trace_path.parent.mkdir(parents=True, exist_ok=True)
        iv._trace_fh = iv.trace_path.open("a", buffering=1)
    iv._trace_fh.write(json.dumps({
        "v": RECORDER_VERSION,
        "kind": kind,
        "node": iv.node(),
        "pid": os.getpid(),
        "t": round(time.time(), 3),
        **fields,
    }, default=str) + "\n")


def age_of(events: list[dict]) -> float | None:
    import time

    stamps = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    return time.time() - max(stamps) if stamps else None


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out, stale = [], 0
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as e:
            raise StateError(
                f"{path}:{n} is not parseable ({e}). A torn line means the trace was cut "
                f"mid-write, so it no longer describes the run. Delete it and re-run with "
                f"IV_TRACE set.") from e
        if ev.get("v") != RECORDER_VERSION:
            stale += 1
            continue
        out.append(ev)
    if stale:
        print(f"  (dropped {stale} event(s) from an older recorder version)")
    return out
