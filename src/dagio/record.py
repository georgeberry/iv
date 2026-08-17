"""What a run actually touched, appended as ndjson.

The AST scan says what the code declares. This says what it did. Off unless `$DAGIO_TRACE`
names a file; when off the cost is one dict lookup per call.

    DAGIO_TRACE=.dagio/trace.ndjson ./refresh.sh
    dagio drift

Two things about the file are load-bearing:

  * **It is appended, never truncated.** A stage that skips on its stamp records nothing,
    so one run is never the whole graph. The trace is a union across runs, and `drift`
    reads it as one.
  * **`RECORDER_VERSION` is bumped when the MEANING of a field changes.** A union across
    two spellings of the same artifact is a graph with two nodes where it wants one, so
    the loader drops older events rather than merging them.

`bookkeeping()` suppresses recording by SCOPE, not by path. Computing whether an artifact
is stale means reading its inputs, and recorded plainly that makes every raw feed an input
of every guarded stage — including the ones that never touch it. Being guarded is not the
same as depending on the data.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

RECORDER_VERSION = 1

_state: dict = {"fh": None, "node": None, "ready": False, "depth": 0}


def _node() -> str:
    """The stage, as a project-relative script path. One process is one stage."""
    override = os.environ.get("DAGIO_STAGE")
    if override:
        return override
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0:
        return "<interactive>"
    p = Path(argv0).resolve()
    try:
        from .config import get
        root = get().project_root
        return str(p.relative_to(root))
    except Exception:
        return p.name


def _init() -> None:
    _state["ready"] = True
    try:
        from .config import get
        dest = get().trace_path
    except Exception:
        dest = None
    if dest is None:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    _state["fh"] = dest.open("a", buffering=1)     # line-buffered; many processes append
    _state["node"] = _node()


def enabled() -> bool:
    if not _state["ready"]:
        _init()
    return _state["fh"] is not None


@contextmanager
def bookkeeping():
    """I/O in this scope is the pipeline inspecting itself, not a data edge. Re-entrant."""
    _state["depth"] += 1
    try:
        yield
    finally:
        _state["depth"] -= 1


def suppressed() -> bool:
    return bool(_state["depth"])


def record(kind: str, **fields) -> None:
    if not enabled() or _state["depth"]:
        return
    _state["fh"].write(json.dumps({
        "v": RECORDER_VERSION,
        "kind": kind,
        "node": _state["node"],
        "pid": os.getpid(),
        "t": round(time.time(), 3),
        **fields,
    }) + "\n")


def load(path: Path) -> list[dict]:
    """Every event in a trace, dropping any written by an older recorder."""
    if not path.exists():
        return []
    out, stale = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ev = json.loads(line)
        if ev.get("v") != RECORDER_VERSION:
            stale += 1
            continue
        out.append(ev)
    if stale:
        print(f"  (dropped {stale} event(s) from an older recorder version)")
    return out


def reset() -> None:
    """Close and forget the handle. Tests only."""
    if _state["fh"] is not None:
        _state["fh"].close()
    _state.update({"fh": None, "node": None, "ready": False, "depth": 0})
