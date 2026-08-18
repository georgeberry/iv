"""What a run actually touched, appended as ndjson.

The static scan says what the code declares. This says what it did. Off unless the
`Invalidator` was given a `trace=` path (or `$INVALIDATOR_TRACE` names one); when off the
cost is one attribute check per call.

    INVALIDATOR_TRACE=.invalidator/trace.ndjson ./refresh.sh
    iv drift

Two things about the file are load-bearing:

  * **It is appended, never truncated.** A stage that skips on its stamp records nothing,
    so one run is never the whole graph. The trace is a union across runs, and `drift`
    reads it as one.
  * **`RECORDER_VERSION` is bumped when the MEANING of a field changes.** A union across
    two spellings of the same artifact is a graph with two nodes where it wants one, so
    the loader drops older events rather than merging them.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

RECORDER_VERSION = 2


def emit(iv, kind: str, **fields) -> None:
    """Append one event, unless tracing is off or we are inside `bookkeeping()`."""
    if iv.trace_path is None or iv._depth:
        return
    if iv._trace_fh is None:
        iv.trace_path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered: many stage processes append to one file concurrently.
        iv._trace_fh = iv.trace_path.open("a", buffering=1)
    iv._trace_fh.write(json.dumps({
        "v": RECORDER_VERSION,
        "kind": kind,
        "node": iv.node(),
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
