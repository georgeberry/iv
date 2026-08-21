"""What a run actually touched, appended as ndjson.

The static scan says what the code declares. This says what it did. Off unless the
`Pipeline` was given a `trace=` path (or `$IV_TRACE` names one); when off the
cost is one attribute check per call.

    IV_TRACE=.iv/trace.ndjson ./refresh.sh
    iv drift

Three things about the file are load-bearing:

  * **It is appended, never truncated.** A stage that skips on its stamp records nothing,
    so one run is never the whole graph. The trace is a union across runs, and `drift`
    reads it as one.
  * **`RECORDER_VERSION` is bumped when the MEANING of a field changes.** A union across
    two spellings of the same artifact is a graph with two nodes where it wants one, so
    the loader drops older events rather than merging them.
  * **One event is one line, and the file is SHARED across stages.** The stamps are
    per-artifact files precisely because a shared file is where parallel runs go wrong
    (shards are one file each), so this one is worth being explicit about rather than
    assuming. It stays shared: a trace is a union by construction, one greppable file is
    the point of it, and a file per process per run would accumulate forever.

    MEASURED, because the obvious worry is that two stages interleave inside one line.
    Eight processes and eight threads, 200 events each, at 200 B, 12 KiB and 200 KiB per
    event: zero torn lines in every combination. Line-buffered text flushes on the
    newline and hands an oversized write straight through, so each event is one `write()`
    to an `O_APPEND` fd — and that the kernel does not split. Nothing here needed
    changing, and unbuffered binary would have been WORSE: a raw `FileIO.write` may
    short-write without looping, where the buffered writer completes.

    What is not covered is the filesystem: NFS does not honour `O_APPEND` atomically, and
    a kill can land mid-write. So `load` drops an unparseable line with a count rather
    than taking down `iv drift` over a file that is advisory to begin with.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .errors import StateError

RECORDER_VERSION = 2


def emit(iv, kind: str, **fields) -> None:
    """Append one event, unless tracing is off or we are inside `bookkeeping()`.

    RAISES. It used to swallow everything, on the grounds that a pipeline should not die
    because its logging could not serialise a value — which is true, and it was the wrong
    fix. A trace that quietly stops being written produces an `iv drift` report that is
    confidently wrong about the graph, and nothing anywhere says so.

    The right fix is that serialising cannot fail: `default=str` handles the case that
    actually occurred, a `part=` value that is not a string. Anything else is a bug in this
    function and should be seen.
    """
    if iv.trace_path is None or iv._depth:
        return
    if iv._trace_fh is None:
        iv.trace_path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered: many stage processes append to one file concurrently, and the
        # flush-on-newline keeps one event to one write. See the module docstring.
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
    """Seconds since the NEWEST event, or None if the trace is empty or unstamped."""
    import time

    stamps = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    return time.time() - max(stamps) if stamps else None


def load(path: Path) -> list[dict]:
    """Every event in a trace, dropping any written by an older recorder."""
    if not path.exists():
        return []
    out, stale = [], 0
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as e:
            # Two writers interleaved, or a kill cut one in half. Dropping it silently
            # would make `iv drift` report on a graph it only partly saw, and report it
            # with confidence — so the file is unusable and says so.
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
