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

RECORDER_VERSION = 2


def emit(iv, kind: str, **fields) -> None:
    """Append one event, unless tracing is off or we are inside `bookkeeping()`.

    NEVER RAISES. The trace is observability: it describes the run, it is not part of
    it, and a pipeline that dies because its logging could not serialise a value has its
    priorities backwards. A `datetime.date` in a `part=` did exactly that — 102 seconds
    into a stage, after the fits.

    `default=str` handles the ordinary case, which is a `part=` value that is not a
    string; `render` already stringifies those for the PATH, so nothing required them to
    be strings in the first place. Anything stranger is reported once and dropped.
    """
    if iv.trace_path is None or iv._depth:
        return
    try:
        if iv._trace_fh is None:
            iv.trace_path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered: many stage processes append to one file concurrently, and
            # the flush-on-newline keeps one event to one write. See the module docstring.
            iv._trace_fh = iv.trace_path.open("a", buffering=1)
        iv._trace_fh.write(json.dumps({
            "v": RECORDER_VERSION,
            "kind": kind,
            "node": iv.node(),
            "pid": os.getpid(),
            "t": round(time.time(), 3),
            **fields,
        }, default=str) + "\n")
    except Exception as e:                  # noqa: BLE001 — see the docstring
        global _WARNED
        if not _WARNED:
            _WARNED = True
            print(f"  iv: trace disabled — {type(e).__name__}: {e}. "
                  f"The run continues; `iv drift` will have nothing to read.")
        iv.trace_path = None


_WARNED = False


def age_of(events: list[dict]) -> float | None:
    """Seconds since the NEWEST event, or None if the trace is empty or unstamped."""
    import time

    stamps = [e["t"] for e in events if isinstance(e.get("t"), (int, float))]
    return time.time() - max(stamps) if stamps else None


def load(path: Path) -> list[dict]:
    """Every event in a trace, dropping any written by an older recorder."""
    if not path.exists():
        return []
    out, stale, torn = [], 0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            # A line that does not parse is a line two writers interleaved, or one a kill
            # cut in half. Dropping it costs one event out of a union; raising costs
            # `iv drift` entirely, over a file that is advisory to begin with.
            torn += 1
            continue
        if ev.get("v") != RECORDER_VERSION:
            stale += 1
            continue
        out.append(ev)
    if stale:
        print(f"  (dropped {stale} event(s) from an older recorder version)")
    if torn:
        print(f"  (dropped {torn} unparseable line(s) — a torn concurrent append)")
    return out
