"""Two stages running at once, and the state they share.

The stamps used to live in one `state.json` that every stage rewrote whole, so a parallel
run lost records: eight artifacts written, four stamped. The failure direction was safe —
an unstamped artifact rebuilds — but the cache did not work, which is the same as not
having one.

The barrier is what gives these tests teeth. Started together, N python processes drift
apart by more than the read-modify-write window; held at a barrier until all of them have
arrived, they enter it together, which is the case that has to be safe.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import polars as pl
import pytest

from invalidator import record as _rec
from invalidator.state import read_records

VENV_PY = sys.executable
N = 6

PIPELINE = '''
    from invalidator import Invalidator

    iv = Invalidator(data_root="data", data_version="v1", source_dirs=["stages"])
'''

BARRIER = '''
    """Hold every stage until all of them are here, so the stamps really do collide."""
    import os
    import time


    def wait(n):
        d = os.path.join(os.path.dirname(__file__), "ready")
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f"{os.getpid()}"), "w").close()
        for _ in range(2000):
            if len(os.listdir(d)) >= n:
                return
            time.sleep(0.005)
        raise SystemExit(f"barrier timed out with {len(os.listdir(d))}/{n}")
'''

STAGE = '''
    """One of several stages run at the same time."""
    import polars as pl
    from barrier import wait
    from pipeline import iv

    @iv.step("processed/out_{k}.parquet",
             why="one stage's output; every stage writes its own", terminal=True)
    def build(out):
        src = pl.read_parquet(iv.reads("raw/src.parquet", why="the shared root"))
        wait({n})
        src.with_columns(pl.col("a") + {k}).write_parquet(out)

    build()
'''

FRAME = pl.DataFrame({"a": [1, 2, 3]})


@pytest.fixture
def fanout(project):
    """N independent stages over one root, each writing its own artifact."""
    (project / "pipeline.py").write_text(textwrap.dedent(PIPELINE).lstrip())
    (project / "barrier.py").write_text(textwrap.dedent(BARRIER).lstrip())
    (project / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0"\n\n'
        '[tool.invalidator]\ninstance = "pipeline:iv"\n')
    for k in range(N):
        (project / "stages" / f"stage_{k}.py").write_text(
            textwrap.dedent(STAGE).lstrip().replace("{k}", str(k)).replace("{n}", str(N)))
    raw = project / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    FRAME.write_parquet(raw / "src.parquet")
    return project


def run_together(project, trace=None) -> list[str]:
    """Every stage at once, one process each. Returns their output."""
    env = {**os.environ, "PYTHONPATH": str(project), "NO_COLOR": "1",
           "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("VIRTUAL_ENV", None)
    for var in ("INVALIDATOR_TRACE", "INVALIDATOR_FORCE"):
        env.pop(var, None)
    if trace is not None:
        env["INVALIDATOR_TRACE"] = str(trace)
    procs = [subprocess.Popen([VENV_PY, f"stages/stage_{k}.py"], cwd=project, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for k in range(N)]
    out = []
    for p in procs:
        body = p.communicate()[0]
        assert p.returncode == 0, body
        out.append(body)
    return out


def state_dir(project):
    return project / "data" / ".invalidator" / "state"


def test_parallel_stages_all_get_stamped(fanout):
    """The bug this layout exists for: every stage's record survives the others."""
    run_together(fanout)

    built = {f"processed/out_{k}.parquet" for k in range(N)}
    for rel in built:
        assert (fanout / "data" / rel).exists(), f"{rel} was not written"
    assert set(read_records(state_dir(fanout))) == built, \
        "a stage's stamp was erased by another stage finishing after it"


def test_a_second_parallel_run_skips_everything(fanout):
    """The point of the stamps: nothing moved, so nothing rebuilds."""
    run_together(fanout)
    out = "".join(run_together(fanout))
    assert out.count("is current — skipping") == N, out


def test_no_temp_files_are_left_behind(fanout):
    """Each writer's temp name carries its pid and thread, so nobody renames another's
    half-written file into place — and nothing survives the run."""
    run_together(fanout)
    assert [p.name for p in state_dir(fanout).iterdir() if not p.name.endswith(".json")] == []


# ── the trace ─────────────────────────────────────────────────────────────────

# Longer than the 8 KiB buffer the trace stream uses, which is the size at which two
# stages could plausibly interleave inside one line. `why=` is user text and has to be a
# literal, so it is the one field a caller can make arbitrarily long — the realistic shape
# of the problem, at an unrealistic size.
LONG_WHY = "one stage's output, explained at length: " + "detail. " * 1200


@pytest.fixture
def chatty(fanout):
    """The same fan-out, but every stage emits an event too big to flush in one piece."""
    for k in range(N):
        p = fanout / "stages" / f"stage_{k}.py"
        p.write_text(p.read_text().replace(
            "why=\"one stage's output; every stage writes its own\"",
            f'why="{LONG_WHY}"'))
    return fanout


def test_the_trace_survives_stages_appending_at_once(chatty):
    """One shared file, N processes, and every line has to arrive whole.

    The stamps are per-artifact files because a shared file broke under exactly this; the
    trace stays shared, so the claim that it holds up is worth a test rather than an
    assumption. See `invalidator.record` for what was measured.
    """
    trace = chatty / ".invalidator" / "trace.ndjson"
    run_together(chatty, trace=trace)

    lines = [l for l in trace.read_text().splitlines() if l.strip()]
    for line in lines:
        json.loads(line)                 # a torn line raises here, which is the point
    written = [e for e in _rec.load(trace) if e.get("op") == "write"]
    assert {e["rel"] for e in written} == {f"processed/out_{k}.parquet" for k in range(N)}


def test_a_torn_line_costs_one_event_not_the_whole_trace(tmp_path):
    """NFS does not honour O_APPEND, so tearing stays possible. `iv drift` reporting a
    dropped line beats `iv drift` raising over an advisory file."""
    trace = tmp_path / "trace.ndjson"
    good = json.dumps({"v": _rec.RECORDER_VERSION, "kind": "io", "rel": "a.parquet"})
    trace.write_text(good + "\n" + good[:40] + good + "\n" + good + "\n")
    assert len(_rec.load(trace)) == 2
