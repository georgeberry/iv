"""`reads` and `writes`. The whole thesis is in this file.

The metadata about an artifact lives at the line that touches it, as arguments, because
that is the only place it cannot drift from the code and the only place a reader — human
or agent — is guaranteed to see it. There is no declaration block, no manifest of inputs,
no registry of policies. Delete this file's call sites and the pipeline stops describing
itself, which is the point: the description cannot be stale, because it IS the code.

    poss = pl.scan_parquet(dagio.reads(
        "processed/possessions.parquet",
        why="lineup-level possessions; the per-player minutes denominator"))

    with dagio.writes("processed/box_features.parquet",
                      why="per-(season, player) box prior for the xPM fit") as p:
        out.write_parquet(p)

`reads` hands back a concrete path and remembers the edge. `writes` hands back a concrete
path and, ONLY if the block exits cleanly, fingerprints what you wrote and folds that
fingerprint, the artifact's metadata, and the ids of everything read so far into one id,
which it stamps. A raise inside the block stamps nothing, so a half-written artifact reads
as stale rather than as fresh.

`why=` is required. It is not decoration: it is what `dagio stage` and `dagio graph`
print, and there is nowhere else for it to live.

ORDER MATTERS WITHIN A PROCESS. A write folds in the reads that happened BEFORE it. Read
everything, then write. A read that happens after a write is not in that write's inputs,
and dagio says so rather than letting it pass.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from . import paths as _paths
from . import record as _rec
from . import state as _state
from .errors import DeclError
from .state import Spec

# Everything this process has read, and the strategy to fingerprint it with if it turns
# out to be a root. One process is one stage in the normal case, so this is the stage's
# input set.
_reads: dict[str, object] = {}
_writes: list[str] = []
_pending_fp: dict[str, str] = {}          # set by stamp_content(), consumed by writes()


def _check_why(why: object, path: str) -> str:
    if not isinstance(why, str) or not why.strip():
        raise DeclError(
            f"{path!r}: why= is required and must be a non-empty string saying what this "
            f"artifact is for. It is what `dagio stage` prints; there is nowhere else for "
            f"it to live.")
    return why


def _norm_scope(scope) -> tuple[str, ...] | None:
    if scope is None:
        return None
    return (scope,) if isinstance(scope, str) else tuple(scope)


def reads(path: str, *, why: str,
          optional: bool = False,
          prior: bool = False,
          scope: str | tuple[str, ...] | None = None,
          fp: str | Callable = "data",
          part: dict[str, str] | None = None) -> Path:
    """Declare and resolve an input. Returns the concrete path.

    optional  an absent input degrades a feature rather than failing the stage. Without
              it, a missing input raises here — loudly, at the line that wanted it.
    prior     this deliberately reads the PREVIOUS run's copy, so a producer that runs
              later in the same pipeline is correct rather than an ordering bug.
    scope     which pipeline variant this line belongs to. Affects the STATIC graph only;
              the trace records what actually happened.
    fp        how to fingerprint it IF it is a root. Ignored for anything the pipeline
              writes, whose id is already stamped.
    """
    _check_why(why, path)
    rel = _paths.render(path, part)
    p = _paths.resolve(rel)

    if _writes:
        print(f"  dagio: {rel} is read AFTER this process already wrote "
              f"{_writes[-1]} — it is not among that artifact's inputs. Read first, "
              f"then write.")

    if not p.exists() and not optional:
        raise FileNotFoundError(
            f"{rel} is required by this stage and is not on disk. Its producer has not "
            f"run, or has failed. Pass optional=True if absence is meant to degrade "
            f"rather than fail."
        )

    _reads[rel] = fp
    _rec.record("io", op="read", rel=rel, why=why, optional=optional, prior=prior,
                scope=list(_norm_scope(scope) or ()), part=part or {})
    return p


def external(name: str, *, why: str) -> None:
    """Declare that this stage pulls from something outside the pipeline.

    A fetcher's real input is an API, a scraped page, a vendor drop — something with no
    path and no fingerprint. Declaring it puts the provenance in the graph, where
    `dagio stage` and `dagio graph` will show it, instead of leaving the fetcher looking
    like an artifact that appeared from nowhere.

    It carries no id, so it cannot make anything stale. That is not a gap being papered
    over: it is why a fetcher must not be guarded on its output — see the NEVER REBUILDS
    check in dagio.graph.
    """
    _check_why(why, name)
    _rec.record("io", op="external", rel=f"external:{name}", why=why)


def stamp_content(path, frame) -> None:
    """Fingerprint from a frame already in memory instead of re-reading the file.

    `writes()` yields a plain path so you can write through any library, which means it
    has no way to see what you wrote except by reading it back. When you still hold the
    frame, hand it over and skip the pass.
    """
    rel = _paths.to_rel(path)
    if rel is None:
        raise DeclError(f"{path} is outside the data tree; nothing to stamp")
    if not hasattr(frame, "hash_rows"):
        raise DeclError(
            f"stamp_content needs a frame with .hash_rows() (a polars DataFrame); "
            f"got {type(frame).__name__}")
    _pending_fp[rel] = _frame_fp(frame)


def _frame_fp(frame) -> str:
    import hashlib
    from .fingerprint import DIGEST_LEN
    if frame.height == 0:
        body = "empty|" + ",".join(f"{c}:{t}" for c, t in frame.schema.items())
    else:
        body = f"{frame.height}|{len(frame.columns)}|{frame.hash_rows(seed=0).sum()}"
    return hashlib.sha256(body.encode()).hexdigest()[:DIGEST_LEN]


@contextmanager
def writes(path: str, *, why: str,
           terminal: bool = False,
           scope: str | tuple[str, ...] | None = None,
           fp: str | Callable = "data",
           versions: tuple[str, ...] = ("data",),
           policy: str = "tracked",
           part: dict[str, str] | None = None) -> Iterator[Path]:
    """Declare an output. Yields the concrete path; stamps it on clean exit.

    terminal  consumed outside this pipeline — an app, a human. Makes "nothing here reads
              it" correct rather than an orphan.
    versions  which configured version axes enter this artifact's id. Bump one and every
              artifact that selected it rebuilds.
    policy    tracked | manual | settled | exempt | clock. See dagio.state.
    fp        how to fingerprint what you wrote. A coarse strategy on a DERIVED artifact
              is a correctness hazard: if the id does not move, downstream wrongly skips.
    """
    _check_why(why, path)
    rel = _paths.render(path, part)
    p = _paths.resolve(rel)
    p.parent.mkdir(parents=True, exist_ok=True)

    yield p

    # Only reached on a clean exit. An exception propagates and nothing is stamped.
    _writes.append(rel)
    spec = Spec(why=why, fp=fp, versions=tuple(versions), policy=policy,
                terminal=terminal, scope=_norm_scope(scope))
    inputs = {k: v for k, v in _reads.items() if k != rel}
    new_id = _state.stamp(rel, spec=spec, inputs=inputs, by=_rec._node(),
                          fp_value=_pending_fp.pop(rel, None))
    _rec.record("io", op="write", rel=rel, why=why, terminal=terminal,
                scope=list(_norm_scope(scope) or ()), part=part or {},
                policy=policy, versions=sorted(versions), id=new_id)


@contextmanager
def updates(path: str, *, why: str,
            terminal: bool = False,
            scope: str | tuple[str, ...] | None = None,
            fp: str | Callable = "data",
            versions: tuple[str, ...] = ("data",),
            policy: str = "tracked",
            part: dict[str, str] | None = None) -> Iterator[Path]:
    """An artifact this stage reads AND writes — an append, a patch, an incremental cache.

    Without this the graph reads a self-edge as a cycle and a second writer as a conflict,
    and the temptation is to allowlist the three subtlest stages out of the validator,
    which defeats it. An update is excluded from its own input set: an artifact cannot be
    its own dependency.
    """
    _check_why(why, path)
    rel = _paths.render(path, part)
    p = _paths.resolve(rel)
    p.parent.mkdir(parents=True, exist_ok=True)

    _rec.record("io", op="read", rel=rel, why=why, optional=True, prior=False,
                scope=list(_norm_scope(scope) or ()), part=part or {}, update=True)

    yield p

    _writes.append(rel)
    spec = Spec(why=why, fp=fp, versions=tuple(versions), policy=policy,
                terminal=terminal, scope=_norm_scope(scope))
    inputs = {k: v for k, v in _reads.items() if k != rel}
    new_id = _state.stamp(rel, spec=spec, inputs=inputs, by=_rec._node(),
                          fp_value=_pending_fp.pop(rel, None))
    _rec.record("io", op="write", rel=rel, why=why, terminal=terminal,
                scope=list(_norm_scope(scope) or ()), part=part or {},
                policy=policy, versions=sorted(versions), id=new_id, update=True)


def declared_reads() -> dict[str, object]:
    """What this process has read so far, as `{rel: fp strategy}`."""
    return dict(_reads)


def reset() -> None:
    """Forget this process's reads and writes. Tests only."""
    _reads.clear()
    _writes.clear()
    _pending_fp.clear()
