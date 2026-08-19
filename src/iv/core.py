"""`Invalidator` — the one object. Everything is a method on it.

    from iv import Invalidator

    iv = Invalidator(data_root="gs://wvorp-state/data", data_version="wnba-3.07")

    @iv.step("processed/box_features.parquet",
             why="per-(season, player) box prior for the xPM fit")
    def build(out):
        poss = pl.scan_parquet(iv.reads("processed/possessions.parquet",
                                        why="lineup possessions; the minutes denominator"))
        ...
        frame.write_parquet(out)

    build()          # runs, or skips because nothing upstream moved

NO GLOBAL STATE, ON PURPOSE. The configuration is constructor arguments, not a TOML file
discovered by walking up from the working directory and not environment variables. Two
pipelines in one process are two `Invalidator`s; a test is an `Invalidator` pointed at a
temp directory. There is no "which config is live right now" question to get wrong, and
nothing to keep in sync with the code.

`data_version` is the global escape hatch. It enters EVERY artifact's id, so bumping it
rebuilds the world. That is what covers the one thing no fingerprint of the inputs can
see: a builder whose logic changed. It lives here, next to the data root, because that is
where a project already keeps its version.
"""
from __future__ import annotations

import ast
import atexit
import functools
import hashlib
import inspect
import os
import re
import threading as _threading
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Iterator, Sequence

from . import fingerprint as _fp
from . import paths as _paths
from .errors import ConfigError, DeclError
from .state import POLICIES, Spec, State

# How a pipeline script names a stage: `python x.py`, `uv run python x.py`, `python -m …`.
_INVOKE = re.compile(r"(?:python[0-9.]*|uv run python|python -m)\s+(?:-\S+\s+)*"
                     r"([\w./-]+\.py)")


def _pl_col(name: str):
    """`pl.col(name)`, imported late — polars is an optional dependency of this package."""
    import polars as pl
    return pl.col(name)


class Invalidator:
    """One pipeline: where its data lives, what version it is, and how to touch it.

    data_root     where artifacts live. A path, or a `gs://`/`s3://` URI (needs
                  cloudpathlib). Artifacts are named relative to it, so the ids survive
                  moving the data somewhere else.
    data_version  a single string in every artifact's id. Bump it and everything rebuilds.
    source_dirs   what the static scan walks to build the graph and to answer "what does
                  this stage read NOW".
    project_root  defaults to the directory containing the file that constructed this.
    trace         where to append the runtime trace, or None for off.
    force         rebuild regardless. Also picked up from $IV_FORCE.
    """

    def __init__(self, *,
                 data_root: str | os.PathLike,
                 data_version: str,
                 versions: dict[str, str] | None = None,
                 out_root: str | os.PathLike | None = None,
                 overlay: bool = True,
                 state_path: str | os.PathLike | None = None,
                 source_dirs: Sequence[str] = ("src", "scripts", "stages"),
                 stages: Sequence[str] | None = None,
                 order_from: str | os.PathLike | None = None,
                 slices: dict[str, Callable] | None = None,
                 roots: Sequence[str] = ("raw/",),
                 project_root: str | os.PathLike | None = None,
                 trace: str | os.PathLike | None = None,
                 state_rel: str = ".iv/state",
                 force: bool | None = None) -> None:
        if not data_version or not isinstance(data_version, str):
            raise ConfigError(
                "data_version is required and must be a non-empty string. It is the one "
                "thing that can invalidate everything at once, which is what covers a "
                "builder whose logic changed.")
        self.project_root = Path(project_root) if project_root is not None \
            else _caller_root()
        self.data_root = _paths.mkpath(data_root, self.project_root)
        # Writes land in out_root; reads fall back to data_root. Same split wvorp's
        # DATA_BASE/DATA_OUT_BASE exists for, and for the same reason: a local run has to
        # be able to rebuild three stages without clobbering the tree everything else
        # reads. Defaults to data_root, so a project that does not want it never sees it.
        self.out_root = _paths.mkpath(out_root, self.project_root) \
            if out_root is not None else self.data_root
        # With `overlay`, a read prefers whatever this run has already built — usually
        # what you want when testing a chain locally. Without it, reads ALWAYS come from
        # data_root and only writes go to out_root, which is what a project does when the
        # shared tree is the definition of the inputs and local output is a side effect.
        self.overlay = overlay
        self.data_version = data_version
        # Extra versions a step can opt into by NAME. The name is the literal at the call
        # site so the static scan can read it; the value lives here, next to data_version,
        # where a project already keeps such things. Without this indirection
        # `version=MODEL_VERSION` would be a name the scanner cannot resolve, and
        # `iv status` could never see a bump that a run would see.
        self.versions = dict(versions or {})
        self.source_dirs = tuple(source_dirs)
        # `{label: predicate}`. A slice says an artifact is not one population — which
        # ROWS a site touches. The label is what the graph reads; the predicate is what
        # makes the label true rather than merely claimed, by doing the filtering itself.
        self.slices = dict(slices or {})
        self.stages = tuple(stages) if stages is not None else None
        self.order_from = Path(order_from) if order_from else None
        self.roots = tuple(roots)
        # A DIRECTORY, one JSON per artifact — so two stages stamping at once cannot
        # erase each other. A `.json` here (or in `state_path`) names the old single file
        # and is read as the directory beside it; see State.dir.
        self.state_rel = state_rel
        self.state_path_override = _paths.mkpath(state_path, self.project_root) \
            if state_path is not None else None
        self.trace_path = Path(trace) if trace else _env_trace()
        self.force = _env_force() if force is None else force

        self.state = State(self)
        self._reads: dict[str, object] = {}          # rel -> fp strategy, this process
        self._writes: list[str] = []
        # (input, artifact) pairs where the input was read after that artifact was
        # written. Judged at EXIT, not here — see `_report_read_after_write`.
        # Inputs read with `prior=True`. Kept apart because they are recorded for
        # lineage but must NOT be compared for staleness — see `writes`.
        self._prior_reads: set[str] = set()
        # rel -> the slice label it was read under. "" means the whole artifact, which is
        # also what two DIFFERENT slices of one artifact collapse to: together they are
        # not a slice of anything.
        self._read_slices: dict[str, str] = {}
        # Set by `PartitionCache.building`: {input rel: (period column, bound)}. While it
        # holds, `frame()` on one of those inputs returns only rows at or before the
        # bound — the past-only invariant, enforced rather than intended.
        self._period_bound: dict[str, tuple[str, str]] = {}
        self._late_reads: list[tuple[str, str]] = []
        self._late_hooked = False
        self._pending_fp: dict[str, str] = {}
        # Reads made before any step ran — a module-level `SRC = iv.reads(...)`. They
        # belong to EVERY step in the file, so a step inherits them rather than starting
        # from nothing. Captured once, so step B does not also inherit step A's reads.
        self._module_reads: dict[str, object] | None = None
        # Concrete paths handed out by `reads()`. With overlay=False these point at the
        # SHARED tree, so a stray write through one lands in prod even though every
        # declared write was redirected to out_root. That is how a migration script's
        # variable-name collision overwrote a live parquet with a JSON dump: the write
        # went through a path that had been resolved for READING.
        self._read_paths: set[str] = set()
        self._trace_fh = None
        # THREAD-LOCAL, because "the pipeline is inspecting itself" is a property of the
        # CALL STACK, not of the process. A stage that loads seven feeds concurrently has
        # one thread globbing inside `bookkeeping()` while another declares a read, and a
        # shared counter makes the second one invisible — six of seven declared inputs
        # silently vanished from the record that way.
        self._local = _threading.local()

    def __repr__(self) -> str:
        split = "" if self.out_root is self.data_root \
            else f", out_root={str(self.out_root)!r}"
        return (f"Invalidator(data_root={str(self.data_root)!r}{split}, "
                f"data_version={self.data_version!r})")

    # ── paths ─────────────────────────────────────────────────────────────────

    def resolve(self, rel: str):
        """A rendered rel path -> the concrete path to READ.

        With a read/write split this is an OVERLAY: whatever you have built locally
        shadows the shared tree, so a partial local rebuild reads its own outputs and
        falls back to prod for everything it did not touch. Without a split (the default)
        both roots are the same and this is just `data_root / rel`.
        """
        if self.overlay and self.out_root is not self.data_root:
            out = _paths.resolve_under(self.out_root, rel)
            with self.bookkeeping():
                if out.exists():
                    return out
        return _paths.resolve_under(self.data_root, rel)

    def resolve_out(self, rel: str):
        """Where a WRITE goes, and where an artifact's own existence is judged."""
        return _paths.resolve_under(self.out_root, rel)

    def to_rel(self, path) -> str | None:
        return _paths.to_rel(self, path)

    # ── the trace ─────────────────────────────────────────────────────────────

    @property
    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @contextmanager
    def bookkeeping(self):
        """I/O in this scope is the pipeline inspecting itself, not a data edge.

        Re-entrant. Computing whether an artifact is stale means reading its inputs, and
        recorded plainly that makes every root feed an input of every guarded stage —
        including the ones that never touch it. Being guarded is not the same as
        depending on the data.
        """
        self._local.depth = self._depth + 1
        try:
            yield
        finally:
            self._local.depth -= 1

    def record(self, kind: str, **fields) -> None:
        from . import record as _rec
        _rec.emit(self, kind, **fields)

    # ── read-after-write, judged at the end ───────────────────────────────────

    def _note_late_read(self, rel: str, after: str) -> None:
        """Remember that `rel` was read after `after` was written. Do not judge yet.

        Reading something new after a write is only a bug if that write was the LAST one.
        An `iv.updates` artifact built incrementally — the rollforward fits one week,
        saves, fits the next — writes many times, and every read in week two follows week
        one's write. Warning there stated a falsehood: it named six inputs as "not among
        that artifact's inputs" while all six were on the final stamp, because the next
        write picked them up. A diagnostic that is usually wrong is worse than none; it
        teaches you to skip the whole block.

        So the question is deferred to exit, where the answer is a FACT — the stored
        record either lists the input or it does not.
        """
        self._late_reads.append((rel, after))
        if not self._late_hooked:
            self._late_hooked = True
            atexit.register(self._report_read_after_write)

    def _report_read_after_write(self) -> None:
        """At exit: report only the late reads the final stamp really did miss."""
        seen, missed = set(), []
        for rel, art in self._late_reads:
            if (rel, art) in seen:
                continue
            seen.add((rel, art))
            rec = self.state.record_of(art)
            if rec is not None and rel not in (rec.get("in") or {}):
                missed.append((rel, art))
        for rel, art in missed:
            print(f"  iv: {art} was written before {rel} was read, and its "
                  f"stored inputs do not list {rel}. If it feeds that artifact, read it "
                  f"first; if it does not, this stage writes two unrelated things.")

    # ── reads and writes ──────────────────────────────────────────────────────

    def reads(self, path: str, *, why: str,
              optional: bool = False,
              prior: bool = False,
              fp: str | Callable = "data",
              part: dict[str, str] | None = None,
              slice: str = "") -> Path:
        """Declare and resolve an input. Returns the concrete path.

        optional  an absent input degrades a feature rather than failing the stage.
                  Without it, a missing input raises here, at the line that wanted it.
        prior     this deliberately reads the PREVIOUS run's copy, so a producer that runs
                  later in the same pipeline is correct rather than an ordering bug.
        fp        how to fingerprint it IF it is a root. Ignored for anything the pipeline
                  writes, whose id is already stamped.
        part      the partition values for a `{template}` path.
        """
        _check_why(why, path)
        rel = _paths.render(path, part)
        p = self.resolve(rel)

        if self._depth:
            # Inside `bookkeeping()`: the pipeline is inspecting ITSELF, so this is not a
            # data edge and must not become one. Suppressing only the trace and still
            # registering the input is worse than not suppressing at all — the artifact
            # gains a dependency on a file it never used, and the trace no longer shows
            # where it came from.
            return p

        if self._writes and rel not in self._writes:
            self._note_late_read(rel, self._writes[-1])

        if not p.exists() and not optional:
            raise FileNotFoundError(
                f"{rel} is required by this stage and is not on disk. Its producer has "
                f"not run, or has failed. Pass optional=True if absence is meant to "
                f"degrade rather than fail.")

        self._reads[rel] = fp
        if rel in self._read_slices and self._read_slices[rel] != slice:
            self._read_slices[rel] = ""      # two slices read: that is the whole thing
        else:
            self._read_slices[rel] = slice
        if prior:
            self._prior_reads.add(rel)
        self._read_paths.add(str(p))
        self.record("io", op="read", rel=rel, why=why, optional=optional, prior=prior,
                    part=part or {}, slice=slice)
        return p

    def frame(self, path: str, *, why: str,
              optional: bool = False,
              prior: bool = False,
              fp: str | Callable = "data",
              part: dict[str, str] | None = None,
              slice: str = "",
              **read_kwargs):
        """`reads()`, and give me the frame. None if `optional` and it is not there.

        Every read is otherwise spelled `pl.read_parquet(iv.reads(...))`, twice as wide
        as the fact it states. This exists so the top of a stage reads as its I/O
        contract — a list of what it takes and why — with the loading incidental.

        `**read_kwargs` goes to the reader, so `columns=` still works.
        """
        p = self.reads(path, why=why, optional=optional, prior=prior, fp=fp, part=part,
                       slice=slice)
        if optional and not p.exists():
            return None
        from .fingerprint import read_frame
        df = read_frame(p, **read_kwargs)
        if slice:
            # The label and the filter are the SAME fact, so the label does the filtering.
            # Stated and not applied, a slice is a comment — and a comment is what this
            # replaced: `game_predictions` was read whole and filtered by hand, with the
            # reason those rows were the safe ones written above it in prose.
            if slice not in self.slices:
                raise DeclError(
                    f"{path} declares slice={slice!r}, which this Invalidator does not "
                    f"define. Pass slices={{{slice!r}: <predicate>}} so the label filters "
                    f"rather than merely asserts. Known: {sorted(self.slices) or 'none'}")
            df = self.slices[slice](df)
        # THE PAST-ONLY INVARIANT. Inside `PartitionCache.building(T)` a period-scoped
        # input is cut to its bound, so a stage building period T physically cannot see
        # a later one. Filtered, not asserted, for the same reason `slices` filters: a
        # rule the caller has to remember is a rule that gets forgotten.
        bound = self._period_bound.get(_paths.render(path, part))
        if bound is not None:
            col, upto = bound
            if col not in df.columns:
                raise DeclError(
                    f"{path} is declared period-scoped on {col!r} and the frame has no "
                    f"such column. Columns: {sorted(df.columns)[:8]}")
            df = df.filter(_pl_col(col) <= upto)
        return df

    def collection(self, path: str, *, why: str, optional: bool = False,
                   prior: bool = False,
                   fp: str | Callable = "rows",
                   part: dict[str, str] | None = None) -> list[Path]:
        """Declare a whole feed at once and return the files it currently holds.

        A collection is the read whose unit is the GROWING SET rather than a named file:
        the seasons on disk are DISCOVERED, not known, so there is no `part` to pass for
        the free fields. `reads()` cannot express that — it renders one concrete path, and
        would have to be told the answer it is being asked for.

        Its id folds every member's id, so a new season moves it and a refetch that
        changed nothing does not. `fp` defaults to `"rows"` rather than the full data hash
        because a collection of ROOTS is fingerprinted on every check, and a whole feed is
        the expensive place to ask an expensive question.

        `prior` means the same as it does on `reads()`, and a collection reaches it just
        as easily: a stage that runs before the fetchers reads the PREVIOUS run's copies
        of a whole feed, by design rather than by an ordering mistake.
        """
        _check_why(why, path)
        rel = _paths.render_partial(path, part)
        if not _paths.fields(rel):
            raise DeclError(
                f"{path!r} has no free field left, so it names one file rather than a "
                f"collection. Use reads().")
        # Read BEFORE the glob: `instances_of` enters `bookkeeping()` itself, and asking
        # afterwards would answer about this call rather than about its caller.
        inspecting = bool(self._depth)
        files = [self.resolve(r) for r in self.state.instances_of(rel)]
        if inspecting:
            return files
        if not files and not optional:
            raise FileNotFoundError(
                f"{rel} matches nothing on disk. Its producer has not run, or has "
                f"failed. Pass optional=True if absence is meant to degrade rather "
                f"than fail.")
        self._reads[rel] = fp
        if prior:
            # `prior` was accepted here, documented as meaning what it means on
            # `reads()`, and then never recorded — so a collection read from the
            # PREVIOUS run was still compared against this one. wvorp's
            # `rookie_prior` and `rookie_projections` read the roster snapshots
            # that `fetch_rosters` rewrites later in the same refresh, and came
            # out of every run stale for it.
            self._prior_reads.add(rel)
        self._read_paths.update(str(f) for f in files)
        self.record("io", op="read", rel=rel, why=why, optional=optional, prior=prior,
                    collection=True, part=part or {})
        return files

    @contextmanager
    def writes(self, path: str, *, why: str,
               terminal: bool = False,
               fp: str | Callable = "data",
               policy: str = "tracked",
               part: dict[str, str] | None = None,
               version: str = "",
               allow_missing: bool = False,
               slice: str = "",
               code: str = "") -> Iterator[Path]:
        """Declare an output. Yields the concrete path; stamps it on clean exit.

        terminal  consumed outside this pipeline — an app, a human. Makes "nothing here
                  reads it" correct rather than an orphan.
        policy    tracked | manual | settled | clock. See iv.state.
        fp        how to fingerprint what you wrote. A coarse strategy on a DERIVED
                  artifact is a correctness hazard: if the id does not move, everything
                  downstream wrongly skips.
        version   the NAME of one of the Invalidator's `versions`, folded into THIS
                  artifact's id on top of data_version. For something beyond the data
                  that governs it: a model version, a vendor API version, a hand-tuned
                  table. Only the artifacts that name it move when it changes — which is
                  the point, since a model bump must not rebuild a feature pipeline it
                  cannot have affected.
        allow_missing  the builder may legitimately produce nothing — a projection with no
                  season to project yet, a roster that does not exist until July. Then
                  there is nothing to stamp, and the artifact stays stale so the next run
                  tries again. Without it, not writing is an error.
        code      an opaque string folded into the id — `step(code=True)` puts the
                  function's normalised source here.
        """
        _check_why(why, path)
        rel = _paths.render(path, part)
        p = self.resolve_out(rel)
        p.parent.mkdir(parents=True, exist_ok=True)

        yield p

        # Only reached on a clean exit. An exception propagates and nothing is stamped,
        # so a half-written artifact reads as stale rather than as fresh.
        self._writes.append(rel)
        if allow_missing:
            with self.bookkeeping():
                wrote = p.exists()
            if not wrote:
                # NOT an error. The guard cannot tell "should have written and did not"
                # from "correctly had nothing to write", so it refuses to stamp — a stamp
                # means THIS code produced THIS file — and says so. The artifact stays
                # stale and the next run tries again.
                print(f"  {rel}: nothing produced — not stamped")
                self._pending_fp.pop(rel, None)
                return
        spec = Spec(why=why, fp=fp, policy=policy, terminal=terminal, code=code,
                    version=self.version_value(version))
        inputs = {k: v for k, v in self._reads.items() if k != rel}
        new_id = self.state.stamp(rel, spec=spec, inputs=inputs, by=self.node(),
                                  prior=self._prior_reads & set(inputs),
                                  slices={k: v for k, v in self._read_slices.items()
                                          if v and k in inputs},
                                  fp_value=self._pending_fp.pop(rel, None))
        self.record("io", op="write", rel=rel, why=why, terminal=terminal,
                    part=part or {}, policy=policy, id=new_id, slice=slice)

    @contextmanager
    def updates(self, path: str, *, why: str,
                terminal: bool = False,
                fp: str | Callable = "data",
                policy: str = "tracked",
                part: dict[str, str] | None = None,
                version: str = "",
                allow_missing: bool = False,
                slice: str = "",
                code: str = "") -> Iterator[Path]:
        """An artifact this stage reads AND writes — an append, a patch, a cache.

        Without this the graph reads a self-edge as a cycle and a second writer as a
        conflict, and the temptation is to allowlist the subtlest stages out of the
        validator, which defeats it. An update is excluded from its own input set: an
        artifact cannot be its own dependency.

        A CHAIN needs nothing extra. Several stages writing one artifact each CONTRIBUTE
        their own reads to its record — see `State.stamp` — so the head of a chain stays
        in the identity after the tail has rewritten the file, without this stage having
        to name it.

        `allow_missing` means the same as it does on `writes`, and an update reaches it
        by a route a write cannot: a `--list-only` or `--dry-run` path that returns from
        inside the block, on a first run where there is no file yet.
        """
        _check_why(why, path)
        rel = _paths.render(path, part)
        p = self.resolve_out(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.record("io", op="read", rel=rel, why=why, optional=True, prior=False,
                    part=part or {}, update=True, slice=slice)

        yield p

        self._writes.append(rel)
        if allow_missing:
            with self.bookkeeping():
                if not p.exists():
                    print(f"  {rel}: nothing produced — not stamped")
                    self._pending_fp.pop(rel, None)
                    return
        spec = Spec(why=why, fp=fp, policy=policy, terminal=terminal, code=code,
                    version=self.version_value(version))
        inputs = {k: v for k, v in self._reads.items() if k != rel}
        new_id = self.state.stamp(rel, spec=spec, inputs=inputs, by=self.node(),
                                  prior=self._prior_reads & set(inputs),
                                  slices={k: v for k, v in self._read_slices.items()
                                          if v and k in inputs},
                                  fp_value=self._pending_fp.pop(rel, None))
        self.record("io", op="write", rel=rel, why=why, terminal=terminal,
                    part=part or {}, policy=policy, id=new_id, update=True, slice=slice)

    def external(self, name: str, *, why: str) -> None:
        """Declare that this stage pulls from something outside the pipeline.

        A fetcher's real input is an API, a scraped page, a vendor drop — something with
        no path and no fingerprint. Declaring it puts the provenance in the graph instead
        of leaving the fetcher looking like an artifact that appeared from nowhere.

        It carries no id, so it cannot make anything stale. That is why a fetcher must not
        be guarded on its output — see the GUARDED FETCH check in iv.graph.
        """
        _check_why(why, name)
        self.record("io", op="external", rel=f"external:{name}", why=why)

    def guard_writes(self) -> None:
        """Make a write through a path that was resolved for READING raise.

        `reads()` hands back a real path into the shared tree. Nothing stops
        `p.write_text(...)` on it, and then no amount of out_root redirection helps —
        the declared writes were all redirected and this one was not declared at all.

        Opt-in because it patches `Path.write_*`, and a process may have perfectly good
        reasons to write elsewhere. It only ever objects to a path THIS Invalidator
        handed out as an input.
        """
        from pathlib import Path as _P
        if getattr(self, "_write_guard_on", False):
            return
        self._write_guard_on = True
        iv = self

        owners = [_P]
        try:
            # The one that matters in practice. A pipeline on a bucket hands out
            # CloudPaths, not pathlib.Paths — guarding only the latter guards nothing
            # where the damage would actually be done.
            from cloudpathlib import CloudPath
            owners.append(CloudPath)
        except ImportError:
            pass

        def wrap(owner, name):
            orig = getattr(owner, name, None)
            if orig is None:
                return

            def patched(self, *a, **kw):
                # `open` is only a write when the MODE says so. Fingerprinting an input
                # opens it to hash it, and blocking that blocks the check itself.
                if name == "open":
                    mode = kw.get("mode", a[0] if a else "r")
                    if not any(c in str(mode) for c in "wax+"):
                        return orig(self, *a, **kw)
                if str(self) in iv._read_paths:
                    raise DeclError(
                        f"refusing to write {self}: that path was handed out by "
                        f"iv.reads(), so it is an INPUT. Writing through it bypasses "
                        f"out_root and lands in the shared tree. Declare the output with "
                        f"iv.writes()/@iv.step and write to the path it yields.")
                return orig(self, *a, **kw)

            patched.__name__ = name
            setattr(owner, name, patched)

        for owner in owners:
            for n in ("write_text", "write_bytes", "open"):
                wrap(owner, n)

    def stamp_content(self, path, frame) -> None:
        """Fingerprint from a frame already in memory instead of re-reading the file.

        `writes()` yields a plain path so you can write through any library, which means
        it has no way to see what you wrote except by reading it back. When you still hold
        the frame, hand it over and skip the pass.
        """
        rel = self.to_rel(path)
        if rel is None:
            raise DeclError(f"{path} is outside the data tree; nothing to stamp")
        if not hasattr(frame, "hash_rows"):
            raise DeclError(
                f"stamp_content needs a frame with .hash_rows() (a polars DataFrame); "
                f"got {type(frame).__name__}")
        self._pending_fp[rel] = _fp.frame_digest(frame)

    # ── the step decorator ────────────────────────────────────────────────────

    def step(self, output: str | Sequence[str], *, why: str,
             code: bool | None = None,
             terminal: bool = False,
             fp: str | Callable = "data",
             policy: str = "tracked",
             part: dict[str, str] | None = None,
             version: str = "",
             allow_missing: bool = False,
             if_needed: bool = True) -> Callable:
        """Make a function into a guarded step. The normal way to write one.

            @iv.step("processed/box.parquet", why="the box prior for the xPM fit")
            def build(out):
                ...
                frame.write_parquet(out)

            build()      # runs, or skips and returns False

        The decorated function receives one concrete path per artifact, in order, then
        whatever you pass at the call site. It returns True if it ran.

        WHY THIS AND NOT A `with` BLOCK: a context manager cannot decline to run its body
        — PEP 377 proposed exactly that and was rejected. A callable is the only thing you
        can choose not to call, which is why every system in this category wraps the unit
        of work in a function.

        The body is also a natural SCOPE, which is what makes the input set per-artifact
        rather than per-file: reads inside the function belong to this step's outputs and
        to nothing else.

`code` (ON by default) folds a hash of the function's own source into the id, so
        editing the transform rebuilds it with no version bump.

        It belongs in the id because it is a DECLARATION — a fact about the code as
        written — not a prediction about what will execute. That distinction is the whole
        design: `ast.unparse` of this function is computed the same deterministic way at
        run time and by the CLI, so it is a hash compared against a hash. Asking the same
        parser to predict which reads will happen is a different kind of question, and
        answering it wrongly is silent.

        The hash is over the PARSED tree, so reformatting and comments are free. It is
        SHALLOW — it sees this function and the decorators are stripped, but not the
        helpers it calls — so a change can still slip past. `data_version` remains the
        blunt instrument; `code=False` opts out.
        """
        rels = [output] if isinstance(output, str) else list(output)
        if not rels:
            raise DeclError("step() needs at least one output path")

        def decorate(fn: Callable) -> Callable:
            # `code=None` is the default: hash the source if it can be read, and carry on
            # quietly if it cannot. A default must not break code that worked — a notebook
            # or a REPL has no readable source, and refusing to run there would be a
            # strange price for an optimisation. `code=True` is the explicit ask, and that
            # one raises rather than silently giving less than was asked for.
            code_key = "" if code is False else source_digest(fn, required=bool(code))

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                def run():
                    if self._module_reads is None:
                        self._module_reads = dict(self._reads)
                    outer, outer_w = dict(self._reads), list(self._writes)
                    # This step's reads AND writes are its own, plus whatever the module
                    # read before any step started. Without scoping the writes, a second
                    # step reading in the same process trips the read-after-write warning
                    # against the FIRST step's output, which is true and irrelevant.
                    self._reads.clear()
                    self._reads.update(self._module_reads)
                    self._writes.clear()
                    try:
                        with ExitStack() as stack:
                            outs = [stack.enter_context(
                                self.writes(r, why=why, terminal=terminal, fp=fp,
                                            policy=policy, part=part, code=code_key,
                                            version=version,
                                            allow_missing=allow_missing))
                                for r in rels]
                            return fn(*outs, *args, **kwargs)
                    finally:
                        self._reads.update(outer)
                        self._writes[:] = outer_w + self._writes

                return self.build_if_needed(rels, run, if_needed=if_needed)

            wrapper.__iv_step__ = tuple(rels)
            return wrapper

        return decorate

    # ── the guard ─────────────────────────────────────────────────────────────

    def why_stale(self, rel: str, fingerprint: bool = True) -> str | None:
        """One line saying why `rel` needs rebuilding, or None if it does not.

        `fingerprint=False` answers without touching the data: roots are taken on trust,
        so only code, versions, existence and DERIVED inputs are checked. That is the
        cheap question, and the right one when you are asking "what did my edit break?"
        rather than "is the pipeline current?". A live run always fingerprints — there is
        no point guessing when you are about to do the work anyway.
        """
        return self.state.why_stale(rel, self.code_hash(rel),
                                    self.declared_version(rel), fingerprint)

    def version_value(self, name: str) -> str:
        """`"model"` -> `"model:w-v3.98"`. Empty for no extra version.

        An unknown name is an error with no fallback: defaulting would key the artifact on
        a constant, and an artifact keyed on a constant is permanently, silently current.
        """
        if not name:
            return ""
        if name not in self.versions:
            raise ConfigError(
                f"version={name!r} is not one of this Invalidator's versions "
                f"{sorted(self.versions) or '(none)'}. Add it to versions={{...}}.")
        return f"{name}:{self.versions[name]}"

    def declared_version(self, rel: str) -> str | None:
        """The extra version its write site names NOW, resolved through `versions`.

        From the static scan, so the decorator and `iv status` compute it the
        same way. Read it out of the record instead and a bumped model version could never
        be seen, which is the whole point of the option.
        """
        try:
            from .static import version_for_artifact
            name = version_for_artifact(self, rel)
            return None if name is None else self.version_value(name)
        except ConfigError:
            raise
        except Exception:
            return None

    def code_hash(self, rel: str) -> str | None:
        """The CURRENT source hash of the `step(code=True)` function that writes `rel`.

        From the static scan, so the decorator and `iv status` compute it the
        same way and cannot disagree. None when this artifact does not track its code, or
        when the scan cannot answer.
        """
        try:
            from .static import code_hash_for_artifact
            return code_hash_for_artifact(self, rel)
        except Exception:
            return None

    def is_current(self, paths: str | Sequence[str], fingerprint: bool = True) -> bool:
        rels = [paths] if isinstance(paths, str) else list(paths)
        return all(self.why_stale(r, fingerprint) is None for r in rels)

    def declared_inputs(self, rel: str) -> dict[str, object] | None:
        """What the code reads NOW, from the static scan.

        Without it an input that was ADDED to a stage is invisible: the stored record has
        no entry for a path the last build never read, so there is nothing to compare
        against. None when the scan cannot answer, in which case the check falls back to
        the recorded input set.
        """
        try:
            from .static import inputs_for_artifact
            return inputs_for_artifact(self, rel)
        except Exception:
            return None

    def build_if_needed(self, paths: str | Sequence[str], build: Callable[[], object], *,
                        if_needed: bool = True, force: bool | None = None) -> bool:
        """Run `build` unless every path is current. Returns True if it ran.

        Every output is guarded, not just the first: a stage that writes four artifacts
        and is guarded on one leaves the other three to rot.

        This does not stamp anything — `writes()` does, and only on a clean exit. A shell
        wrapper whose stamp did not depend on the build succeeding is how `|| echo` once
        swallowed failures into "current".
        """
        rels = [paths] if isinstance(paths, str) else list(paths)
        forced = self.force if force is None else force
        if forced:
            self.force = True                    # reaches the partition cache too

        if if_needed and not forced:
            reasons = [(r, self.why_stale(r)) for r in rels]
            if all(reason is None for _, reason in reasons):
                for r in rels:
                    print(f"  {r} is current — skipping")
                    # Recorded so `drift` can tell "ran and never read this" from "did
                    # not run at all". Without it every skipped stage reports each of its
                    # declared inputs as unseen, which is the cache working.
                    self.record("skip", rel=r)
                return False
            for r, reason in reasons:
                if reason is not None:
                    print(f"  {r}: {reason}")
                    break

        build()
        return True

    # ── partitions ────────────────────────────────────────────────────────────

    def for_each(self, over, build_one, **kw):
        """Run `build_one(partition)` only for the partitions that moved."""
        from .partition import for_each
        return for_each(self, over, build_one, **kw)

    def partitions(self, output: str, key: str, **kw):
        from .partition import PartitionCache
        return PartitionCache(self, output, key, **kw)

    # ── introspection ─────────────────────────────────────────────────────────

    def record_of(self, rel: str) -> dict | None:
        return self.state.record_of(rel)

    def graph(self):
        from .graph import build
        return build(self)

    def declared_order(self) -> list[str] | None:
        """The order the pipeline actually runs its stages in, if the project says.

            Invalidator(..., stages=["stages/fetch.py", "stages/build.py"])
            Invalidator(..., order_from="refresh.sh")     # scanned from the script

        Without either, `iv check` still finds a cycle, but it cannot find a
        stage that runs BEFORE its producer — there is nothing to compare the graph
        against, so it says so rather than passing quietly.
        """
        if self.stages is not None:
            return list(self.stages)
        if self.order_from is None:
            return None
        script = self.order_from if self.order_from.is_absolute() \
            else self.project_root / self.order_from
        if not script.exists():
            raise ConfigError(f"order_from names {script}, which does not exist")
        order, seen = [], set()
        for line in script.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue                     # a commented-out stage does not run
            for m in _INVOKE.finditer(stripped):
                node = m.group(1)
                if node not in seen:
                    seen.add(node)
                    order.append(node)       # first invocation wins
        return order

    def node(self) -> str:
        """The stage, as a project-relative script path. One process is one stage."""
        import sys
        override = os.environ.get("IV_STAGE")
        if override:
            return override
        argv0 = sys.argv[0] if sys.argv else ""
        if not argv0:
            return "<interactive>"
        p = Path(argv0).resolve()
        try:
            return str(p.relative_to(self.project_root))
        except ValueError:
            return p.name

    def reset(self) -> None:
        """Forget this process's reads, writes and cached state. Tests, and long-lived
        hosts where one process really is many stages."""
        self._reads.clear()
        self._writes.clear()
        self._pending_fp.clear()
        self.state.reset()


# ── helpers ───────────────────────────────────────────────────────────────────

def _check_why(why: object, path: str) -> str:
    if not isinstance(why, str) or not why.strip():
        raise DeclError(
            f"{path!r}: why= is required and must be a non-empty string saying what this "
            f"artifact is for. It is what `iv stage` prints; there is nowhere "
            f"else for it to live.")
    return why


def source_digest(fn: Callable, required: bool = True) -> str:
    """A hash of what a function DOES, insensitive to how it is spelled.

    `ast.unparse` normalises whitespace, comments and formatting away, and the decorators
    are stripped — so reformatting, or editing a `why=`, does not invalidate data, while a
    real change to the logic does. Raw-source hashing (joblib, redun) churns on all three.

    Still shallow: it sees THIS function, not the helpers it calls.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        if required:
            raise DeclError(
                f"code=True needs readable source for {getattr(fn, '__name__', fn)!r}, "
                f"and there is none (a REPL, or a C function). Drop code=True.") from e
        return ""
    import textwrap
    tree = ast.parse(textwrap.dedent(src))
    node = tree.body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.decorator_list = []
    return hashlib.sha256(ast.unparse(node).encode()).hexdigest()[:_fp.DIGEST_LEN]


def _caller_root() -> Path:
    """The directory of the file that constructed the Invalidator, walked up to a project
    marker. Guessing beats requiring the argument in the common case, and `project_root=`
    is there for when the guess is wrong."""
    frame = inspect.currentframe()
    # Walk out of this package rather than counting frames — a fixed count is right
    # exactly once and silently wrong the moment anything wraps the constructor.
    while frame is not None and \
            frame.f_globals.get("__name__", "").split(".")[0] == "iv":
        frame = frame.f_back
    file = frame.f_globals.get("__file__") if frame else None
    here = Path(file).resolve().parent if file else Path.cwd()
    for d in (here, *here.parents):
        if any((d / m).exists() for m in ("pyproject.toml", ".git", "setup.py")):
            return d
    return here


def _env_trace() -> Path | None:
    dest = os.environ.get("IV_TRACE")
    return Path(dest) if dest else None


def _env_force() -> bool:
    return os.environ.get("IV_FORCE", "").lower() in ("1", "true", "yes")
