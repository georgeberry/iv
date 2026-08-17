"""`Invalidator` — the one object. Everything is a method on it.

    from invalidator import Invalidator

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
import functools
import hashlib
import inspect
import os
import re
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
    force         rebuild regardless. Also picked up from $INVALIDATOR_FORCE.
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
                 roots: Sequence[str] = ("raw/",),
                 project_root: str | os.PathLike | None = None,
                 trace: str | os.PathLike | None = None,
                 state_rel: str = ".invalidator/state.json",
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
        # `invalidator status` could never see a bump that a run would see.
        self.versions = dict(versions or {})
        self.source_dirs = tuple(source_dirs)
        self.stages = tuple(stages) if stages is not None else None
        self.order_from = Path(order_from) if order_from else None
        self.roots = tuple(roots)
        self.state_rel = state_rel
        self.state_path_override = _paths.mkpath(state_path, self.project_root) \
            if state_path is not None else None
        self.trace_path = Path(trace) if trace else _env_trace()
        self.force = _env_force() if force is None else force

        self.state = State(self)
        self._reads: dict[str, object] = {}          # rel -> fp strategy, this process
        self._writes: list[str] = []
        self._pending_fp: dict[str, str] = {}
        self._trace_fh = None
        self._depth = 0                              # bookkeeping suppression

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

    @contextmanager
    def bookkeeping(self):
        """I/O in this scope is the pipeline inspecting itself, not a data edge.

        Re-entrant. Computing whether an artifact is stale means reading its inputs, and
        recorded plainly that makes every root feed an input of every guarded stage —
        including the ones that never touch it. Being guarded is not the same as
        depending on the data.
        """
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def record(self, kind: str, **fields) -> None:
        from . import record as _rec
        _rec.emit(self, kind, **fields)

    # ── reads and writes ──────────────────────────────────────────────────────

    def reads(self, path: str, *, why: str,
              optional: bool = False,
              prior: bool = False,
              fp: str | Callable = "data",
              part: dict[str, str] | None = None) -> Path:
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

        # Reading something NEW after this process has already written is a smell: it
        # cannot be among that artifact's inputs, so if it was meant to feed it, the code
        # is in the wrong order. Reading back what this process itself wrote is not that.
        if self._writes and rel not in self._writes:
            print(f"  invalidator: {rel} is read AFTER this process wrote "
                  f"{self._writes[-1]} — it is not among that artifact's inputs.")

        if not p.exists() and not optional:
            raise FileNotFoundError(
                f"{rel} is required by this stage and is not on disk. Its producer has "
                f"not run, or has failed. Pass optional=True if absence is meant to "
                f"degrade rather than fail.")

        self._reads[rel] = fp
        self.record("io", op="read", rel=rel, why=why, optional=optional, prior=prior,
                    part=part or {})
        return p

    @contextmanager
    def writes(self, path: str, *, why: str,
               terminal: bool = False,
               fp: str | Callable = "data",
               policy: str = "tracked",
               part: dict[str, str] | None = None,
               version: str = "",
               allow_missing: bool = False,
               code: str = "") -> Iterator[Path]:
        """Declare an output. Yields the concrete path; stamps it on clean exit.

        terminal  consumed outside this pipeline — an app, a human. Makes "nothing here
                  reads it" correct rather than an orphan.
        policy    tracked | manual | settled | exempt | clock. See invalidator.state.
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
                                  fp_value=self._pending_fp.pop(rel, None))
        self.record("io", op="write", rel=rel, why=why, terminal=terminal,
                    part=part or {}, policy=policy, id=new_id)

    @contextmanager
    def updates(self, path: str, *, why: str,
                terminal: bool = False,
                fp: str | Callable = "data",
                policy: str = "tracked",
                part: dict[str, str] | None = None,
                version: str = "",
                code: str = "") -> Iterator[Path]:
        """An artifact this stage reads AND writes — an append, a patch, a cache.

        Without this the graph reads a self-edge as a cycle and a second writer as a
        conflict, and the temptation is to allowlist the subtlest stages out of the
        validator, which defeats it. An update is excluded from its own input set: an
        artifact cannot be its own dependency.
        """
        _check_why(why, path)
        rel = _paths.render(path, part)
        p = self.resolve_out(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.record("io", op="read", rel=rel, why=why, optional=True, prior=False,
                    part=part or {}, update=True)

        yield p

        self._writes.append(rel)
        spec = Spec(why=why, fp=fp, policy=policy, terminal=terminal, code=code,
                    version=self.version_value(version))
        inputs = {k: v for k, v in self._reads.items() if k != rel}
        new_id = self.state.stamp(rel, spec=spec, inputs=inputs, by=self.node(),
                                  fp_value=self._pending_fp.pop(rel, None))
        self.record("io", op="write", rel=rel, why=why, terminal=terminal,
                    part=part or {}, policy=policy, id=new_id, update=True)

    def external(self, name: str, *, why: str) -> None:
        """Declare that this stage pulls from something outside the pipeline.

        A fetcher's real input is an API, a scraped page, a vendor drop — something with
        no path and no fingerprint. Declaring it puts the provenance in the graph instead
        of leaving the fetcher looking like an artifact that appeared from nowhere.

        It carries no id, so it cannot make anything stale. That is why a fetcher must not
        be guarded on its output — see the GUARDED FETCH check in invalidator.graph.
        """
        _check_why(why, name)
        self.record("io", op="external", rel=f"external:{name}", why=why)

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
             code: bool = False,
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

        code=True folds a hash of the function's own source into the id, so editing the
        transform rebuilds it with no version bump. It is off by default because the hash
        is SHALLOW — it sees this function and not the helpers it calls, so a change can
        still slip past. `data_version` is the honest blunt instrument.
        """
        rels = [output] if isinstance(output, str) else list(output)
        if not rels:
            raise DeclError("step() needs at least one output path")

        def decorate(fn: Callable) -> Callable:
            code_key = source_digest(fn) if code else ""

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                def run():
                    outer, outer_w = dict(self._reads), list(self._writes)
                    # This step's reads AND writes are its own. Without scoping the
                    # writes, a second step reading in the same process trips the
                    # read-after-write warning against the FIRST step's output, which is
                    # true and entirely irrelevant.
                    self._reads.clear()
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

            wrapper.__invalidator_step__ = tuple(rels)
            return wrapper

        return decorate

    # ── the guard ─────────────────────────────────────────────────────────────

    def why_stale(self, rel: str) -> str | None:
        """One line saying why `rel` needs rebuilding, or None if it does not."""
        return self.state.why_stale(rel, self.declared_inputs(rel), self.code_hash(rel),
                                    self.declared_version(rel))

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

        From the static scan, so the decorator and `invalidator status` compute it the
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

        From the static scan, so the decorator and `invalidator status` compute it the
        same way and cannot disagree. None when this artifact does not track its code, or
        when the scan cannot answer.
        """
        try:
            from .static import code_hash_for_artifact
            return code_hash_for_artifact(self, rel)
        except Exception:
            return None

    def is_current(self, paths: str | Sequence[str]) -> bool:
        rels = [paths] if isinstance(paths, str) else list(paths)
        return all(self.why_stale(r) is None for r in rels)

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

        Without either, `invalidator check` still finds a cycle, but it cannot find a
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
        override = os.environ.get("INVALIDATOR_STAGE")
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
            f"artifact is for. It is what `invalidator stage` prints; there is nowhere "
            f"else for it to live.")
    return why


def source_digest(fn: Callable) -> str:
    """A hash of what a function DOES, insensitive to how it is spelled.

    `ast.unparse` normalises whitespace, comments and formatting away, and the decorators
    are stripped — so reformatting, or editing a `why=`, does not invalidate data, while a
    real change to the logic does. Raw-source hashing (joblib, redun) churns on all three.

    Still shallow: it sees THIS function, not the helpers it calls.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise DeclError(
            f"code=True needs readable source for {getattr(fn, '__name__', fn)!r}, and "
            f"there is none (a REPL, or a C function). Drop code=True.") from e
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
            frame.f_globals.get("__name__", "").split(".")[0] == "invalidator":
        frame = frame.f_back
    file = frame.f_globals.get("__file__") if frame else None
    here = Path(file).resolve().parent if file else Path.cwd()
    for d in (here, *here.parents):
        if any((d / m).exists() for m in ("pyproject.toml", ".git", "setup.py")):
            return d
    return here


def _env_trace() -> Path | None:
    dest = os.environ.get("INVALIDATOR_TRACE")
    return Path(dest) if dest else None


def _env_force() -> bool:
    return os.environ.get("INVALIDATOR_FORCE", "").lower() in ("1", "true", "yes")
