"""The pipeline, and the six things a stage says.

    iv = Pipeline(root="gs://bucket/data", source_dirs=["scripts", "src"])

    iv.constants("config/model/", why="what the fits answer to", version="w-v3.100")

    @iv.step("processed/box_features/", why="the per-(season, player) box matrix")
    def build(out):
        iv.reads("config/model/", why="a model change must rebuild this")
        poss = pl.read_parquet(iv.reads(
            "processed/possessions/", why="one row per possession, with lineups"))
        features(poss).write_parquet(out)

`constants` · `reads` · `writes` · `step` · `for_each` · `external`.

EVERYTHING IS A FILE, AND THAT IS THE WHOLE VOCABULARY. A model version is a file. A
hyperparameter is a file. Today's date is a file. A stage that answers to one reads it, and
the ordinary machinery does the rest — so what depends on a value is a thing you can list,
diff and draw, rather than a label on a call site that nothing can point at.

A consequence worth saying out loud: there is no sledgehammer that rebuilds everything. A
value rebuilds exactly the stages that declared they read it. That is more honest, and it is
also more work to wield, and it is the trade this design makes on purpose.

TWO QUESTIONS, AND THEY ARE NOT THE SAME QUESTION.

    what a DEPENDANT sees      the fingerprints of the shards it read. Nothing else.
    whether a STAGE re-runs    have those fingerprints moved since it last ran, or has
                               its own source changed

The first lives in the filename, so a dependant needs a directory listing and no more. The
second compares against what the last build actually saw, which is what `_index.json`
holds: the input datasets, the partitions taken from each, and their ids.

Keeping those apart is the point. Editing a builder re-runs that builder — and if the
numbers come out the same, the fingerprint is the same, the filename is the same, and
NOTHING downstream moves. The 287-second fit does not re-run because a stage two steps up
was reformatted.

The index is load-bearing, and it is worth being exact about how: **losing it causes a
REBUILD, never a false skip.** No record of what a shard was built from means the inputs
cannot be compared, so the shard cannot be shown current, so it is rebuilt. Corrupt, raced,
deleted — every failure lands on the safe side, which is not true of any state file whose
absence reads as "nothing has changed".

WHAT GOVERNS IS WHAT THE LAST BUILD READ, not the static scan. The two are derived by
different mechanisms, and a rebuild can reconcile a disagreement about DATA but not one
about DERIVATION — so a blind spot in the scan would become an artifact that rebuilds
forever with correct output and no error. The scan answers what is DECLARED, which is what
the graph and the checks are for.
"""
from __future__ import annotations

import ast
import datetime as _dt
import hashlib
import inspect
import os
import textwrap
import threading
import time
from contextlib import contextmanager
from typing import Callable, Sequence

from . import record as _record
from . import shards as _sh
from .errors import DeclError, StateError
from .paths import mkpath


def _canon(dataset: str) -> str:
    """One spelling of a dataset name, so two sites cannot mean one thing two ways."""
    if not isinstance(dataset, str) or not dataset.strip():
        raise DeclError(f"a dataset is a relative directory path, got {dataset!r}")
    d = dataset.strip().strip("/")
    if not d or d.startswith(".") or ":" in d:
        raise DeclError(
            f"{dataset!r} is not a relative dataset path. Datasets are named relative to "
            f"the root — 'processed/box_features/' — never absolutely and never as a URI, "
            f"which is what lets an id survive the data moving.")
    return d + "/"


def _why(why: object, dataset: str) -> str:
    if not isinstance(why, str) or not why.strip():
        raise DeclError(
            f"{dataset} needs why= — one line on what it is for. It is required because "
            f"there is nowhere else for it to live, which is what stops it going stale.")
    return why


class Pipeline:
    """One data tree, and the call sites that describe it."""

    def __init__(self, *,
                 root,
                 out_root=None,
                 source_dirs: Sequence[str] = ("src", "scripts"),
                 roots: Sequence[str] = ("raw/", "config/"),
                 project_root=None,
                 order_from=None,
                 trace=None,
                 stage_dir=None,
                 force: bool | None = None) -> None:
        self.project_root = mkpath(str(project_root), None) if project_root else None
        self.root = mkpath(root, self.project_root)
        self.out_root = mkpath(out_root, self.project_root) if out_root is not None else self.root
        self.source_dirs = tuple(source_dirs)
        self.roots = tuple(_canon(r) for r in roots)
        self.order_from = order_from
        self.stage_dir = stage_dir
        self.force = _env_force() if force is None else force
        self.trace_path = _abs_trace(trace)
        self._trace_fh = None
        self._local = threading.local()
        # dataset -> {"parts": [...], "id": "name:..."} for the scope being built. Cleared on
        # entering a step, so a stage's inputs are exactly the reads inside its body.
        self._reads: dict[str, dict] = {}
        self._prior: set[str] = set()
        self._externals: list[str] = []
        # Set by `step` for the duration of its body, so a `writes` inside inherits the
        # stage's partition and code hash instead of repeating them at every output.
        self._part: dict | None = None
        self._code: str = ""
        self._in_step = False

    def __repr__(self) -> str:
        return f"<Pipeline {self.root}>"

    # ── paths ─────────────────────────────────────────────────────────────────

    def resolve(self, dataset: str):
        """Where a dataset is READ from."""
        return self.root / _canon(dataset).rstrip("/")

    def resolve_out(self, dataset: str):
        """Where a dataset is WRITTEN to. Separate, so a local run cannot touch the tree."""
        return self.out_root / _canon(dataset).rstrip("/")

    # ── bookkeeping ───────────────────────────────────────────────────────────

    @property
    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @contextmanager
    def bookkeeping(self):
        """I/O in here is the pipeline inspecting itself, not a data edge.

        THREAD-LOCAL. A shared counter loses reads when a stage fetches concurrently, and
        the symptom is an artifact that silently depends on less than it did.
        """
        self._local.depth = self._depth + 1
        try:
            yield
        finally:
            self._local.depth -= 1

    def record(self, kind: str, **fields) -> None:
        _record.emit(self, kind, **fields)

    def _fresh_scope(self) -> None:
        self._reads, self._prior, self._externals = {}, set(), []

    # ── constants ─────────────────────────────────────────────────────────────

    def constants(self, dataset: str, *, why: str, **values):
        """Write values that come from OUTSIDE the data, as a shard like any other.

        A model version, a hyperparameter, today's date — anything that governs a result
        without being data the pipeline produced. As a file it is an ordinary input: a stage
        that answers to it reads it, `iv graph` draws the edge, and a change moves exactly
        the stages that declared they care.

        Idempotent by construction. The same values fingerprint the same, so the filename
        is the same and nothing is rewritten — calling this at the top of every run costs
        one local hash and touches the tree only when a value actually changed.

        THE CLOCK IS ONE OF THESE, and there is deliberately no `today()` shortcut for it:

            iv.constants("config/today/", why="poll once a day",
                           date=date.today().isoformat())

        A helper with a default path and a default `why=` would be a call the static scan
        cannot read — no literal to find — so the one node every polled feed hangs off
        would be the one node missing from the graph. The value is computed; the path and
        the reason are literals. That is the rule everywhere else and it holds here.
        """
        import polars as pl
        if not values:
            raise DeclError(f"{dataset} needs at least one value — that is what it is for.")
        self._fresh_scope()
        with self.writes(dataset, why=why) as out:
            pl.DataFrame({k: [v] for k, v in values.items()}).write_parquet(out)

    # ── reads ─────────────────────────────────────────────────────────────────

    def reads(self, dataset: str, *, why: str, where: dict | None = None,
              optional: bool = False, prior: bool = False) -> list:
        """Declare a dataset as an input; get the selected shards back, SORTED.

        The order is a guarantee, not a convenience. Row order is an input to anything that
        slices, takes the first row, or sums floats, and a listing that comes back in a
        different order every run makes the stage downstream irreproducible.

        `where` picks FILES, never rows. `where={"season": lambda s: s < t}` is how a stage
        that must not see the future is stopped from seeing it — it never opens the file,
        rather than opening it and filtering. `prior=True` reads what is there now and is
        excluded from the comparison: its producer runs later, so comparing it would make
        this artifact permanently stale one step behind itself.
        """
        name = _canon(dataset)
        _why(why, name)
        with self.bookkeeping():
            present = _sh.current_shards(self.resolve(name))
        sel = _sh.select(present, where, dataset=name)
        if not sel and not optional:
            raise StateError(
                f"{name} selected no shards"
                + (f" out of {len(present)} present" if present else " and is empty")
                + f". Read here because: {why}. Pass optional=True if producing nothing "
                f"here is legitimate.")
        if not self._depth:
            # THE RULE GOES IN THE RECORD, not the shards it happened to match. That is
            # what makes the check exact rather than a guess: `_replay` re-runs the same
            # selection against the dataset as it stands now, so a partition that appears
            # later inside the same range is picked up, and one outside it is not.
            # `parts` rides along for `iv why` to print; nothing decides on it.
            self._reads[name] = {
                "where": where,
                "parts": [s.part_str for s in sel],
                "id": _sh.dataset_id(sel),
            }
            if prior:
                self._prior.add(name)
            self.record("io", op="read", rel=name, why=why, n=len(sel), prior=prior)
        return [s.path for s in sel]

    def external(self, name: str, *, why: str) -> None:
        """A source outside the pipeline. Provenance, and no id.

        It cannot make anything stale, which is exactly why a stage built only from one runs
        once and never again. If it should re-run, it needs something that moves — read the
        clock.
        """
        _why(why, name)
        if not self._depth:
            self._externals.append(name)
            self.record("io", op="external", rel=f"external:{name}", why=why)

    def _inputs_now(self) -> dict[str, dict]:
        """`{dataset: {parts, id}}` for the reads in this scope, minus the prior ones."""
        return {name: dict(v) for name, v in self._reads.items() if name not in self._prior}

    # ── writes ────────────────────────────────────────────────────────────────

    @contextmanager
    def writes(self, dataset: str, *, why: str, part: dict | None = None,
               terminal: bool = False, code: str | None = None,
               allow_missing: bool = False):
        """Yield a LOCAL path to write one shard to. Commit it on a clean exit.

        Staged locally even when the dataset is a bucket, so fingerprinting the rows costs a
        local read rather than a download of what was just uploaded, and so the dataset never
        contains a partial file. See `shards.commit`.

        Nothing is recorded if the body raises. A record that did not depend on the build
        succeeding is how a shell `|| echo` once turned a failure into "current".
        """
        name = _canon(dataset)
        _why(why, name)
        part = self._part if part is None else part
        code = self._code if code is None else code
        staged = _sh.stage(f"{_sh.encode_part(part) or 'all'}-{time.time_ns()}",
                           self.stage_dir)
        started = time.time()
        try:
            yield staged
        except BaseException:
            with self.bookkeeping():
                if staged.exists():
                    staged.unlink()
            raise
        if not staged.exists():
            if allow_missing:
                self.record("io", op="skip-empty", rel=name, why=why)
                return
            raise DeclError(
                f"{name} was declared written but nothing was written to {staged}. Pass "
                f"allow_missing=True if producing nothing is a legitimate outcome — then "
                f"the shard stays absent and the next run tries again.")
        out_dir = self.resolve_out(name)
        with self.bookkeeping():
            final = _sh.commit(staged, out_dir, part=part)
            _sh.write_entry(out_dir, _sh.encode_part(part), {
                "fp": _sh.parse_name(final).fp,
                "code": code,
                "inputs": self._inputs_now(),
                "prior": sorted(self._prior),
                "external": list(self._externals),
                "terminal": terminal,
                "why": why,
                "by": self.node(),
                "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                "seconds": round(time.time() - started, 2),
            })
        self.record("io", op="write", rel=name, why=why, part=_sh.encode_part(part),
                    seconds=round(time.time() - started, 2))
        if not self._in_step:
            # A WRITE OUTSIDE A STEP CONSUMES THE SCOPE. Inside a step the scope is the
            # body — cleared on entry, shared by every output, which is what makes several
            # outputs of one computation record the same inputs. Outside one there is no
            # body to bound it, so without this a bare `writes` inherits whatever the last
            # stage happened to read and records it as its own upstream. Seen in the wild:
            # a raw season shard came out claiming the dataset it belongs to as an input.
            self._fresh_scope()

    # ── is it current ─────────────────────────────────────────────────────────

    def why_stale(self, dataset: str, part: dict | None = None,
                  *, code: str | None = None) -> str | None:
        """One line naming what moved, or None. Reads no parquet files.

        Compares the RECORD, never the filename's fingerprint: the fingerprint says what is
        IN the shard, which is a fact about the data and cannot be stale. What can be stale
        is the relationship between that data and the inputs it came from.

        `code=None` means nobody could tell us, and then the recorded value stands — turning
        the flag off stops keying on it rather than invalidating everything. It has to be a
        parameter because the recorded hash is the OLD one: detecting an edit needs the
        function as it stands now, and only the caller holds that.
        """
        name = _canon(dataset)
        part_str = _sh.encode_part(part)
        d = self.resolve_out(name)
        with self.bookkeeping():
            got = _sh.current_shards(d).get(part_str)
            if got is None:
                return f"not on disk ({part_str or 'the only shard'})"
            entry = (_sh.read_index(d).get("shards") or {}).get(part_str)
            if not entry or "inputs" not in entry:
                return ("no record of what it was built from, so its inputs cannot be "
                        "compared — rebuilding is the only safe answer")
            was_code = entry.get("code", "")
            if code is not None and code != was_code:
                return f"code changed: {was_code or '(none)'} -> {code or '(none)'}"
            ids_now, problem = self._replay(entry["inputs"])
            if problem:
                return problem
            moved = [k for k, v in entry["inputs"].items() if ids_now.get(k) != v["id"]]
            if moved:
                return f"input moved: {', '.join(moved)}"
        return None

    def _replay(self, recorded: dict) -> tuple[dict[str, str], str]:
        """Re-run each recorded selection against the dataset as it stands now.

        EXACT, because the selection is data. `where=None` is every shard, so a joint fit
        notices a season that did not exist at build time. `where={"season": {"lt": T}}` is
        re-evaluated, so a season backfilled BELOW the bound is picked up and one added
        above it is not — which is what makes a walk-forward stage both correct and
        precise. Remembering the shards a rule matched, rather than the rule, could do only
        one of those.
        """
        out: dict[str, str] = {}
        for name, was in recorded.items():
            live = _sh.current_shards(self.resolve(name))
            try:
                sel = _sh.select(live, was.get("where"), dataset=name)
            except StateError as e:
                # A named partition is gone. That is a fact about the tree, not a crash.
                return out, f"{name}: {e}"
            out[name] = _sh.dataset_id(sel)
        return out, ""

    def is_current(self, dataset: str, part: dict | None = None, **kw) -> bool:
        return self.why_stale(dataset, part, **kw) is None

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, *, why: str, part: dict | None = None,
             code: bool | None = None, if_needed: bool = True) -> Callable:
        """Mark a function as a stage. Skip it when everything it writes is up to date.

        IT NAMES NOTHING. `writes` is the only place an output is declared, so there is one
        declaration per output and no way for the two to disagree. What this adds is the one
        thing a context manager structurally cannot do: DECLINE TO RUN THE BODY. `__enter__`
        has already been entered by the time it gets control, so `writes` can never skip the
        expensive work that precedes it — a callable is the only thing you can choose not to
        call. PEP 377 proposed letting a context manager skip its body and was rejected.

        The outputs are read out of THIS FUNCTION'S OWN SOURCE at decoration time, not out
        of the project scan. Local, exact, and it needs no configuration — it works from a
        REPL, a test file, or a script that is not under `source_dirs`. The cost is that it
        sees `iv.writes(...)` in the body and not in a helper the body calls; `iv check`
        uses the project-wide scan to catch that case, where it is a warning about the code
        rather than a silent hole in the skip check.

        `part=` and `code=` are AMBIENT for the body: a `writes` inside inherits them unless
        it passes its own. That keeps one declaration of each rather than repeating the
        partition at every output of a stage that writes several.
        """
        _why(why, "step")

        def decorate(fn: Callable) -> Callable:
            code_hash = "" if code is False else source_digest(fn)
            outputs = writes_in(fn)

            def wrapper(*args, **kwargs):
                # THE SKIP CHECK. Every output, not just one — a stage that writes three
                # datasets and is missing the third has work to do.
                if if_needed and not self.force and outputs:
                    reasons = {o: self.why_stale(o, part, code=code_hash) for o in outputs}
                    if not any(reasons.values()):
                        print(f"  {', '.join(outputs)} — up to date, skipping")
                        for o in outputs:
                            self.record("skip", rel=o, part=_sh.encode_part(part))
                        return False
                    for o, r in reasons.items():
                        if r:
                            print(f"  {o}: {r}")
                self._fresh_scope()
                prev = (self._part, self._code, self._in_step)
                self._part, self._code, self._in_step = part, code_hash, True
                try:
                    fn(*args, **kwargs)
                    return True
                finally:
                    self._part, self._code, self._in_step = prev
                    # ON THE WAY OUT TOO. The body's reads belong to the body; leaving them
                    # in scope means the next bare `writes` records them as its own.
                    self._fresh_scope()

            wrapper.__name__ = getattr(fn, "__name__", "step")
            wrapper.__doc__ = fn.__doc__
            wrapper.run = fn
            wrapper.outputs = outputs
            return wrapper
        return decorate

    # ── for_each ──────────────────────────────────────────────────────────────

    def for_each(self, over, build_one: Callable, *, dataset: str, key: str, why: str,
                 code: bool | None = None, quiet: bool = False) -> list[str]:
        """One shard per partition. Build only the ones that are not current.

        There is no assembly step and no map of which partitions are fresh, because the
        directory is both. A partition is current when its shard is there and its inputs
        have not moved — the same question asked of a whole dataset, asked once per file.
        """
        name = _canon(dataset)
        _why(why, name)
        code_hash = "" if code is False else source_digest(build_one)
        want = [str(p) for p in over]
        reuse, rebuild = [], []
        for p in want:
            current = (not self.force
                       and self.why_stale(name, {key: p}, code=code_hash) is None)
            (reuse if current else rebuild).append(p)
        if not quiet:
            print(f"  partitions [{name}] by {key}")
            print(f"    reuse   ({len(reuse):>2}): {_span(reuse)}")
            print(f"    rebuild ({len(rebuild):>2}): {_span(rebuild)}")
        for p in rebuild:
            self._fresh_scope()
            with self.writes(name, why=why, part={key: p}, code=code_hash) as out:
                build_one(p, out)
        return rebuild

    # ── misc ──────────────────────────────────────────────────────────────────

    def node(self) -> str:
        """This stage's name: the script that is running."""
        import sys
        override = os.environ.get("IV_STAGE")
        if override:
            return override
        p = mkpath(sys.argv[0] or "<repl>", self.project_root)
        try:
            return str(p.relative_to(self.project_root)) if self.project_root else str(p)
        except ValueError:
            return str(p)

    def reset(self) -> None:
        self._fresh_scope()


def writes_in(fn: Callable) -> tuple[str, ...]:
    """The datasets a function's own body writes, read out of its source.

    Only `iv.writes(...)` and `iv.constants(...)` with a literal dataset and a literal
    `why=` — the same rule the project scan applies, for the same reason: a computed name
    cannot be read without running the code.

    UNREADABLE SOURCE IS AN ERROR. It used to degrade to `()`, which meant a step with no
    known outputs and therefore no skip check — every stage running every time, silently,
    with nothing to see. Safe in the sense that it never skips wrongly, and useless in the
    sense that nobody would notice the cache had stopped working.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise DeclError(
            f"cannot read the source of {getattr(fn, '__name__', fn)!r}, so there is no "
            f"way to know what it writes and no way to decide whether to skip it. This "
            f"happens in a REPL, a notebook, or a script piped in on stdin. Run it from a "
            f"file, or pass if_needed=False to say the stage should always run.") from e
    out = []
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in ("writes", "constants"):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        if not isinstance(kw.get("why"), ast.Constant):
            continue
        target = node.args[0] if node.args else kw.get("dataset")
        if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
            # A COMPUTED DATASET NAME IS AN ERROR, not something to skip past. Skipping it
            # would leave the step with an incomplete list of outputs and no sign of it —
            # the stage would go on being "up to date" while one of its outputs was
            # missing, which is the exact failure the skip check exists to prevent.
            shown = ast.dump(target, annotate_fields=False)[:60] if target else "missing"
            raise DeclError(
                f"{getattr(fn, '__name__', fn)!r} writes a dataset this cannot read: "
                f"{shown}. Inside an @iv.step the dataset must be a string LITERAL, "
                f"because that is how the skip check learns what the stage produces. "
                f"A partition goes in part=, not in the name.")
        name = _canon(target.value)
        if name not in out:
            out.append(name)
    return tuple(out)


def source_digest(fn: Callable) -> str:
    """A hash of what a function DOES, insensitive to how it is spelled.

    `ast.unparse` normalises whitespace, comments and formatting away, and the decorators
    are stripped — so reformatting, or editing a `why=`, does not invalidate data, while a
    real change to the logic does. Raw-source hashing churns on all three.

    SHALLOW: it sees this function, not the helpers it calls.

    Unreadable source raises, for the same reason `writes_in` does — quietly returning ""
    would stop keying on the code with nothing to show for it.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise DeclError(
            f"cannot read the source of {getattr(fn, '__name__', fn)!r} to hash it. "
            f"Run the stage from a file, or pass code=False to stop keying on it.") from e
    node = ast.parse(textwrap.dedent(src)).body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.decorator_list = []
    return hashlib.sha256(ast.unparse(node).encode()).hexdigest()[:_sh.DIGEST_LEN]


def _span(parts: list[str]) -> str:
    if not parts:
        return "—"
    return ", ".join(parts) if len(parts) <= 3 else f"{parts[0]}..{parts[-1]}"


def _env_force() -> bool:
    return os.environ.get("IV_FORCE", "").lower() in ("1", "true", "yes")


def _abs_trace(trace):
    from pathlib import Path
    t = trace or os.environ.get("IV_TRACE")
    return Path(t).expanduser().resolve() if t else None
