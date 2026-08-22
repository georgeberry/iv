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
from pathlib import Path
from typing import Callable, Sequence

from . import assets as _assets
from . import decl as _decl
from . import record as _record
from . import shards as _sh
from .static import _lit_sel
from .errors import DeclError, StateError

#: Stands for the partition being built, inside a `where=`. It is what makes a
#: partition-relative selector readable without running the closure that would
#: otherwise supply it — and that is what lets a shard's key be computed before
#: its body runs, so nothing has to be written down.
PART = "\x00PART"
from .paths import mkpath


def _canon(dataset: str) -> str:
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


_ACTIVE: list = []


def _check_declared(target) -> None:
    """Reading inside a pipeline's tree without going through `iv.reads()` raises.

    Absent from the graph AND from the recorded inputs, so whatever it depends on can
    change and the artifact never rebuilds. Nothing detects that afterwards: the read
    succeeds and the number is simply wrong.
    """
    s = str(target)
    for iv in _ACTIVE:
        if s in iv._handed_out or iv._depth:
            return
    for iv in _ACTIVE:
        for base in (iv.out_root, iv.root):
            if base and s.startswith(str(base)):
                raise DeclError(
                    f"{s} is inside the data tree but was not handed back by iv.reads(). "
                    f"An undeclared read is absent from the graph and from the recorded "
                    f"inputs, so its source can change and this will never rebuild. "
                    f"Declare it: iv.reads('<dataset>/', why='...').")


class Pipeline:

    def __init__(self, *,
                 root,
                 out_root=None,
                 source_dirs: Sequence[str] = ("src", "scripts"),
                 roots: Sequence[str] = ("raw/", "config/"),
                 project_root=None,
                 trace=None,
                 stage_dir=None,
                 force: bool | None = None) -> None:
        self.project_root = mkpath(str(project_root), None) if project_root else None
        self.root = mkpath(root, self.project_root)
        self.out_root = mkpath(out_root, self.project_root) if out_root is not None else self.root
        self.source_dirs = tuple(source_dirs)
        self.roots = tuple(_canon(r) for r in roots)
        self.stage_dir = stage_dir
        self.force = _env_force() if force is None else force
        self.trace_path = _abs_trace(trace)
        self._trace_fh = None
        self._local = threading.local()
        self._reads: dict[str, dict] = {}
        self._updating: set[str] = set()
        self._plain: set[str] = set()
        self._externals: list[str] = []
        self._handed_out: set[str] = set()
        # Undeclared I/O against the data tree raises. Not opt-in: an undeclared
        # WRITE makes a shard's fingerprint-name a lie, and an undeclared READ is
        # absent from the recorded inputs, so its source can change forever and
        # nothing rebuilds. Neither is detectable after the fact.
        _ACTIVE.append(self)
        self._enforce_writes()
        self._enforce_reads()
        self._part: dict | None = None
        self._in_step = False
        self._node = ""
        self._inputs: tuple = ()
        self._outputs: tuple[str, ...] = ()
        # dataset -> the upstreams its stage declares. Filled in by @step and for_each at
        # DECLARATION time, not at run time, so `why_stale("processed/x/")` answers on its
        # own — the question does not need the stage to have run, only to exist.
        self._declared: dict[str, tuple] = {}
        # dataset -> the Asset that builds it. Populated by @iv.data at DECLARATION time,
        # so `iv graph` and `iv status` know the pipeline by importing it rather than by
        # parsing it — which is what lets a stage defined in a notebook declare as well as
        # one in a scanned file.
        self._assets: dict[str, _assets.Asset] = {}

    def __repr__(self) -> str:
        return f"<Pipeline {self.root}>"


    def resolve(self, dataset: str):
        return self.root / _canon(dataset).rstrip("/")

    def resolve_out(self, dataset: str):
        return self.out_root / _canon(dataset).rstrip("/")


    @property
    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @contextmanager
    def bookkeeping(self):
        self._local.depth = self._depth + 1
        try:
            yield
        finally:
            self._local.depth -= 1

    def record(self, kind: str, **fields) -> None:
        _record.emit(self, kind, **fields)

    def _fresh_scope(self) -> None:
        self._reads, self._updating, self._plain, self._externals = {}, set(), set(), []


    def constants(self, dataset: str, *, why: str, **values):
        import polars as pl
        if not values:
            raise DeclError(f"{dataset} needs at least one value — that is what it is for.")
        self._fresh_scope()
        with self.writes(dataset, why=why) as out:
            pl.DataFrame({k: [v] for k, v in values.items()}).write_parquet(out)


    def reads(self, dataset: str, *, why: str, where: dict | None = None,
              optional: bool = False, update_file_on_disk: bool = False) -> list:
        name = _canon(dataset)
        _why(why, name)
        where = _sub_part(where, self._part, name)
        if update_file_on_disk and not self._depth:
            self._check_updates_own(name)
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
            plain = name not in self._updating or not update_file_on_disk
            if plain or name not in self._reads:
                self._reads[name] = {
                    "where": where,
                    "parts": [s.part_str for s in sel],
                    "id": _sh.dataset_id(sel),
                }
            if update_file_on_disk and name not in self._plain:
                self._updating.add(name)
            if not update_file_on_disk:
                self._plain.add(name)
                self._updating.discard(name)
            self.record("io", op="read", rel=name, why=why, n=len(sel),
                        update_file_on_disk=update_file_on_disk)
        self._handed_out.update(str(s.path) for s in sel)
        return [s.path for s in sel]

    def _check_updates_own(self, name: str) -> None:
        """`update_file_on_disk=` may only name a dataset this same stage writes.

        The flag excludes a dataset from the staleness comparison, which is mandatory when
        the stage is about to overwrite that dataset — otherwise it is permanently one step
        behind its own last output. Pointed at SOMEONE ELSE'S dataset it means the opposite:
        the dependency is real, and hiding it from the comparison is how a stage silently
        never rebuilds when its input moves.

        Checked here, at the read, rather than at the write: a stage that updates two
        datasets writes them one at a time, so at the first write the second is legitimately
        flagged and not yet written.
        """
        if name in self._outputs:
            return
        raise DeclError(
            f"{name} was read with update_file_on_disk=True, but this stage writes "
            f"{', '.join(self._outputs) or 'nothing'}. That flag means \"the copy of this "
            f"dataset I am about to overwrite\", and it is excluded from the staleness "
            f"comparison — so on another stage's dataset it hides a real dependency and "
            f"this one never rebuilds when that input moves. Run its producer first and "
            f"read it normally.")

    def external(self, name: str, *, why: str) -> None:
        _why(why, name)
        if not self._depth:
            self._externals.append(name)
            self.record("io", op="external", rel=f"external:{name}", why=why)

    def _enforce_writes(self) -> None:
        owners = [Path]
        try:
            from cloudpathlib import CloudPath
            owners.append(CloudPath)
        except ImportError:
            pass
        for owner in owners:
            for name in ("write_text", "write_bytes", "open"):
                fn = getattr(owner, name, None)
                if fn is None or getattr(fn, "_iv_checked", False):
                    continue
                setattr(owner, name, self._checked_write(owner, name, fn))

    @staticmethod
    def _checked_write(owner, name, fn):
        def patched(target, *a, **kw):
            writing = name != "open" or "w" in (a[0] if a else kw.get("mode", "r")) \
                or "a" in (a[0] if a else kw.get("mode", "r"))
            if writing and any(str(target) in iv._handed_out for iv in _ACTIVE):
                raise DeclError(
                    f"{target} was handed back by iv.reads() and is being written to. A "
                    f"shard's name is a fingerprint of its contents, so overwriting one "
                    f"in place makes the name a lie that nothing can detect. Write "
                    f"through iv.writes().")
            return fn(target, *a, **kw)
        patched._iv_checked = True
        return patched

    def _enforce_reads(self) -> None:
        owners = [Path]
        try:
            from cloudpathlib import CloudPath
            owners.append(CloudPath)
        except ImportError:
            pass
        for owner in owners:
            for name in ("read_text", "read_bytes", "open"):
                fn = getattr(owner, name, None)
                if fn is None or getattr(fn, "_iv_read_checked", False):
                    continue
                setattr(owner, name, self._checked_read(name, fn))
        try:
            import polars as pl
        except ImportError:
            return
        for name in ("read_parquet", "scan_parquet", "read_csv", "read_parquet_schema"):
            fn = getattr(pl, name, None)
            if fn is None or getattr(fn, "_iv_read_checked", False):
                continue
            setattr(pl, name, self._checked_read(name, fn, first_arg=True))

    @staticmethod
    def _checked_read(name, fn, first_arg=False):
        def patched(target, *a, **kw):
            reading = first_arg or name != "open" or not any(
                m in (a[0] if a else kw.get("mode", "r")) for m in ("w", "a", "x"))
            if reading:
                for one in (target if isinstance(target, (list, tuple)) else [target]):
                    _check_declared(one)
            return fn(target, *a, **kw)
        patched._iv_read_checked = True
        return patched

    def snapshot(self):
        """A consistent, memoised view of the tree for a read-only pass — see
        `shards.snapshot`. Wrap a loop that asks `is_current` many times in it; never wrap
        anything that writes, because a commit would not be seen."""
        return _sh.snapshot()

    def verify(self, dataset: str) -> list[str]:
        with self.bookkeeping():
            out = []
            d = self.resolve_out(dataset)
            live = _sh.current_shards(d)
            for part, shard in live.items():
                actual = _sh.fingerprint_of_file(shard.path)
                if actual != shard.fp:
                    out.append(f"{shard.name}: contents fingerprint {actual}, name says "
                               f"{shard.fp} — the file was changed after it was committed")
            by_schema = _sh.schemas_of(live.values()) if len(live) > 1 else {}
            if len(by_schema) > 1:
                groups = sorted(by_schema.items(), key=lambda kv: -len(kv[1]))
                base = set(c for c, _ in groups[0][0])
                lines = []
                for cols, parts in groups[1:]:
                    d2 = set(c for c, _ in cols) ^ base
                    lines.append(f"{_span_parts(parts)} differ by {sorted(d2)}")
                out.append(f"SCHEMA DRIFT: {len(by_schema)} column sets across {len(live)} "
                           f"shards — {_span_parts(groups[0][1])} is the majority; "
                           + "; ".join(lines)
                           + ". A read of the whole dataset cannot produce one frame.")
            return out

    def key_of(self, dataset: str, part: dict | None, inputs) -> str:
        """The derivation key: a digest of what this shard is built FROM, resolved now.

        This is the whole record. Recomputed from the declared upstreams and the files on
        disk, it either matches a name that is here or it does not — so there is nothing to
        write down, nothing to lose, and no way for a record to disagree with the tree.

        An input that selects nothing for THIS partition is not an input to it. That is what
        lets a stage branch — the settled half of a feed for an old season, the live half for
        the current one — without the branch it did not take dragging the other's identity
        into the key and rebuilding a finished shard every day.

        No declared inputs at all means no derivation: the empty key, and a root-shaped name.
        A fetcher's output really is a root, and this says so.
        """
        pairs = []
        for name, sel, optional in inputs:
            if name == dataset:
                # An artifact cannot be its own upstream. Writing it changes its own
                # identity, so a key folding it in would move every time it was built and
                # never settle. `update_file_on_disk=` says this out loud; this catches the
                # stage that reads the same dataset plainly as well.
                continue
            if sel is None:
                raise DeclError(
                    f"{name} is read with a where= this cannot read without running the "
                    f"stage, so the shard's key cannot be computed and nothing can decide "
                    f"whether to skip. Selectors have to be data: a literal, or iv.PART for "
                    f"the partition being built.")
            live = _sh.current_shards(self.resolve(name))
            try:
                got = _sh.select(live, _resolve_sel(sel, part, name), dataset=name)
            except StateError:
                # An explicit list is a COVERAGE CLAIM, so a value that is not there is a
                # fact worth saying out loud rather than a key that quietly differs. Unless
                # the read is optional — which is how a stage that branches over two halves
                # of one feed says that the half it did not take is not its business.
                if optional:
                    continue
                raise
            if got:
                pairs.append((name, _sh.dataset_id(got)))
        if not inputs:
            return ""
        body = "|".join(f"{n}={i}" for n, i in sorted(pairs))
        return _sh._short(f"key:{dataset}|{_sh.encode_part(part)}|{body}")


    @contextmanager
    def writes(self, dataset: str, *, why: str, part: dict | None = None,
               terminal: bool = False,
               allow_missing: bool = False, ext: str = _sh.EXT):
        name = _canon(dataset)
        _why(why, name)
        part = self._part if part is None else part
        staged = _sh.stage(f"{_sh.encode_part(part) or 'all'}-{time.time_ns()}",
                           self.stage_dir, ext)
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
        # The name is the record, so it has to be a true one. A read this stage really made
        # and did not declare would be absent from the key, which means its source could
        # change forever and nothing would rebuild — the exact failure the index used to
        # catch after the fact, now refused at the write. The other direction is fine: a
        # declared input on a branch that was not taken is not a lie.
        declared = {n for n, _, _ in self._inputs}
        undeclared = sorted(set(self._reads) - self._updating - declared)
        if self._inputs and undeclared:
            raise DeclError(
                f"{name} was built from {undeclared}, which this stage does not declare. "
                f"The shard's key is computed from the DECLARED upstreams, so an input "
                f"outside them cannot move it and this would never rebuild. Read it with a "
                f"literal dataset name at the top of the stage.")
        out_dir = self.resolve_out(name)
        with self.bookkeeping():
            key = self.key_of(name, part, self._inputs)
            final = _sh.commit(staged, out_dir, part=part, key=key)
        self.record("io", op="write", rel=name, why=why, part=_sh.encode_part(part),
                    key=key, seconds=round(time.time() - started, 2))
        if not self._in_step:
            self._fresh_scope()


    def why_stale(self, dataset: str, part: dict | None = None, *,
                  inputs=None) -> str | None:
        """Why this artifact would be rebuilt, or None if it would not.

        Three things and no fourth: the upstreams this stage declares, their identity as the
        files on disk stand right now, and the name of the shard already here. The name
        carries the key it was built under, so the comparison is a string equality against a
        value recomputed from scratch — there is no record to go missing, to be stale, or to
        disagree with the tree.

        What it cannot say any more is WHICH input moved. A key is a hash and hashes do not
        invert. `iv why` prints the resolved upstreams instead, which is the same question
        asked forwards.
        """
        name = _canon(dataset)
        inputs = self._declared.get(name, ()) if inputs is None else inputs
        d = self.resolve_out(name)
        with self.bookkeeping():
            live = _sh.current_shards(d)
            if part is None and "" not in live:
                # A PARTITIONED DATASET, ASKED ABOUT AS A WHOLE. One computation may write
                # many shards — `box_features` has career-cumulative terms, so it is built
                # in one pass and split — and then the stage's question is "is every shard
                # of mine current", not "is the unpartitioned one".
                if not live:
                    return "not on disk (nothing built)"
                for p in sorted(live, key=_sh.sort_key):
                    reason = self.why_stale(name, _sh.decode_part(p), inputs=inputs)
                    if reason:
                        return f"{p}: {reason}"
                return None
            part_str = _sh.encode_part(part)
            got = live.get(part_str)
            if got is None:
                return f"not on disk ({part_str or 'the only shard'})"
            try:
                want = self.key_of(name, part, inputs)
            except StateError as e:
                return str(e)
            if not want:
                # Nothing derives it: a root, or a fetcher with no declared upstream. Its
                # identity is its contents, and there is no question to ask.
                return None
            if got.key != want:
                return ("its inputs moved — the key in its name is not the one its declared "
                        "upstreams produce now")
        return None

    def is_current(self, dataset: str, part: dict | None = None, **kw) -> bool:
        return self.why_stale(dataset, part, **kw) is None


    #: The selector vocabulary, on the instance so a declaration reads as what it is at
    #: the call site: `def fit(box=iv.same_part("raw/box/", why="..."))`.
    all_of = staticmethod(_decl.all_of)
    same_part = staticmethod(_decl.same_part)
    before_part = staticmethod(_decl.before_part)
    after_part = staticmethod(_decl.after_part)
    between = staticmethod(_decl.between)
    parts = staticmethod(_decl.parts)
    own_last_copy = staticmethod(_decl.own_last_copy)

    def data(self, dataset: str, *, why: str, part: str | None = None,
             ext: str = _sh.EXT, terminal: bool = False,
             if_needed: bool = True, once: bool = False) -> Callable:
        """Name a dataset and decorate the function that builds it.

        The upstreams are parameter defaults, which is what makes the whole declaration
        readable off the function object — no source text, nothing run:

            @iv.data("processed/cohorts/", why="a fit per cohort", part="season")
            def cohorts(past=iv.before_part("processed/features/", why="prior seasons")):
                return past.group_by("player").agg(pl.col("z").mean())
        """
        name = _canon(dataset)
        _why(why, name)

        def decorate(fn: Callable) -> _assets.Asset:
            asset = _assets.Asset(self, name, fn, why=why, part=part, ext=ext,
                                  terminal=terminal, if_needed=if_needed, once=once)
            if name in self._assets and self._assets[name].fn is not fn:
                raise DeclError(
                    f"{name} is already built by "
                    f"{self._assets[name].__name__!r}. A dataset has one producer — two "
                    f"would race, and whichever ran last would win.")
            self._assets[name] = asset
            self._declared[name] = asset.triples()
            return asset
        return decorate

    def step(self, *, why: str, part: dict | None = None,
             if_needed: bool = True) -> Callable:
        _why(why, "step")

        def decorate(fn: Callable) -> Callable:
            outputs = writes_in(fn)
            inputs = reads_in(fn)
            node_name = self._node_name(fn)
            for o in outputs:
                self._declared[o] = inputs

            def wrapper(*args, **kwargs):
                if if_needed and not self.force and outputs:
                    reasons = {o: self.why_stale(o, part, inputs=inputs) for o in outputs}
                    if not any(reasons.values()):
                        print(f"  {', '.join(outputs)} — up to date, skipping")
                        for o in outputs:
                            self.record("skip", rel=o, part=_sh.encode_part(part))
                        return False
                    for o, r in reasons.items():
                        if r:
                            print(f"  {o}: {r}")
                self._fresh_scope()
                prev = (self._part, self._in_step, self._node, self._inputs,
                        self._outputs)
                self._part, self._in_step, self._node = part, True, node_name
                self._inputs, self._outputs = inputs, outputs
                try:
                    fn(*args, **kwargs)
                    return True
                finally:
                    (self._part, self._in_step, self._node, self._inputs,
                     self._outputs) = prev
                    self._fresh_scope()

            wrapper.__name__ = getattr(fn, "__name__", "step")
            wrapper.__doc__ = fn.__doc__
            wrapper.run = fn
            wrapper.outputs = outputs
            return wrapper
        return decorate


    def for_each(self, over, build_one: Callable, *, dataset: str, key: str, why: str,
                 quiet: bool = False) -> list[str]:
        inputs = reads_in(build_one)
        name = _canon(dataset)
        _why(why, name)
        self._declared[name] = inputs
        want = [str(p) for p in over]
        reuse, rebuild = [], []
        for p in want:
            current = (not self.force
                       and self.why_stale(name, {key: p}, inputs=inputs) is None)
            (reuse if current else rebuild).append(p)
        if not quiet:
            print(f"  partitions [{name}] by {key}")
            print(f"    reuse   ({len(reuse):>2}): {_span(reuse)}")
            print(f"    rebuild ({len(rebuild):>2}): {_span(rebuild)}")
        for p in rebuild:
            self._fresh_scope()
            prev = (self._inputs, self._outputs, self._part)
            self._inputs, self._outputs, self._part = inputs, (name,), {key: p}
            with self.writes(name, why=why, part={key: p}) as out:
                build_one(p, out)
            self._inputs, self._outputs, self._part = prev
        return rebuild


    def _node_name(self, fn: Callable) -> str:
        """The name the static scan gives this step: `<file>::<function>`.

        A node is a step, not a file — so a project may keep every stage in one file and
        still get one node each. Derived from the function's own code object, so it agrees
        with the scan whatever imported it.
        """
        src = getattr(fn, "__code__", None)
        if src is None:
            return getattr(fn, "__name__", "<step>")
        return f"{self._rel_source(src.co_filename)}::{fn.__name__}"

    def _rel_source(self, filename: str) -> str:
        p = mkpath(filename, self.project_root)
        try:
            return str(p.relative_to(self.project_root)) if self.project_root else str(p)
        except ValueError:
            return str(p)

    def node(self) -> str:
        import sys
        if self._node:
            return self._node
        override = os.environ.get("IV_STAGE")
        if override:
            return override
        return self._rel_source(sys.argv[0] or "<repl>")

    def reset(self) -> None:
        self._fresh_scope()


def reads_in(fn: Callable) -> tuple:
    """What this step declares it reads: `[(dataset, selector)]`, off its own source.

    Read from the FUNCTION rather than from a scan of the project, so a step defined inline
    — a test, a notebook, `repro.py` — declares just as well as one in a scanned file.

    An `update_file_on_disk=` read is left out. It is the copy of its own output this stage
    is about to overwrite, and folding a shard's own identity into its own key is a
    definition that never settles.
    """
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        return ()
    out = []
    for node in ast.walk(ast.parse(textwrap.dedent(src))):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "reads" and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        flag = kw.get("update_file_on_disk")
        if isinstance(flag, ast.Constant) and flag.value is True:
            continue
        opt = kw.get("optional")
        out.append((_canon(node.args[0].value), _lit_sel(kw.get("where")),
                    isinstance(opt, ast.Constant) and opt.value is True))
    return tuple(sorted(set(out), key=lambda x: x[0]))


def _sub_part(where: dict | None, part: dict | None, name: str):
    """Replace `iv.PART` with the partition being built, in a real `where=`.

    The same substitution `_resolve_sel` does for the STATIC form. Both exist because the
    selector is read twice — once off the source to compute the key before the body runs,
    once here when the body actually opens the files — and they have to agree, which they
    do by both meaning "the shard being built".
    """
    if not where:
        return where
    def one(v, k):
        if v != PART:
            return v
        if not part or k not in part:
            raise DeclError(
                f"{name} selects on iv.PART for {k!r}, but this stage is not building a "
                f"partition keyed on {k!r}. PART stands for the shard being built, so it "
                f"only means something where there is one.")
        return str(part[k])
    out = {}
    for k, rule in where.items():
        if isinstance(rule, dict):
            out[k] = {op: one(v, k) for op, v in rule.items()}
        elif isinstance(rule, (list, tuple, set)):
            out[k] = [one(v, k) for v in rule]
        else:
            out[k] = one(rule, k)
    return out


def _resolve_sel(sel, part: dict | None, name: str):
    """A statically-read selector plus the partition being built -> a real `where=`."""
    if not sel:
        return None

    def one(v, k):
        if v != PART:
            return v
        if not part or k not in part:
            raise DeclError(
                f"{name} selects on iv.PART for {k!r}, but this stage is not building a "
                f"partition keyed on {k!r}. PART stands for the shard being built, so it "
                f"only means something where there is one.")
        return str(part[k])

    out = {}
    for k, (kind, body) in sel:
        out[k] = ([one(v, k) for v in body] if kind == "in"
                  else {op: one(v, k) for op, v in body})
    return out


def writes_in(fn: Callable) -> tuple[str, ...]:
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


def _span_parts(parts: list[str]) -> str:
    s = sorted(parts)
    return s[0] if len(s) == 1 else f"{s[0]}..{s[-1]}"


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


# `iv.PART` on the instance, so a selector reads as what it is at the call site.
Pipeline.PART = PART
