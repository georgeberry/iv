from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Sequence

from . import assets as _assets
from . import decl as _decl
from .decl import PART, _canon, _why
from . import record as _record
from . import shards as _sh
from .errors import DeclError, StateError

from .paths import mkpath


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
        for base in (iv.out_tree, iv.tree):
            if base and s.startswith(str(base)):
                raise DeclError(
                    f"{s} is inside the data tree but was not handed back by iv.reads(). "
                    f"An undeclared read is absent from the graph and from the recorded "
                    f"inputs, so its source can change and this will never rebuild. "
                    f"Declare it: iv.reads('<dataset>/', why='...').")


class Invalidator:
    """What decides whether a stage has to run.

    Three paths, and they used to be called root, out_root and project_root, which is
    three different things wearing one word. A fourth, `roots`, named dataset PREFIXES that
    arrive from outside — a rule about paths — and is gone: those datasets are declared, one
    at a time, with `iv.source(...)`.

        tree      where the DATA lives. A dataset is named relative to it, which is what
                  lets an id survive the data moving.
        out_tree  where writes GO, if that is somewhere else. Defaults to `tree`.
        project   where the CODE lives. Node names are relative to it, so they read the
                  same however the module was imported.
        code      the modules `iv preflight` reads for undefined names and dead imports.
    """

    def __init__(self, *,
                 tree,
                 out_tree=None,
                 code: Sequence[str] = ("src", "scripts"),
                 project=None,
                 trace=None,
                 stage_dir=None,
                 force: bool | None = None) -> None:
        self.project = mkpath(str(project), None) if project else None
        self.tree = mkpath(tree, self.project)
        self.out_tree = mkpath(out_tree, self.project) if out_tree is not None else self.tree
        self.code = tuple(code)
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
        # dataset -> the Asset that builds it. Populated by @iv.step at DECLARATION time,
        # so `iv graph` and `iv status` know the pipeline by importing it rather than by
        # parsing it — which is what lets a stage defined in a notebook declare as well as
        # one in a scanned file.
        self._assets: dict[str, _assets.Asset] = {}
        # dataset -> the Source it arrives as. Declared, not inferred from a path prefix,
        # so every dataset in the pipeline has exactly one declaration somewhere.
        self._sources: dict[str, _assets.Source] = {}
        #: Datasets declared on their own line with `iv.data(...)`, so a read can name one
        #: before the stage that writes it exists. A stage's inline `output="..."` is a
        #: declaration too; it just has nowhere else to be said.
        self._datasets: dict[str, _assets.Dataset] = {}

    def __repr__(self) -> str:
        return f"<Invalidator {self.tree}>"


    def resolve(self, dataset: str):
        return self.tree / _canon(dataset).rstrip("/")

    def resolve_out(self, dataset: str):
        return self.out_tree / _canon(dataset).rstrip("/")


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


    def reads(self, dataset: str, *, why: str, where: dict | None = None,
              optional: bool = False, update_file_on_disk: bool = False) -> list:
        name = _canon(dataset)
        _why(why, name)
        where = _sub_part(where, self._part, name)
        if update_file_on_disk and not self._depth:
            self._check_updates_own(name)
        with self.bookkeeping():
            present = _sh.current_shards(self.resolve(name))
        try:
            sel = _sh.select(present, where, dataset=name)
        except StateError:
            # An explicit list is a COVERAGE CLAIM, and a value that is not there is a fact
            # worth saying out loud — unless the read is optional, which is how a stage says
            # the half it did not take is not its business. `key_of` has always read it that
            # way; this did not, so a stage could be told its key was fine and then raise
            # opening the same read. Two answers is one too many.
            if not optional:
                raise
            sel = []
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
        # DEDUPED, because the same upstream can arrive twice. A stage that branches —
        # the W draft and the NBA draft read the same box scores for different reasons —
        # has two call sites for one dataset, and the scan reports a site each while
        # `reads_in` reports the set. Folding it in twice makes a different key, so
        # `iv status` called such a stage stale forever while the run, asking the same
        # question the other way round, skipped it. Two answers is one too many.
        body = "|".join(f"{n}={i}" for n, i in sorted(set(pairs)))
        return _sh._short(f"key:{dataset}|{_sh.encode_part(part)}|{body}")


    @contextmanager
    def writes(self, dataset: str, *, why: str, part: dict | None = None,
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
            _sh.commit(staged, out_dir, part=part, key=key)
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

    def data(self, dataset: str, *, why: str, part=None, ext: str = _sh.EXT,
             allow_missing: bool = False) -> _assets.Dataset:
        """Declare a dataset this pipeline writes, so that something can NAME it.

            XPM = iv.data("processed/xpm/", why="the player ratings")

        This says a dataset exists and how it is stored. It does not say how it is computed
        — some `@iv.step` does that, by putting this in its `output=`:

            @iv.step(output={"ratings": XPM, "career": XPM_CAREER}, why="the joint fit")
            def xpm(poss=iv.all_of(possessions, why="the design matrix")):
                return {"ratings": r, "career": c}

            def wvorp(x=iv.all_of(XPM, why="the headline table")):

        Most stages write ONE dataset, and a declaration on its own line would be a name
        used once — so `output="processed/box/"` declares it inline and the read names the
        stage. Reach for this when the name has to be said somewhere else: a stage writing
        several, or one whose output another stage names before it is written.
        """
        d = _assets.Dataset(_canon(dataset), ext, allow_missing,
                            _assets._fixed(part, _canon(dataset)),
                            _why(why, _canon(dataset)))
        if d.dataset in self._sources:
            raise DeclError(
                f"{d.dataset} was declared a source — something outside this pipeline puts "
                f"it there. It is one or the other.")
        self._datasets[d.dataset] = d
        return d

    def source(self, dataset: str, *, why: str, external=None) -> _assets.Source:
        """Declare a dataset that arrives from outside, so a read can name it.

            pbp = iv.source("raw/pbp_official/", why="the official play-by-play dump")

            @iv.step(output="derived/panel/", why="...", part="season")
            def panel(raw=iv.same_part(pbp, why="one season of it")):
                ...
        """
        src = _assets.Source(dataset, why=why, external=external)
        if src.dataset in self._datasets:
            raise DeclError(
                f"{src.dataset} was declared with iv.data(...) — a dataset this pipeline "
                f"writes — so it does not arrive from outside. It is one or the other.")
        if src.dataset in self._assets:
            raise DeclError(
                f"{src.dataset} is built by "
                f"{self._assets[src.dataset].__name__!r}, so it does not arrive from "
                f"outside. A dataset is declared once, as one thing or the other.")
        self._sources[src.dataset] = src
        return src

    def _register(self, asset: _assets.Asset) -> _assets.Asset:
        for name in asset.datasets:
            if name in self._sources:
                raise DeclError(
                    f"{name} was declared a source — something outside this pipeline puts "
                    f"it there — and {asset.__name__!r} writes it. It is one or the other.")
        for name in asset.datasets:
            for other in self._assets.values():
                if name not in other.datasets or other.fn is asset.fn:
                    continue
                # Two stages may share a dataset only by writing DIFFERENT shards of it —
                # three blocks of a college feature table, the played and unplayed halves
                # of a prediction table. Without a literal part= on both, whichever ran
                # last would simply win.
                mine, theirs = asset.part_for(name), other.part_for(name)
                if mine and theirs and mine != theirs:
                    continue
                raise DeclError(
                    f"{name} is already written by {other.__name__!r}. Two stages may "
                    f"share a dataset only by writing different partitions of it, each "
                    f"declared with a literal part=. Otherwise they race, and whichever "
                    f"runs last wins.")
        self._assets[self._node_name(asset.fn)] = asset
        for name in asset.datasets:
            self._declared[name] = asset.triples()
        return asset

    def producers_of(self, dataset: str) -> list:
        name = _canon(dataset)
        return [a for a in self._assets.values() if name in a.datasets]

    def step(self, output=None, *, why: str, part=None,
             ext: str = _sh.EXT, allow_missing: bool = False,
             if_needed: bool = True, once: bool = False, split: bool = False,
             external=None) -> Callable:
        """The function that BUILDS one or more datasets. Its upstreams are its parameters.

            @iv.step(output="processed/cohorts/", why="a fit per cohort", part="season")
            def cohorts(past=iv.before_part(features, why="prior seasons")):
                return past.group_by("player").agg(pl.col("z").mean())

        `output=` is:

          a path            one dataset, declared right here because nothing else needs to
                            name it — a read names `cohorts`, the stage. The body returns
                            its contents.
          an `iv.data(...)` the same, where the dataset was declared above so that reads
                            can name it directly.
          a dict            several, keyed by what the body returns. One expensive fit
                            produces six tables without being run six times.
          omitted           nothing lands in the tree — a fetch filling a download cache, a
                            publish copying out to a bucket. There is no artifact to be
                            stale against, so it runs every time.
        """
        _why(why, "step")

        def declared(fn: Callable) -> _assets.Asset:
            return self._register(_assets.Asset(
                self, output, fn, why=why, part=part, ext=ext,
                allow_missing=allow_missing, if_needed=if_needed, once=once,
                split=split, single=not isinstance(output, dict), external=external))
        return declared

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
        p = mkpath(filename, self.project)
        try:
            return str(p.relative_to(self.project)) if self.project else str(p)
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


def _span_parts(parts: list[str]) -> str:
    s = sorted(parts)
    return s[0] if len(s) == 1 else f"{s[0]}..{s[-1]}"


def _env_force() -> bool:
    return os.environ.get("IV_FORCE", "").lower() in ("1", "true", "yes")


def _abs_trace(trace):
    from pathlib import Path
    t = trace or os.environ.get("IV_TRACE")
    return Path(t).expanduser().resolve() if t else None


# `iv.PART` on the instance, so a selector reads as what it is at the call site.
Invalidator.PART = PART
