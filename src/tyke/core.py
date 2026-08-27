from __future__ import annotations

import builtins
import io
import os
import shutil
import threading
import time
from contextlib import ExitStack, contextmanager
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


try:
    from cloudpathlib import CloudPath as _CloudPath
except ImportError:
    _CloudPath = ()


def _path_text(target) -> str | None:

    if isinstance(target, int):
        return None
    if _CloudPath and isinstance(target, _CloudPath):
        return str(target)
    try:
        return os.fspath(target)
    except TypeError:
        return None


def _in_tree(target, base) -> bool:
    s = _path_text(target)
    if s is None or base is None:
        return False
    root = str(base).rstrip("/")
    return s == root or s.startswith(root + "/")


def _path_owners():
    from pathlib import Path as _P
    owners = [_P]
    try:
        from cloudpathlib import CloudPath
    except ImportError:
        return owners
    seen, stack = [], [CloudPath]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.append(c)
        stack.extend(c.__subclasses__())
    return owners + seen


_PROBING = threading.local()


def _check_declared(target, probe: str | None = None) -> None:


    s = _path_text(target)
    if s is None:
        return
    for tyke in _ACTIVE:
        if s in tyke._handed_out or tyke._depth:
            return
    for tyke in _ACTIVE:
        for base in (tyke.out_tree, tyke.tree):
            if not _in_tree(s, base):
                continue
            if probe is not None and not any(x._in_step for x in _ACTIVE):
                return
            if probe is None:
                raise DeclError(
                    f"{s} is inside the data tree but was not handed back by tyke.reads(). "
                    f"An undeclared read is absent from the graph and from the recorded "
                    f"inputs, so its source can change and this will never rebuild. "
                    f"Declare it: tyke.reads('<dataset>/', why='...').")
            raise DeclError(
                f"{s}.{probe}() asks the data tree a question about a path nobody "
                f"declared. A probe IS a read: branching on it lets a moved or missing "
                f"dataset come back as silently empty instead of rebuilding, which is "
                f"the failure declaring is meant to stop. Read it through the stage: "
                f"tyke.reads('<dataset>/', why='...'), or pass optional=True and branch "
                f"on what came back.")
    if any(x._in_step for x in _ACTIVE) and "://" in s and not _declared_external(s):
        what = f"{s}.{probe}()" if probe else s
        raise DeclError(
            f"{what} reaches remote storage outside every declared tree. Cloud storage "
            f"holds data, so a read of it belongs in the graph wherever it lives: "
            f"declare it with tyke.source('<dataset>/', why='...') and read it through "
            f"the stage. Undeclared, it can move or empty without anything rebuilding, "
            f"and the run has no record of what it actually read.")


def _declared_external(s: str) -> bool:
    for tyke in _ACTIVE:
        for name, _reason in tyke._declared_externals:
            if "://" in name and (s == name.rstrip("/")
                                  or s.startswith(name.rstrip("/") + "/")):
                return True
    return False


def _check_write(target, removing: bool = False) -> None:


    s = _path_text(target)
    if s is None:
        return
    for tyke in _ACTIVE:
        if tyke._depth or s in tyke._staged:
            return
    for tyke in _ACTIVE:
        if s in tyke._handed_out and not removing:
            raise DeclError(
                f"{s} was handed back by tyke.reads() and is being written to. A shard's "
                f"name is a fingerprint of its contents, so overwriting one in place makes "
                f"the name a lie that nothing can detect. Write through tyke.writes().")
        if tyke._in_step and "://" in s and not _declared_external(s) and not any(
                _in_tree(s, base) for base in (tyke.out_tree, tyke.tree)):
            raise DeclError(
                f"{s} is remote storage outside every declared tree and is being written "
                f"from inside a stage. A write that lands outside the graph has no "
                f"declared output and nothing downstream can key on it. Declare it as an "
                f"output, or as external= if it leaves the pipeline for good.")
        if tyke._in_step and any(_in_tree(s, base) for base in (tyke.out_tree, tyke.tree)):
            raise DeclError(
                f"{s} is inside the data tree and is being written outside tyke.writes(). "
                f"A direct write has no declared output, staged commit, or fingerprinted "
                f"name. Write it through tyke.writes(...), or write outside the data tree.")


@contextmanager
def _internal_io():

    with ExitStack() as stack:
        for tyke in _ACTIVE:
            stack.enter_context(tyke.bookkeeping())
        yield


def _stage_is_running() -> bool:
    return any(tyke._in_step and not tyke._depth for tyke in _ACTIVE)


class Pipeline:


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
        self._declared_externals: tuple = ()
        self._handed_out: set[str] = set()
        self._staged: set[str] = set()


        self._part: dict | None = None
        self._in_step = False
        _ACTIVE.append(self)
        self._enforce_writes()
        self._enforce_reads()
        self._node = ""
        self._inputs: tuple = ()
        self._outputs: tuple[str, ...] = ()


        self._declared: dict[str, tuple] = {}


        self._assets: dict[str, _assets.Asset] = {}


        self._sources: dict[str, _assets.Source] = {}


        self._datasets: dict[str, _assets.Dataset] = {}

        self._schemas: dict[str, tuple | None] = {}
        self._changes: set[tuple[str, str]] = set()


        self._versions: dict[str, str | None] = {}

    def __repr__(self) -> str:
        return f"<Pipeline {self.tree}>"


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
        live = [s for s in sel if not _sh.is_empty(s)]
        self._handed_out.update(str(s.path) for s in live)
        with self.bookkeeping():
            self._validate_schema(name, live)
        return [s.path for s in live]

    def _schema(self, dataset: str) -> tuple | None:
        return self._schemas.get(_canon(dataset))

    def _validate_schema(self, dataset: str, shards) -> None:

        contract = self._schema(dataset)
        if contract is None:
            return
        actual = [(s.part_str, _sh.schema_of_file(s.path)) for s in shards]
        bad = [(part, got) for part, got in actual if got != contract]
        if not bad:
            return
        parts = ", ".join(p or "(unpartitioned)" for p, _ in bad)
        raise StateError(
            f"{_canon(dataset)} has shard(s) outside its declared schema: {parts}. "
            f"This is a schema migration in progress: rebuild those shard(s) to the "
            f"schema declared in Python before reading them.")

    def _validate_staged_schema(self, dataset: str, staged) -> None:
        contract = self._schema(dataset)
        if contract is None:
            return
        actual = _sh.schema_of_file(staged)
        if actual != contract:
            raise DeclError(
                f"{_canon(dataset)} produced a Parquet schema different from its declared "
                f"schema. Expected {dict(contract)!r}; got {dict(actual)!r}.")

    def _check_updates_own(self, name: str) -> None:


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
        self._patch_openers()
        for owner in _path_owners():
            for name in ("write_text", "write_bytes", "open", "mkdir", "unlink",
                         "rmdir", "rename", "replace", "touch", "write_parquet"):
                fn = getattr(owner, name, None)
                if fn is None or getattr(fn, "_iv_checked", False):
                    continue
                setattr(owner, name, self._checked_write(owner, name, fn))
        for name in ("remove", "unlink", "rmdir", "mkdir", "makedirs",
                     "rename", "replace"):
            fn = getattr(os, name, None)
            if fn is None or getattr(fn, "_iv_checked", False):
                continue
            setattr(os, name, self._checked_os_mutation(name, fn))
        for name in ("copyfile", "copy", "copy2", "copytree", "move", "rmtree"):
            fn = getattr(shutil, name, None)
            if fn is None or getattr(fn, "_iv_checked", False):
                continue
            setattr(shutil, name, self._checked_shutil(name, fn))

    @staticmethod
    def _patch_openers() -> None:

        for owner, name in ((builtins, "open"), (io, "open"), (os, "open")):
            fn = getattr(owner, name)
            if getattr(fn, "_iv_checked", False):
                continue
            if owner is os:
                patched = Pipeline._checked_os_open(fn)
            else:
                patched = Pipeline._checked_open(fn)
            setattr(owner, name, patched)

    @staticmethod
    def _checked_write(owner, name, fn):
        def patched(target, *a, **kw):
            mode = a[0] if a else kw.get("mode", "r")
            if name != "open" or any(flag in mode for flag in ("w", "a", "x", "+")):
                _check_write(target, removing=name in ("unlink", "rmdir"))
            return fn(target, *a, **kw)
        patched._iv_checked = True
        return patched

    @staticmethod
    def _checked_frame_write(name, fn):
        def patched(frame, target, *a, **kw):
            _check_write(target)
            return fn(frame, target, *a, **kw)
        patched._iv_checked = True
        return patched

    @staticmethod
    def _checked_open(fn):
        def patched(file, *a, **kw):
            mode = a[0] if a else kw.get("mode", "r")
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                _check_write(file)
            else:
                _check_declared(file)
            return fn(file, *a, **kw)
        patched._iv_checked = True
        return patched

    @staticmethod
    def _checked_os_open(fn):
        def patched(path, flags, *a, **kw):
            writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)
            (_check_write if writing else _check_declared)(path)
            return fn(path, flags, *a, **kw)
        patched._iv_checked = True
        return patched

    @staticmethod
    def _checked_os_mutation(name, fn):
        def patched(path, *a, **kw):
            if not _stage_is_running():
                return fn(path, *a, **kw)
            _check_write(path)
            if name in ("rename", "replace") and a:
                _check_write(a[0])
            return fn(path, *a, **kw)
        patched._iv_checked = True
        return patched

    @staticmethod
    def _checked_shutil(name, fn):
        def patched(src, *a, **kw):
            if _stage_is_running():
                if name == "rmtree":
                    _check_write(src)
                elif a:
                    dst = a[0]
                    _check_declared(src)
                    _check_write(dst)
                    if name == "move":
                        _check_write(src)


            with _internal_io():
                return fn(src, *a, **kw)
        patched._iv_checked = True
        return patched

    def _enforce_reads(self) -> None:
        owners = [Path]
        try:
            from cloudpathlib import CloudPath
            owners.append(CloudPath)
        except ImportError:
            pass
        for owner in _path_owners():
            for name in ("read_text", "read_bytes", "open"):
                fn = getattr(owner, name, None)
                if fn is None or getattr(fn, "_iv_read_checked", False):
                    continue
                setattr(owner, name, self._checked_read(name, fn))
            for name in ("exists", "is_file", "is_dir", "iterdir", "glob", "rglob"):
                fn = getattr(owner, name, None)
                if fn is None or getattr(fn, "_iv_read_checked", False):
                    continue
                setattr(owner, name, self._checked_probe(name, fn))
        try:
            import polars as pl
        except ImportError:
            return
        for name in ("read_parquet", "scan_parquet", "read_csv", "read_parquet_schema"):
            fn = getattr(pl, name, None)
            if fn is None or getattr(fn, "_iv_read_checked", False):
                continue
            setattr(pl, name, self._checked_read(name, fn, first_arg=True))
        for owner, name in ((pl.DataFrame, "write_parquet"), (pl.DataFrame, "write_csv"),
                            (pl.DataFrame, "write_json"), (pl.LazyFrame, "sink_parquet")):
            fn = getattr(owner, name, None)
            if fn is None or getattr(fn, "_iv_checked", False):
                continue
            setattr(owner, name, self._checked_frame_write(name, fn))

    @staticmethod
    def _checked_probe(name, fn):
        def patched(target, *a, **kw):
            prev = getattr(_PROBING, "on", False)
            _PROBING.on = True
            try:
                if not prev:
                    _check_declared(target, probe=name)
                return fn(target, *a, **kw)
            finally:
                _PROBING.on = prev
        patched._iv_read_checked = True
        return patched

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


        return _sh.snapshot()

    def verify(self, dataset: str) -> list[str]:
        with self.bookkeeping():
            out = []
            d = self.resolve_out(dataset)
            live = _sh.current_shards(d)
            for part, shard in live.items():
                if _sh.is_empty(shard):
                    if shard.path.stat().st_size:
                        out.append(f"{shard.name}: named as an empty partition but the "
                                   f"file has contents")
                    continue
                actual = _sh.fingerprint_of_file(shard.path)
                if actual != shard.fp:
                    out.append(f"{shard.name}: contents fingerprint {actual}, name says "
                               f"{shard.fp} — the file was changed after it was committed")
            contract = self._schema(dataset)
            parq = {p: s for p, s in live.items()
                    if str(s.path).endswith(".parquet") and not _sh.is_empty(s)}
            by_schema = _sh.schemas_of(parq.values()) if parq else {}
            if contract is not None:
                bad = [part or "(unpartitioned)" for part, cols in
                       ((part, _sh.schema_of_file(shard.path)) for part, shard in parq.items())
                       if cols != contract]
                if bad:
                    out.append(
                        f"SCHEMA CONTRACT: {_canon(dataset)} has shard(s) outside the declared schema: "
                        f"{', '.join(sorted(bad))}.")
            elif len(by_schema) > 1:
                groups = sorted(by_schema.items(), key=lambda kv: -len(kv[1]))
                base = set(c for c, _ in groups[0][0])
                lines = []
                for cols, parts in groups[1:]:
                    d2 = set(c for c, _ in cols) ^ base
                    lines.append(f"{_span_parts(parts)} differ by {sorted(d2)}")
                out.append(f"SCHEMA DRIFT: {len(by_schema)} column sets across {len(parq)} "
                           f"shards — {_span_parts(groups[0][1])} is the majority; "
                           + "; ".join(lines)
                           + ". A read of the whole dataset cannot produce one frame.")
            return out

    def key_of(self, dataset: str, part: dict | None, inputs) -> str:


        pairs = []
        for name, sel, optional in inputs:
            if name == dataset:


                continue
            if sel is None:
                raise DeclError(
                    f"{name} is read with a where= this cannot read without running the "
                    f"stage, so the shard's key cannot be computed and nothing can decide "
                    f"whether to skip. Selectors have to be data: a literal, or tyke.PART for "
                    f"the partition being built.")
            live = _sh.current_shards(self.resolve(name))
            try:
                got = _sh.select(live, _resolve_sel(sel, part, name), dataset=name)
            except StateError:


                if optional:
                    continue
                raise
            if got:
                pairs.append((name, _sh.dataset_id(got)))
        contract = self._schema(dataset)
        version = self._versions.get(dataset)
        if not inputs and contract is None and version is None:
            return ""


        body = "|".join(f"{n}={i}" for n, i in sorted(set(pairs)))
        schema = _sh._short(repr(contract)) if contract is not None else ""
        return _sh._short(f"key:{dataset}|{_sh.encode_part(part)}|schema={schema}|version={version or ''}|{body}")


    @contextmanager
    def writes(self, dataset: str, *, why: str, part: dict | None = None,
               allow_missing: bool = False, ext: str = _sh.EXT):
        name = _canon(dataset)
        _why(why, name)
        part = self._part if part is None else part
        with self.bookkeeping():
            staged = _sh.stage(f"{_sh.encode_part(part) or 'all'}-{time.time_ns()}",
                               self.stage_dir, ext)
        self._staged.add(str(staged))
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
                out_dir = self.resolve_out(name)
                with self.bookkeeping():
                    _sh.commit_empty(
                        out_dir, part=part,
                        key=self.key_of(name, part, self._inputs), ext=ext)
                self.record("io", op="empty", rel=name, why=why,
                            part=_sh.encode_part(part))
                self._staged.discard(str(staged))
                if not self._in_step:
                    self._fresh_scope()
                return
            raise DeclError(
                f"{name} was declared written but nothing was written to {staged}. Pass "
                f"allow_missing=True if producing nothing is a legitimate outcome — then "
                f"the shard stays absent and the next run tries again.")
        with self.bookkeeping():
            try:
                self._validate_staged_schema(name, staged)
            except BaseException:
                if staged.exists():
                    staged.unlink()
                self._staged.discard(str(staged))
                raise


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
            _sh.commit(
                staged, out_dir, part=part, key=key,
                on_commit=lambda _, changed: self._changes.add(
                    (name, _sh.encode_part(part))) if changed else None,
            )
        self.record("io", op="write", rel=name, why=why, part=_sh.encode_part(part),
                    key=key, seconds=round(time.time() - started, 2))
        if not self._in_step:
            self._fresh_scope()
        self._staged.discard(str(staged))


    def why_stale(self, dataset: str, part: dict | None = None, *,
                  inputs=None) -> str | None:


        name = _canon(dataset)
        inputs = self._declared.get(name, ()) if inputs is None else inputs
        d = self.resolve_out(name)
        with self.bookkeeping():
            live = _sh.current_shards(d)
            if part is None and "" not in live:


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


                return None
            if got.key != want:
                return ("its inputs moved — the key in its name is not the one its declared "
                        "upstreams produce now")
        return None

    def is_current(self, dataset: str, part: dict | None = None, **kw) -> bool:
        return self.why_stale(dataset, part, **kw) is None


    all_of = staticmethod(_decl.all_of)
    same_part = staticmethod(_decl.same_part)
    before_part = staticmethod(_decl.before_part)
    after_part = staticmethod(_decl.after_part)
    between = staticmethod(_decl.between)
    parts = staticmethod(_decl.parts)
    own_last_copy = staticmethod(_decl.own_last_copy)

    def dataset(self, dataset: str, *, why: str, part=None, ext: str = _sh.EXT,
                allow_missing: bool = False, schema=None) -> _assets.Dataset:


        d = _assets.Dataset(_canon(dataset), ext, allow_missing,
                            _assets._fixed(part, _canon(dataset)),
                            _why(why, _canon(dataset)),
                            _assets.schema_contract(schema, ext, _canon(dataset)), standalone=True)
        if d.dataset in self._sources:
            raise DeclError(
                f"{d.dataset} was declared a source — something outside this pipeline puts "
                f"it there. It is one or the other.")
        if d.dataset in self._datasets:
            raise DeclError(
                f"{d.dataset} is already declared: {self._datasets[d.dataset].why!r}. A "
                f"dataset is declared ONCE and named everywhere else — that is the whole "
                f"point of declaring it. Two declarations are two why= lines that can "
                f"disagree, and a rename that gets one of them.")
        if d.dataset in self._declared:
            writer = next(a.__name__ for a in self._assets.values()
                          if d.dataset in a.datasets)
            raise DeclError(
                f"{d.dataset} is already declared by {writer!r}, which names it in its own "
                f"output=. Declare it here and have that stage name THIS — "
                f"output={dataset.strip('/').rsplit('/', 1)[-1].upper()} — or leave it "
                f"there and drop this line. Not both.")
        self._datasets[d.dataset] = d
        self._schemas[d.dataset] = d.schema
        return d

    def data(self, dataset, *, why: str, part=None, ext: str = _sh.EXT,
             allow_missing: bool = False, once: bool = False,
             split: bool = False, external=None, schema=None, version=None,
             universe=None) -> Callable:


        _why(why, _canon(dataset) if isinstance(dataset, str) else str(dataset))
        if isinstance(dataset, dict):
            raise DeclError(
                "@tyke.data builds ONE dataset and its body returns that dataset's "
                "contents. For several — one fit, six tables — @tyke.step(output=...) "
                "takes the dict and the body returns one keyed by the same names.")

        def declared(fn: Callable) -> _assets.Asset:
            return self._register(_assets.Asset(
                self, dataset, fn, why=why, part=part, ext=ext,
                allow_missing=allow_missing, once=once,
                split=split, single=True, external=external, schema=schema, version=version,
                universe=universe))
        return declared

    def source(self, dataset: str, *, why: str, external=None, schema=None) -> _assets.Source:


        src = _assets.Source(dataset, why=why, external=external, schema=schema)
        if src.dataset in self._sources:
            raise DeclError(
                f"{src.dataset} is already declared a source: "
                f"{self._sources[src.dataset].why!r}. A dataset is declared once.")
        if src.dataset in self._datasets:
            raise DeclError(
                f"{src.dataset} was declared with tyke.data(...) — a dataset this pipeline "
                f"writes — so it does not arrive from outside. It is one or the other.")
        if src.dataset in self._assets:
            raise DeclError(
                f"{src.dataset} is built by "
                f"{self._assets[src.dataset].__name__!r}, so it does not arrive from "
                f"outside. A dataset is declared once, as one thing or the other.")
        self._sources[src.dataset] = src
        self._schemas[src.dataset] = src.schema
        return src

    def _register(self, asset: _assets.Asset) -> _assets.Asset:
        for name, o in asset.outputs.items():
            if o.dataset in self._datasets and not o.standalone:
                raise DeclError(
                    f"{o.dataset} is already declared on its own line — "
                    f"{self._datasets[o.dataset].why!r} — and {asset.__name__!r} writes "
                    f"the path out again. Name the declaration instead, so the path is "
                    f"written once and a rename cannot get half of it.")
        for name in asset.datasets:
            if name in self._sources:
                raise DeclError(
                    f"{name} was declared a source — something outside this pipeline puts "
                    f"it there — and {asset.__name__!r} writes it. It is one or the other.")
        for name in asset.datasets:
            for other in self._assets.values():
                if name not in other.datasets or other.fn is asset.fn:
                    continue


                mine, theirs = asset.part_for(name), other.part_for(name)
                if mine and theirs and mine != theirs:
                    continue
                raise DeclError(
                    f"{name} is already written by {other.__name__!r}. Two stages may "
                    f"share a dataset only by writing different partitions of it, each "
                    f"declared with a literal part=. Otherwise they race, and whichever "
                    f"runs last wins.")
        for output in asset.outputs.values():
            known = self._schemas.get(output.dataset)
            if known is not None and output.schema is not None and known != output.schema:
                raise DeclError(
                    f"{output.dataset} has conflicting declared schemas. A dataset has one "
                    f"contract, so declare it once with tyke.dataset(..., schema=...).")
        self._assets[self._node_name(asset.fn)] = asset
        for name in asset.datasets:
            self._declared[name] = asset.triples()
            self._versions[name] = asset.version
        for output in asset.outputs.values():
            known = self._schemas.get(output.dataset)
            self._schemas[output.dataset] = output.schema if output.schema is not None else known
        return asset

    def producers_of(self, dataset: str) -> list:
        name = _canon(dataset)
        return [a for a in self._assets.values() if name in a.datasets]

    def step(self, output=None, *, why: str, part=None,
             ext: str = _sh.EXT, allow_missing: bool = False,
             once: bool = False, split: bool = False,
             external=None, version=None, universe=None) -> Callable:


        _why(why, "step")
        if output is not None and not isinstance(output, dict):
            raise DeclError(
                f"@tyke.step builds SEVERAL datasets — output= is a dict naming each, and "
                f"the body returns one keyed the same way. For one, @tyke.data(dataset="
                f"{output!r}) is the same stage with the body returning its contents.")

        def declared(fn: Callable) -> _assets.Asset:
            return self._register(_assets.Asset(
                self, output, fn, why=why, part=part, ext=ext,
                allow_missing=allow_missing, once=once,
                split=split, single=False, external=external, version=version,
                universe=universe))
        return declared

    def _node_name(self, fn: Callable) -> str:


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
        override = os.environ.get("TYKE_STAGE")
        if override:
            return override
        return self._rel_source(sys.argv[0] or "<repl>")

    def reset(self) -> None:
        self._fresh_scope()

def _sub_part(where: dict | None, part: dict | None, name: str):


    if not where:
        return where
    def one(v, k):
        if v != PART:
            return v
        if not part or k not in part:
            raise DeclError(
                f"{name} selects on tyke.PART for {k!r}, but this stage is not building a "
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

    if not sel:
        return None

    def one(v, k):
        if v != PART:
            return v
        if not part or k not in part:
            raise DeclError(
                f"{name} selects on tyke.PART for {k!r}, but this stage is not building a "
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
    return os.environ.get("TYKE_FORCE", "").lower() in ("1", "true", "yes")


def _abs_trace(trace):
    from pathlib import Path
    t = trace or os.environ.get("TYKE_TRACE")
    return Path(t).expanduser().resolve() if t else None


Pipeline.PART = PART
