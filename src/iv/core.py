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
        self._reads: dict[str, dict] = {}
        self._prior: set[str] = set()
        self._real: set[str] = set()
        self._externals: list[str] = []
        self._part: dict | None = None
        self._code: str = ""
        self._in_step = False

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
        self._reads, self._prior, self._real, self._externals = {}, set(), set(), []


    def constants(self, dataset: str, *, why: str, **values):
        import polars as pl
        if not values:
            raise DeclError(f"{dataset} needs at least one value — that is what it is for.")
        self._fresh_scope()
        with self.writes(dataset, why=why) as out:
            pl.DataFrame({k: [v] for k, v in values.items()}).write_parquet(out)


    def reads(self, dataset: str, *, why: str, where: dict | None = None,
              optional: bool = False, prior: bool = False) -> list:
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
            real = name not in self._prior or not prior
            if real or name not in self._reads:
                self._reads[name] = {
                    "where": where,
                    "parts": [s.part_str for s in sel],
                    "id": _sh.dataset_id(sel),
                }
            if prior and name not in self._real:
                self._prior.add(name)
            if not prior:
                self._real.add(name)
                self._prior.discard(name)
            self.record("io", op="read", rel=name, why=why, n=len(sel), prior=prior)
        return [s.path for s in sel]

    def external(self, name: str, *, why: str) -> None:
        _why(why, name)
        if not self._depth:
            self._externals.append(name)
            self.record("io", op="external", rel=f"external:{name}", why=why)

    def _inputs_now(self) -> dict[str, dict]:
        return {name: dict(v) for name, v in self._reads.items() if name not in self._prior}


    @contextmanager
    def writes(self, dataset: str, *, why: str, part: dict | None = None,
               terminal: bool = False, code: str | None = None,
               allow_missing: bool = False):
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
            self._fresh_scope()


    def why_stale(self, dataset: str, part: dict | None = None,
                  *, code: str | None = None) -> str | None:
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
        out: dict[str, str] = {}
        for name, was in recorded.items():
            live = _sh.current_shards(self.resolve(name))
            try:
                sel = _sh.select(live, was.get("where"), dataset=name)
            except StateError as e:
                return out, f"{name}: {e}"
            out[name] = _sh.dataset_id(sel)
        return out, ""

    def is_current(self, dataset: str, part: dict | None = None, **kw) -> bool:
        return self.why_stale(dataset, part, **kw) is None


    def step(self, *, why: str, part: dict | None = None,
             code: bool | None = None, if_needed: bool = True) -> Callable:
        _why(why, "step")

        def decorate(fn: Callable) -> Callable:
            code_hash = "" if code is False else source_digest(fn)
            outputs = writes_in(fn)

            def wrapper(*args, **kwargs):
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
                    self._fresh_scope()

            wrapper.__name__ = getattr(fn, "__name__", "step")
            wrapper.__doc__ = fn.__doc__
            wrapper.run = fn
            wrapper.outputs = outputs
            return wrapper
        return decorate


    def for_each(self, over, build_one: Callable, *, dataset: str, key: str, why: str,
                 code: bool | None = None, quiet: bool = False) -> list[str]:
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


    def node(self) -> str:
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


def source_digest(fn: Callable) -> str:
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
