"""A dataset and the function that builds it, as one object.

`@iv.data` names a dataset and decorates its producer. The upstreams are parameter
defaults, so the whole declaration — dataset, selector, partition key — is readable from
the function object with nothing executed and no source text needed.

Calling the asset is how it gets built. It builds if it is stale and loads if it is not,
which is the same decision `@iv.step` makes, made per shard.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

from . import decl as _decl
from . import shards as _sh
from .errors import DeclError, StateError


# ── turning a returned value into a file, and back ────────────────────────────
#
# The rule is ROUND TRIP: what a cached call hands back must be what the producer returned.
# A dict written to parquet comes back a DataFrame, so the same function gives two types
# depending on whether the shard happened to be current — a difference that shows up as a
# TypeError days later, in the branch that was cached. So a format that cannot return the
# value it was given refuses it and names one that can.

def save_value(value: object, path, ext: str) -> None:
    if ext == ".parquet":
        import polars as pl
        if isinstance(value, pl.LazyFrame):
            value = value.collect()
        if not isinstance(value, pl.DataFrame):
            raise DeclError(
                f"a .parquet dataset is written from a polars DataFrame, got "
                f"{type(value).__name__}. Reading it back would hand you a DataFrame "
                f"rather than what you returned, so the same function would give two "
                f"different types. Return a DataFrame, or declare the format that round "
                f"trips: ext='.json' for a dict or list, ext='.pkl' for anything else.")
        value.write_parquet(str(path))
    elif ext == ".json":
        try:
            body = json.dumps(value, indent=2, sort_keys=True)
        except TypeError as e:
            raise DeclError(
                f"a .json dataset is written from something json can represent, and "
                f"{type(value).__name__} is not: {e}. Use ext='.pkl' for arbitrary "
                f"objects.") from e
        Path(str(path)).write_text(body)
    elif ext == ".pkl":
        import pickle
        Path(str(path)).write_bytes(pickle.dumps(value))
    elif ext in (".html", ".txt"):
        if not isinstance(value, str):
            raise DeclError(
                f"a {ext} dataset is written from a str, got {type(value).__name__}.")
        Path(str(path)).write_text(value)
    else:
        raise DeclError(
            f"no way to write a {ext!r} dataset from a returned value. Known: "
            f"{sorted(_WRITERS)}. Take an `out` parameter and write the file yourself.")


_WRITERS = (".parquet", ".json", ".pkl", ".html", ".txt")


def load_value(paths, ext: str):
    """The other half of the round trip. `paths` is what `Pipeline.reads` handed back."""
    if ext == ".parquet":
        import polars as pl
        return pl.read_parquet([str(p) for p in paths])
    if len(paths) != 1:
        raise StateError(
            f"a {ext} dataset was selected across {len(paths)} shards. Only .parquet is "
            f"concatenated on read; everything else is one shard at a time — select a "
            f"single partition, or store it as parquet.")
    p = Path(str(paths[0]))
    if ext == ".json":
        return json.loads(p.read_text())
    if ext == ".pkl":
        import pickle
        return pickle.loads(p.read_bytes())
    if ext in (".html", ".txt"):
        return p.read_text()
    return p


# ── what a signature declares ─────────────────────────────────────────────────

OUT_PARAM = "out"


def declared_reads(fn, part_key: str | None) -> tuple:
    """The `Read` defaults in a signature, each bound to the stage's partition key."""
    out = []
    for p in inspect.signature(fn).parameters.values():
        if isinstance(p.default, _decl.Read):
            out.append(p.default.bound_to(part_key))
    return tuple(out)


def has_declared_reads(fn) -> bool:
    return any(isinstance(p.default, _decl.Read)
               for p in inspect.signature(fn).parameters.values())


def check_signature(fn, part_key: str | None, dataset: str) -> None:
    """Every parameter has to be something this can supply, or the call would fail late."""
    for name, p in inspect.signature(fn).parameters.items():
        if isinstance(p.default, _decl.Read) or name == OUT_PARAM:
            continue
        if part_key is not None and name == part_key:
            continue
        if p.default is not inspect.Parameter.empty:
            continue
        known = ["a read (iv.all_of(...) and friends)", repr(OUT_PARAM)]
        if part_key:
            known.append(repr(part_key))
        raise DeclError(
            f"{dataset}: parameter {name!r} of {fn.__name__} is not something iv can "
            f"supply. A parameter is one of: {', '.join(known)}, or has a default. "
            f"A partition value is named after the stage's part=.")


class Asset:
    """One dataset, its producer, and the decision about whether to run it."""

    def __init__(self, pipeline, dataset: str, fn, *, why: str,
                 part: str | None = None, ext: str = _sh.EXT,
                 terminal: bool = False, if_needed: bool = True,
                 once: bool = False) -> None:
        self.pipeline = pipeline
        self.dataset = _decl._canon(dataset)
        self.fn = fn
        self.why = _decl._why(why, self.dataset)
        self.part_key = part
        self.ext = ext
        self.terminal = terminal
        self.if_needed = if_needed
        self.once = once
        check_signature(fn, part, self.dataset)
        self.reads = declared_reads(fn, part)
        self.wants_out = OUT_PARAM in inspect.signature(fn).parameters
        self.__name__ = getattr(fn, "__name__", "asset")
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__wrapped__ = fn

    def __repr__(self) -> str:
        p = f", part={self.part_key!r}" if self.part_key else ""
        return f"<iv.data {self.dataset}{p}>"

    # ── what the graph and the skip check read off it ─────────────────────────

    @property
    def triggers(self) -> tuple:
        """The reads that can make it stale. An own-copy read is lineage, not a trigger."""
        return tuple(r for r in self.reads if not r.is_own)

    def triples(self) -> tuple:
        """`(dataset, sel, optional)` per trigger — what `key_of` consumes."""
        return tuple(r.triple() for r in self.triggers)

    @property
    def may_skip(self) -> bool:
        """A ROOT — no declared upstream — always runs, and that is not an oversight.

        Its body is the only thing that knows about the world outside the tree: the
        fetch, the clock, the hyperparameters someone just edited. Nothing on disk can
        say whether that has moved, so `why_stale` has no question to ask and answers
        `current` forever. Skip on that and the pipeline is sealed shut — it serves the
        first run's numbers and never notices anything again.

        Running it is the safe failure: the commit is content-addressed, so a body that
        produces the same bytes commits the same shard, and nothing downstream moves. A
        fetch too expensive to repeat says so with once=True, and then it is the caller
        who has decided that nothing new can arrive.
        """
        return bool(self.triggers) or self.once

    # ── building ──────────────────────────────────────────────────────────────

    def _part(self, args, kwargs) -> dict | None:
        if self.part_key is None:
            if args or kwargs:
                raise DeclError(
                    f"{self.dataset} is not partitioned, so it takes no partition value. "
                    f"Declare one with @iv.data(..., part='season').")
            return None
        if self.part_key in kwargs:
            return {self.part_key: str(kwargs[self.part_key])}
        if len(args) == 1:
            return {self.part_key: str(args[0])}
        raise DeclError(
            f"{self.dataset} is partitioned by {self.part_key!r}, so a call names one: "
            f"{self.__name__}('2024') or {self.__name__}({self.part_key}='2024').")

    def why_stale(self, *args, **kwargs) -> str | None:
        return self.pipeline.why_stale(self.dataset, self._part(args, kwargs),
                                       inputs=self.triples())

    def is_current(self, *args, **kwargs) -> bool:
        return self.why_stale(*args, **kwargs) is None

    def __call__(self, *args, **kwargs):
        """Build this shard if it is stale, and hand back its contents either way."""
        iv = self.pipeline
        if iv._in_step:
            raise DeclError(
                f"{self.dataset} was called from inside another stage. Building one stage "
                f"from within another puts it outside the graph, so nothing orders the two "
                f"and `iv status` cannot see the edge. Declare it as an upstream instead: "
                f"def {iv._node.split('::')[-1] or 'stage'}(x=iv.all_of({self.dataset!r}, "
                f"why='...')).")
        part = self._part(args, kwargs)
        if (self.if_needed and self.may_skip and not iv.force
                and self.why_stale(*args, **kwargs) is None):
            return self.load(part)
        self.build(part)
        return self.load(part)

    def build(self, part: dict | None) -> None:
        """Run the body and commit what it produced. No skip check — the caller made it."""
        iv = self.pipeline
        iv._fresh_scope()
        prev = (iv._part, iv._in_step, iv._node, iv._inputs, iv._outputs)
        iv._part, iv._in_step = part, True
        iv._node = iv._node_name(self.fn)
        iv._inputs, iv._outputs = self.triples(), (self.dataset,)
        try:
            kw = self._resolve(part)
            with iv.writes(self.dataset, why=self.why, part=part, ext=self.ext,
                           terminal=self.terminal) as staged:
                if self.wants_out:
                    kw[OUT_PARAM] = staged
                    self.fn(**kw)
                else:
                    value = self.fn(**kw)
                    if value is None:
                        raise DeclError(
                            f"{self.dataset}: {self.__name__} returned None and takes no "
                            f"{OUT_PARAM!r} parameter, so nothing was produced. Return the "
                            f"value to store, or take `{OUT_PARAM}` and write to it.")
                    save_value(value, staged, self.ext)
        finally:
            (iv._part, iv._in_step, iv._node, iv._inputs, iv._outputs) = prev
            iv._fresh_scope()

    def _resolve(self, part: dict | None) -> dict:
        """Each declared read, opened — through `Pipeline.reads`, so the recording, the
        enforcement and the undeclared-read check all apply exactly as they always have."""
        from .core import _resolve_sel
        iv = self.pipeline
        kw: dict = {}
        if self.part_key is not None and self.part_key in inspect.signature(self.fn).parameters:
            kw[self.part_key] = part[self.part_key]
        for name, p in inspect.signature(self.fn).parameters.items():
            if not isinstance(p.default, _decl.Read):
                continue
            r = p.default.bound_to(self.part_key)
            paths = iv.reads(r.dataset, why=r.why,
                             where=_resolve_sel(r.sel(), part, r.dataset),
                             optional=r.optional, update_file_on_disk=r.is_own)
            kw[name] = load_value(paths, _ext_of(paths, self.ext)) if paths else None
        return kw

    def load(self, part: dict | None = None):
        """The contents of one shard, without deciding anything.

        The whole read happens under `bookkeeping`, which exempts it from the
        undeclared-read check for the duration and no longer. Whitelisting the path
        instead would exempt it FOREVER — and then any later stage could open that file
        behind iv's back, which is the exact thing the check exists to refuse.
        """
        iv = self.pipeline
        with iv.bookkeeping():
            live = _sh.current_shards(iv.resolve_out(self.dataset))
            got = live.get(_sh.encode_part(part))
            if got is None:
                raise StateError(
                    f"{self.dataset} has no shard for "
                    f"{_sh.encode_part(part) or '(one shard)'} — it was not built.")
            return load_value([got.path], got.ext)

    def for_each(self, over) -> list[str]:
        """Build one shard per key, skipping the ones already current."""
        if self.part_key is None:
            raise DeclError(
                f"{self.dataset} is not partitioned, so there is nothing to iterate. "
                f"Declare @iv.data(..., part='season').")
        iv = self.pipeline
        want = [str(p) for p in over]
        rebuild = [p for p in want
                   if iv.force or not self.may_skip or self.why_stale(p) is not None]
        for p in rebuild:
            self.build({self.part_key: p})
        return rebuild


def _ext_of(paths, default: str) -> str:
    suffixes = {Path(str(p)).suffix for p in paths}
    return suffixes.pop() if len(suffixes) == 1 else default
