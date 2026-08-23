"""Datasets, and the stages that produce them.

Two words, and they are not the same word twice. `iv.data(...)` DECLARES a dataset: a name,
a format, a line on what it is for. `@iv.step` PRODUCES one or several, and its upstreams
are parameter defaults, so the whole declaration — datasets, selectors, partition key — is
readable from the function object with nothing executed and no source text needed.

Most stages write one dataset and declare it inline, which is a string in `output=`. The
declaration gets its own line when something else needs to NAME it: a stage writing four
tables from one fit, where a read wants to say WHICH.

Calling the asset is how it gets built. It builds if it is stale and loads if it is not,
made per shard.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from . import decl as _decl
from . import shards as _sh
from .errors import DeclError, StateError
from .paths import mkpath


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
    """The other half of the round trip. `paths` is what `Invalidator.reads` handed back."""
    if ext == ".parquet":
        import polars as pl
        return pl.read_parquet([str(p) for p in paths])
    if len(paths) != 1:
        raise StateError(
            f"a {ext} dataset was selected across {len(paths)} shards. Only .parquet is "
            f"concatenated on read; everything else is one shard at a time — select a "
            f"single partition, or store it as parquet.")
    p = mkpath(paths[0], None)
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


def declared_reads(fn, part_key: str | None, own: str | None = None) -> dict:
    """The `Read` defaults in a signature, by parameter name, bound to what the stage says
    about itself.

    Two things are filled in here rather than written twice: the partition key a relative
    selector applies to, and the output an `own_last_copy()` means — both of which the
    decorator already said, three lines above the signature. Returned BY PARAMETER so the
    body is handed the same reads the graph was told about; deriving them twice is how a
    resolved one and an unresolved one end up in the same stage.
    """
    return {name: p.default.bound_to(part_key).against(own)
            for name, p in inspect.signature(fn).parameters.items()
            if isinstance(p.default, _decl.Read)}


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


@dataclass(frozen=True)
class Dataset:
    """A dataset this pipeline writes: a name, a format, and what it is for.

    What `iv.data(...)` returns. It is a DECLARATION and nothing more — it says a dataset
    exists and how it is stored, not how it is computed. Some `@iv.step` produces it, and
    reads name it.

    `part` is the literal shard THIS dataset gets out of the stage, where it differs from
    the stage's own. One computation can produce a whole dataset and one block of a shared
    one.
    """
    dataset: str
    ext: str = _sh.EXT
    allow_missing: bool = False
    part: tuple = ()
    why: str = ""
    #: True when this came from `iv.dataset(...)` — a declaration on its own line, which a
    #: stage names rather than restates. False when a stage declared the path inline in its
    #: own `output=`, which is the one-writer case. The difference is what lets a second
    #: declaration of the same path be refused: naming `X` is one declaration used twice,
    #: writing "processed/x/" again is two declarations that can disagree.
    standalone: bool = False

    def shard(self, **part) -> "Dataset":
        """The one shard of this dataset a stage writes, where the stage cannot say it.

            @iv.step(output={"raw": INTL_RAW,
                             "features": COLLEGE_FEATURES.shard(source="intl")}, ...)

        A stage writing ONE dataset says which shard with its own `part=`. A stage writing
        several cannot: `part=` there would claim the same shard of every output, and the
        stage above writes all of `derived_data/intl/` and one block of a table three
        stages share. So the claim moves to the output it is about.

        It is not a second declaration — the dataset is declared once, and this is that
        declaration with the shard named.
        """
        if not part:
            raise DeclError(
                f"{self.dataset}: shard() names the literal partition this stage writes — "
                f"shard(source='intl').")
        return Dataset(self.dataset, self.ext, self.allow_missing,
                       _fixed(part, self.dataset), self.why, self.standalone)

    def __repr__(self) -> str:
        p = " " + ",".join(f"{k}={v}" for k, v in self.part) if self.part else ""
        return f"<iv dataset {self.dataset}{p}>"


def _outputs(spec, ext, allow_missing) -> dict:
    """`output=` as {key: Dataset}.

    A bare string is one dataset declared where it is produced, which is the common case
    and reads as one line. A `Dataset` is one declared on its own, which is what a stage
    writing several wants so that a read can name each. A dict is several, keyed by what
    the body returns.

    None is a stage that writes NOTHING — a fetch that fills a download cache, a publish
    that uploads. There is no artifact to be stale, so there is nothing to skip on and it
    runs every time. Its reads are still declared, so the graph draws the edges.
    """
    if spec is None:
        return {}
    if isinstance(spec, (str, Dataset, Source)):
        d = _dataset(spec, ext, allow_missing)
        return {d.dataset: d}
    if not isinstance(spec, dict) or not spec:
        raise DeclError(
            "output= is one dataset — a path, or an iv.data(...) declared above — or a "
            'dict of {name: dataset} naming several: output={"ratings": XPM, '
            '"summary": XPM_SUMMARY}. The names are the keys the body returns.')
    return {k: _dataset(v, ext, allow_missing) for k, v in spec.items()}


def _fixed(part, dataset) -> tuple:
    """The literal shard a declaration owns, as a sorted tuple."""
    if not part:
        return ()
    if not isinstance(part, dict):
        raise DeclError(
            f"{dataset}: part= on a declaration is the literal shard it owns — "
            f"part={{'source': 'ncaa'}}. Got {part!r}.")
    return tuple(sorted((str(k), str(v)) for k, v in part.items()))


def _dataset(v, ext, allow_missing) -> Dataset:
    if isinstance(v, Dataset):
        return Dataset(_decl._canon(v.dataset), v.ext, v.allow_missing, v.part, v.why,
                       v.standalone)
    if isinstance(v, Source):
        raise DeclError(
            f"{v.dataset} was declared a source — something outside this pipeline puts it "
            f"there — so no stage here writes it. Declare it with iv.data(...) instead.")
    return Dataset(_decl._canon(v), ext, allow_missing)


class Source:
    """A dataset that arrives from OUTSIDE: a fetch, a hand-placed file, another system.

    Nothing here builds it, so there is no body and no skip check — but it is declared all
    the same, so a read can name it and the graph knows every dataset the pipeline has
    rather than only the ones something here writes. That is what replaced a `sources=`
    list of path prefixes on the Invalidator: a prefix said "do not ask what writes
    anything under here", which is a rule about paths, where this is a statement about one
    dataset that can be pointed at.
    """

    def __init__(self, dataset: str, *, why: str, external=None) -> None:
        self.dataset = _decl._canon(dataset)
        self.why = _decl._why(why, self.dataset)
        self.externals = _externals(external, self.dataset)
        self.__name__ = self.dataset.rstrip("/").rsplit("/", 1)[-1]

    def __repr__(self) -> str:
        return f"<iv source {self.dataset}>"


class Asset:
    """A stage: what it reads, what it writes, and whether it needs to run.

    One object for all three shapes. `output=` naming ONE dataset is the common case, where
    the body returns its contents; a dict returns a dict keyed by those names, which is how
    one expensive fit produces six tables without being run six times; nothing at all is a
    stage that writes outside the tree.
    """

    def __init__(self, pipeline, output, fn, *, why: str,
                 part=None, ext: str = _sh.EXT, allow_missing: bool = False, if_needed: bool = True,
                 once: bool = False, split: bool = False, single: bool = True,
                 external=None) -> None:
        self.pipeline = pipeline
        self.outputs = _outputs(output, ext, allow_missing)
        self.single = single
        self.fn = fn
        self.acts_only = not self.outputs
        self.why = _decl._why(why, self.primary)
        self.if_needed = if_needed
        self.once = once
        self.split = split
        self.part_key, self.fixed_part = _part_spec(part, self.primary)
        if split and not self.part_key:
            raise DeclError(
                f"{self.primary}: split=True means the body computes every partition at "
                f"once and returns {{partition: value}}, so it needs a partition key — "
                f"@iv.step(..., part='season', split=True).")
        check_signature(fn, self.part_key, self.primary)
        if self.acts_only and (self.split or self.part_key or self.fixed_part):
            raise DeclError(
                f"{self.primary} writes nothing, so it has no shard for part= to name.")
        self.by_param = declared_reads(fn, self.part_key, self._own_output(fn))
        self.reads = tuple(self.by_param.values())
        self.externals = _externals(external, self.primary)
        self.wants_out = OUT_PARAM in inspect.signature(fn).parameters
        if self.wants_out and not self.single:
            raise DeclError(
                f"{self.primary}: a body that writes through `{OUT_PARAM}` produces one "
                f"file, so it cannot serve several outputs.")
        self.__name__ = getattr(fn, "__name__", "asset")
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__wrapped__ = fn

    @property
    def primary(self) -> str:
        if not self.outputs:
            return getattr(self.fn, "__name__", "<stage>")
        return next(iter(self.outputs.values())).dataset

    def __getitem__(self, key: str):
        """One named output of a multi-output stage, by the key the body returns it under.

        Usually the outputs are declared above and a read names the declaration directly,
        which is the same thing said without needing to know the key:

            XPM = iv.data("processed/xpm/", why="the player ratings")

            @iv.step(output={"ratings": XPM, ...}, why="the joint fit")
            def xpm(...): ...

            def wvorp(x=iv.all_of(XPM, why="the headline table")):
        """
        if key not in self.outputs:
            raise DeclError(
                f"{self.__name__} has no output named {key!r}: {sorted(self.outputs)}.")
        return self.outputs[key]

    @property
    def dataset(self) -> str:
        """What `iv.all_of(this_stage, ...)` means: the dataset it writes.

        A stage writing several has no single answer, so it says so rather than picking
        the first — `iv.all_of("processed/xpm/", ...)` names which.
        """
        if len(self.outputs) != 1:
            raise DeclError(
                f"{self.__name__} writes {len(self.outputs)} datasets, so naming the stage "
                f"in a read does not say which. Name the output by the key the body "
                f"returns it under — {self.__name__}[{sorted(self.outputs)[0]!r}], one of "
                f"{sorted(self.outputs)} — or, better where the read is far from here, "
                f"declare it above so the read can name the DATASET: "
                f"X = iv.data('...', why='...'), then output={{...}} and iv.all_of(X, ...). "
                f"A key means something only next to the dict it is a key of.")
        return self.primary

    @property
    def datasets(self) -> tuple:
        return tuple(o.dataset for o in self.outputs.values())

    def _own_output(self, fn) -> str | None:
        """What a bare `own_last_copy()` in this signature means: the one dataset written.

        None when there are several, so `Read.against` can say so rather than pick.
        """
        return self.primary if len(self.outputs) == 1 else None

    def part_for(self, dataset: str) -> tuple:
        """The literal shard this stage owns OF THIS DATASET, if it owns one.

        An output may name its own, which is what lets one computation write a whole
        dataset and one block of a shared one.
        """
        for o in self.outputs.values():
            if o.dataset == dataset:
                if o.part:
                    return o.part
                break
        return tuple(sorted(self.fixed_part.items())) if self.fixed_part else ()

    def __repr__(self) -> str:
        p = f", part={self.part_key or self.fixed_part!r}" if (self.part_key or self.fixed_part) else ""
        return f"<iv stage {', '.join(self.datasets)}{p}>"

    # ── what the graph and the skip check read off it ─────────────────────────

    @property
    def triggers(self) -> tuple:
        """The reads that can make it stale. An own-copy read is lineage, not a trigger."""
        return tuple(r for r in self.reads if not r.is_own)

    def triples(self) -> tuple:
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
        fetch too expensive to repeat says once=True, and then it is the caller who has
        decided that nothing new can arrive.
        """
        return bool(self.outputs) and (bool(self.triggers) or self.once)

    # ── deciding ──────────────────────────────────────────────────────────────

    def _part(self, args, kwargs) -> dict | None:
        if self.fixed_part is not None:
            if args or kwargs:
                raise DeclError(
                    f"{self.primary} writes the fixed partition {self.fixed_part}, so a "
                    f"call names no other.")
            return dict(self.fixed_part)
        if self.part_key is None or self.split:
            if args or kwargs:
                raise DeclError(
                    f"{self.primary} is not built one partition at a time, so it takes no "
                    f"partition value.")
            return None
        if self.part_key in kwargs:
            return {self.part_key: str(kwargs[self.part_key])}
        if len(args) == 1:
            return {self.part_key: str(args[0])}
        raise DeclError(
            f"{self.primary} is partitioned by {self.part_key!r}, so a call names one: "
            f"{self.__name__}('2024') or {self.__name__}({self.part_key}='2024').")

    def why_stale(self, *args, **kwargs) -> str | None:
        """Stale if ANY output is. Losing one table of a six-table fit brings the fit back."""
        part = self._part(args, kwargs)
        if self.acts_only:
            # Nothing was written, so there is nothing on disk to compare a key against.
            return "writes nothing, so nothing can say it is done"
        for o in self.outputs.values():
            r = self.pipeline.why_stale(o.dataset, dict(o.part) if o.part else part,
                                        inputs=self.triples())
            if r:
                return f"{o.dataset}: {r}" if len(self.outputs) > 1 else r
        return None

    def is_current(self, *args, **kwargs) -> bool:
        return self.why_stale(*args, **kwargs) is None

    def __call__(self, *args, **kwargs):
        iv = self.pipeline
        if iv._in_step:
            raise DeclError(
                f"{self.primary} was called from inside another stage. Building one stage "
                f"from within another puts it outside the graph, so nothing orders the two "
                f"and `iv status` cannot see the edge. Declare it as an upstream instead: "
                f"x=iv.all_of({self.primary!r}, why='...').")
        part = self._part(args, kwargs)
        if (self.if_needed and self.may_skip and not iv.force
                and self.why_stale(*args, **kwargs) is None):
            return self.load(part) if self.single and not self.split else False
        self.build(part)
        if self.acts_only:
            return True
        return self.load(part) if self.single and not self.split else True

    def build(self, part: dict | None) -> None:
        """Run the body and commit what it produced. No skip check — the caller made it."""
        iv = self.pipeline
        iv._fresh_scope()
        prev = (iv._part, iv._in_step, iv._node, iv._inputs, iv._outputs)
        iv._part, iv._in_step = part, True
        iv._node = iv._node_name(self.fn)
        iv._inputs, iv._outputs = self.triples(), self.datasets
        try:
            kw = self._resolve(part)
            if self.wants_out:
                o = next(iter(self.outputs.values()))
                with iv.writes(o.dataset, why=self.why, part=part, ext=o.ext,
                               allow_missing=o.allow_missing) as staged:
                    kw[OUT_PARAM] = staged
                    self.fn(**kw)
                return
            if self.acts_only:
                self.fn(**kw)
                return
            value = self.fn(**kw)
            if value is None:
                raise DeclError(
                    f"{self.primary}: {self.__name__} returned None and takes no "
                    f"{OUT_PARAM!r} parameter, so nothing was produced. Return the value "
                    f"to store, or take `{OUT_PARAM}` and write to it.")
            if self.split:
                self._commit_split(value)
            elif self.single:
                self._commit(next(iter(self.outputs.values())), value, part)
            else:
                self._commit_many(value, part)
        finally:
            (iv._part, iv._in_step, iv._node, iv._inputs, iv._outputs) = prev
            iv._fresh_scope()

    def _commit(self, o: Dataset, value, part) -> None:
        part = dict(o.part) if o.part else part
        with self.pipeline.writes(o.dataset, why=self.why, part=part, ext=o.ext,
                                     allow_missing=o.allow_missing) as staged:
            if value is not None:
                save_value(value, staged, o.ext)

    def _commit_many(self, value, part) -> None:
        if not isinstance(value, dict):
            raise DeclError(
                f"{self.__name__} declares {len(self.outputs)} outputs, so it returns a "
                f"dict keyed by their names — {sorted(self.outputs)} — got "
                f"{type(value).__name__}.")
        missing = sorted(set(self.outputs) - set(value))
        extra = sorted(set(value) - set(self.outputs))
        if extra:
            raise DeclError(
                f"{self.__name__} returned {extra}, which it does not declare as outputs. "
                f"Declared: {sorted(self.outputs)}.")
        for key, o in self.outputs.items():
            if key in missing and not o.allow_missing:
                raise DeclError(
                    f"{self.__name__} declares the output {key!r} ({o.dataset}) and did "
                    f"not return it. Pass allow_missing=True if producing nothing there "
                    f"is a legitimate outcome — then the shard stays absent and the next "
                    f"run tries again.")
            if key in value:
                self._commit(o, value[key], part)

    def _commit_split(self, value) -> None:
        """One computation, many shards.

        With one output the body returns {partition: value}. With several it returns
        {output: {partition: value}} — a walk-forward evaluation computes team, possession
        and player accuracy in one pass and cuts each by season.
        """
        shape = (f"{{{self.part_key}: value}}" if self.single
                 else f"{{output: {{{self.part_key}: value}}}}, one per {sorted(self.outputs)}")
        if not isinstance(value, dict):
            raise DeclError(
                f"{self.primary}: split=True means the body returns {shape} for every "
                f"partition it computed, got {type(value).__name__}.")
        if self.single:
            o = next(iter(self.outputs.values()))
            for key, v in value.items():
                self._commit(o, v, {self.part_key: str(key)})
            return
        extra = sorted(set(value) - set(self.outputs))
        if extra:
            raise DeclError(
                f"{self.__name__} returned {extra}, which it does not declare as outputs. "
                f"Declared: {sorted(self.outputs)}.")
        for key, o in self.outputs.items():
            by_part = value.get(key)
            if by_part is None:
                if o.allow_missing:
                    continue
                raise DeclError(
                    f"{self.__name__} declares the output {key!r} ({o.dataset}) and did "
                    f"not return it. Pass allow_missing=True if producing nothing there "
                    f"is legitimate.")
            if not isinstance(by_part, dict):
                raise DeclError(
                    f"{self.__name__} returned {key!r} as {type(by_part).__name__}; with "
                    f"split=True each output is {{{self.part_key}: value}}.")
            for part_val, v in by_part.items():
                self._commit(o, v, {self.part_key: str(part_val)})

    def _resolve(self, part: dict | None) -> dict:
        """Each declared read, opened — through `Invalidator.reads`, so the recording, the
        enforcement and the undeclared-read check all apply exactly as they always have."""
        from .core import _resolve_sel
        iv = self.pipeline
        params = inspect.signature(self.fn).parameters
        kw: dict = {}
        if self.part_key is not None and self.part_key in params and part:
            kw[self.part_key] = part[self.part_key]
        for name, r in self.by_param.items():
            paths = iv.reads(r.dataset, why=r.why,
                             where=_resolve_sel(r.sel(), part, r.dataset),
                             optional=r.optional, update_file_on_disk=r.is_own)
            if r.as_paths:
                kw[name] = list(paths)          # what iv.reads has always handed back
            else:
                kw[name] = load_value(paths, _ext_of(paths, _sh.EXT)) if paths else None
        return kw

    def load(self, part: dict | None = None):
        """The contents of one shard, without deciding anything.

        The whole read happens under `bookkeeping`, which exempts it from the
        undeclared-read check for the duration and no longer. Whitelisting the path
        instead would exempt it FOREVER — and then any later stage could open that file
        behind iv's back, which is the exact thing the check exists to refuse.
        """
        iv = self.pipeline
        o = next(iter(self.outputs.values()))
        with iv.bookkeeping():
            live = _sh.current_shards(iv.resolve_out(o.dataset))
            got = live.get(_sh.encode_part(part if part is not None else self.fixed_part))
            if got is None:
                if o.allow_missing:
                    return None
                raise StateError(
                    f"{o.dataset} has no shard for "
                    f"{_sh.encode_part(part) or '(one shard)'} — it was not built.")
            return load_value([got.path], got.ext)

    def for_each(self, over) -> list[str]:
        """Build one shard per key, skipping the ones already current."""
        if self.part_key is None or self.split or self.fixed_part is not None:
            raise DeclError(
                f"{self.primary} is not built one partition at a time, so there is "
                f"nothing to iterate. Declare part='season' without split=.")
        iv = self.pipeline
        want = [str(p) for p in over]
        rebuild = [p for p in want
                   if iv.force or not self.may_skip or self.why_stale(p) is not None]
        for p in rebuild:
            self.build({self.part_key: p})
        return rebuild


def _externals(spec, dataset) -> tuple:
    """Sources outside the tree — an API, a bucket, a page being scraped.

    Declared rather than called, so `iv graph` draws them without the body running. They
    cannot trigger anything: nothing on disk says whether an endpoint moved, which is what
    a clock read is for.
    """
    if not spec:
        return ()
    if not isinstance(spec, dict):
        raise DeclError(
            f"{dataset}: external= is {{name: why}} — "
            f'external={{"espn/feeds": "ESPN\'s season files"}}.')
    return tuple((str(k), _decl._why(v, f"{dataset} external {k!r}"))
                 for k, v in spec.items())


def _part_spec(part, dataset) -> tuple:
    """`part=` is either a KEY this stage builds one shard per, or a LITERAL shard it owns.

    The literal form is how several stages share one dataset: three blocks of a college
    feature table, or the played and unplayed halves of a prediction table. Each names the
    shard it writes, so the graph can see they do not collide.
    """
    if part is None:
        return None, None
    if isinstance(part, str):
        return part, None
    if isinstance(part, dict) and part:
        fixed = {str(k): str(v) for k, v in part.items()}
        return (next(iter(fixed)) if len(fixed) == 1 else None), fixed
    raise DeclError(
        f"{dataset}: part= is a key this stage builds one shard per — part='season' — or "
        f"a literal shard it owns — part={{'source': 'ncaa'}}. Got {part!r}.")


def _ext_of(paths, default: str) -> str:
    suffixes = {Path(str(p)).suffix for p in paths}
    return suffixes.pop() if len(suffixes) == 1 else default
