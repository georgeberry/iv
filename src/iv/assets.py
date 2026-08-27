

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path

from . import decl as _decl
from . import shards as _sh
from .errors import DeclError, StateError
from .paths import mkpath


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


def schema_contract(schema, ext: str, dataset: str) -> tuple | None:

    if schema is None:
        return None
    if ext != ".parquet":
        raise DeclError(f"{dataset}: schema= is only for .parquet datasets, not {ext}.")
    try:
        import polars as pl
    except ImportError as e:
        raise DeclError(f"{dataset}: schema= needs the data extra (polars).") from e
    try:
        got = pl.Schema(schema)
    except (TypeError, ValueError) as e:
        raise DeclError(
            f"{dataset}: schema= is a Polars schema or ordered {{column: dtype}} mapping; "
            f"got {schema!r}.") from e
    return tuple((str(name), str(dtype)) for name, dtype in got.items())


def load_value(paths, ext: str):

    if ext == ".parquet":
        import polars as pl
        files = [str(p) for p in paths]
        if len(files) == 1:
            return pl.read_parquet(files[0])
        return pl.concat(
            [pl.scan_parquet(f) for f in files], how="diagonal_relaxed"
        ).collect()
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


OUT_PARAM = "out"


def declared_reads(fn, part_keys: tuple[str, ...] | None, own: str | None = None) -> dict:


    return {name: p.default.bound_to(part_keys).against(own)
            for name, p in inspect.signature(fn).parameters.items()
            if isinstance(p.default, _decl.Read)}


def has_declared_reads(fn) -> bool:
    return any(isinstance(p.default, _decl.Read)
               for p in inspect.signature(fn).parameters.values())


def check_signature(fn, part_keys: tuple[str, ...] | None, dataset: str) -> None:

    for name, p in inspect.signature(fn).parameters.items():
        if isinstance(p.default, _decl.Read) or name == OUT_PARAM:
            continue
        if part_keys is not None and name in part_keys:
            continue
        if p.default is not inspect.Parameter.empty:
            continue
        known = ["a read (iv.all_of(...) and friends)", repr(OUT_PARAM)]
        if part_keys:
            known += [repr(k) for k in part_keys]
        raise DeclError(
            f"{dataset}: parameter {name!r} of {fn.__name__} is not something iv can "
            f"supply. A parameter is one of: {', '.join(known)}, or has a default. "
            f"A partition value is named after the stage's part=.")


@dataclass(frozen=True)
class Dataset:


    dataset: str
    ext: str = _sh.EXT
    allow_missing: bool = False
    part: tuple = ()
    why: str = ""
    schema: tuple | None = None


    standalone: bool = False

    def shard(self, **part) -> "Dataset":


        if not part:
            raise DeclError(
                f"{self.dataset}: shard() names the literal partition this stage writes — "
                f"shard(source='intl').")
        return Dataset(self.dataset, self.ext, self.allow_missing,
                       _fixed(part, self.dataset), self.why, self.schema, self.standalone)

    def __repr__(self) -> str:
        p = " " + ",".join(f"{k}={v}" for k, v in self.part) if self.part else ""
        return f"<iv dataset {self.dataset}{p}>"


def _outputs(spec, ext, allow_missing, schema=None) -> dict:


    if spec is None:
        return {}
    if isinstance(spec, (str, Dataset, Source)):
        d = _dataset(spec, ext, allow_missing, schema)
        return {d.dataset: d}
    if not isinstance(spec, dict) or not spec:
        raise DeclError(
            "output= is one dataset — a path, or an iv.data(...) declared above — or a "
            'dict of {name: dataset} naming several: output={"ratings": XPM, '
            '"summary": XPM_SUMMARY}. The names are the keys the body returns.')
    return {k: _dataset(v, ext, allow_missing) for k, v in spec.items()}


def _fixed(part, dataset) -> tuple:

    if not part:
        return ()
    if not isinstance(part, dict):
        raise DeclError(
            f"{dataset}: part= on a declaration is the literal shard it owns — "
            f"part={{'source': 'ncaa'}}. Got {part!r}.")
    return tuple(sorted((str(k), str(v)) for k, v in part.items()))


def _dataset(v, ext, allow_missing, schema=None) -> Dataset:
    if isinstance(v, Dataset):
        if schema is not None:
            raise DeclError(f"{v.dataset}: schema= belongs on its iv.dataset declaration.")
        return Dataset(_decl._canon(v.dataset), v.ext, v.allow_missing, v.part, v.why,
                       v.schema, v.standalone)
    if isinstance(v, Source):
        raise DeclError(
            f"{v.dataset} was declared a source — something outside this pipeline puts it "
            f"there — so no stage here writes it. Declare it with iv.data(...) instead.")
    d = _decl._canon(v)
    return Dataset(d, ext, allow_missing, schema=schema_contract(schema, ext, d))


class Source:


    def __init__(self, dataset: str, *, why: str, external=None, schema=None) -> None:
        self.dataset = _decl._canon(dataset)
        self.why = _decl._why(why, self.dataset)
        self.schema = schema_contract(schema, _sh.EXT, self.dataset)
        self.externals = _externals(external, self.dataset)
        self.__name__ = self.dataset.rstrip("/").rsplit("/", 1)[-1]

    def __repr__(self) -> str:
        return f"<iv source {self.dataset}>"


class Asset:


    def __init__(self, pipeline, output, fn, *, why: str,
                 part=None, ext: str = _sh.EXT, allow_missing: bool = False,
                 once: bool = False, split: bool = False, single: bool = True,
                 external=None, schema=None, version=None, universe=None) -> None:
        self.pipeline = pipeline
        self.outputs = _outputs(output, ext, allow_missing, schema)
        self.single = single
        self.fn = fn
        self.acts_only = not self.outputs
        self.why = _decl._why(why, self.primary)
        self.once = once
        self.split = split
        self.part_keys, self.fixed_part = _part_spec(part, self.primary)
        self.part_key = self.part_keys[0] if self.part_keys and len(self.part_keys) == 1 else None
        self.version = _version(version, self.primary)
        if self.acts_only and self.version is not None:
            raise DeclError(
                f"{self.primary}: version= belongs to an output's derivation key, but "
                "this action writes no output. Remove version=.")
        if (self.part_keys or self.fixed_part) and any(o.part for o in self.outputs.values()):
            raise DeclError(
                f"{self.primary}: partition ownership is declared both on the stage and "
                "on an output with .shard(...). Put part= on the stage for every output, "
                "or put .shard(...) only on the individual outputs.")
        self._check_output_ownership()
        if split and not self.part_keys:
            raise DeclError(
                f"{self.primary}: split=True means the body computes every partition at "
                f"once and returns {{partition: value}}, so it needs a partition key — "
                f"@iv.step(..., part='season', split=True).")
        if split and len(self.part_keys or ()) != 1:
            raise DeclError(f"{self.primary}: split=True currently requires one partition key.")
        self.universe = _universe(universe, self.primary, self.part_keys, split)
        check_signature(fn, self.part_keys, self.primary)
        if self.acts_only and (self.split or self.part_keys or self.fixed_part):
            raise DeclError(
                f"{self.primary} writes nothing, so it has no shard for part= to name.")
        self.by_param = declared_reads(fn, self.part_keys, self._own_output(fn))
        self.reads = tuple(self.by_param.values())
        self.externals = _externals(external, self.primary)
        self.wants_out = OUT_PARAM in inspect.signature(fn).parameters
        if self.wants_out and not self.single:
            raise DeclError(
                f"{self.primary}: a body that writes through `{OUT_PARAM}` produces one "
                f"file, so it cannot serve several outputs.")
        if self.wants_out and self.split:
            raise DeclError(
                f"{self.primary}: a body taking `{OUT_PARAM}` writes one staged file, "
                "but split=True returns many partition shards. Use a returned mapping "
                "instead of out.")
        self.__name__ = getattr(fn, "__name__", "asset")
        self.__doc__ = getattr(fn, "__doc__", None)
        self.__wrapped__ = fn

    @property
    def primary(self) -> str:
        if not self.outputs:
            return getattr(self.fn, "__name__", "<stage>")
        return next(iter(self.outputs.values())).dataset

    def __getitem__(self, key: str):


        if key not in self.outputs:
            raise DeclError(
                f"{self.__name__} has no output named {key!r}: {sorted(self.outputs)}.")
        return self.outputs[key]

    @property
    def dataset(self) -> str:


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


        return self.primary if len(self.outputs) == 1 else None

    def _check_output_ownership(self) -> None:
        """One stage cannot commit two values to the same physical shard."""
        claimed = {}
        stage_part = (tuple(sorted(self.fixed_part.items())) if self.fixed_part else
                      ("<dynamic>", *self.part_keys) if self.part_keys else ())
        for label, output in self.outputs.items():
            shard = output.part or stage_part
            key = (output.dataset, shard)
            if key in claimed:
                raise DeclError(
                    f"{self.primary}: outputs {claimed[key]!r} and {label!r} both write "
                    f"{output.dataset}{dict(shard) if output.part else ''}. Give shared "
                    "datasets distinct .shard(...) partitions, or declare one output.")
            claimed[key] = label

    def part_for(self, dataset: str) -> tuple:


        for o in self.outputs.values():
            if o.dataset == dataset:
                if o.part:
                    return o.part
                break
        return tuple(sorted(self.fixed_part.items())) if self.fixed_part else ()

    def __repr__(self) -> str:
        p = f", part={self.part_keys or self.fixed_part!r}" if (self.part_keys or self.fixed_part) else ""
        return f"<iv stage {', '.join(self.datasets)}{p}>"


    @property
    def triggers(self) -> tuple:

        return tuple(r for r in self.reads if not r.is_own)

    def triples(self) -> tuple:
        return tuple(r.triple() for r in self.triggers)

    @property
    def may_skip(self) -> bool:


        return bool(self.outputs) and (bool(self.triggers) or self.once)


    def _part(self, args, kwargs) -> dict | None:
        if self.fixed_part is not None:
            if args or kwargs:
                raise DeclError(
                    f"{self.primary} writes the fixed partition {self.fixed_part}, so a "
                    f"call names no other.")
            return dict(self.fixed_part)
        if self.part_keys is None or self.split:
            if args or kwargs:
                raise DeclError(
                    f"{self.primary} is not built one partition at a time, so it takes no "
                    f"partition value.")
            return None
        if set(kwargs) == set(self.part_keys):
            return self._validate_part({k: str(kwargs[k]) for k in self.part_keys})
        if len(self.part_keys) == 1 and len(args) == 1:
            return self._validate_part({self.part_keys[0]: str(args[0])})
        if len(args) == len(self.part_keys):
            return self._validate_part(dict(zip(self.part_keys, map(str, args))))
        if len(self.part_keys) == 1:
            raise DeclError(
                f"{self.primary} is partitioned by {self.part_keys[0]!r}, so a call names one: "
                f"{self.__name__}('2024') or {self.__name__}({self.part_keys[0]}='2024').")
        raise DeclError(
            f"{self.primary} is partitioned by {self.part_keys!r}, so name every partition: "
            f"{self.__name__}(league='nba', season='2025').")

    def why_stale(self, *args, **kwargs) -> str | None:

        part = self._part(args, kwargs)
        if self.acts_only:

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
        if (self.may_skip and not iv.force
                and self.why_stale(*args, **kwargs) is None):
            return self.load(part) if self.single and not self.split else False
        self.build(part)
        if self.acts_only:
            return True
        return self.load(part) if self.single and not self.split else True

    def build(self, part: dict | None) -> None:

        iv = self.pipeline
        iv._fresh_scope()
        prev = (iv._part, iv._in_step, iv._node, iv._inputs, iv._outputs,
                iv._declared_externals)
        iv._part, iv._in_step = part, True
        iv._node = iv._node_name(self.fn)
        iv._inputs, iv._outputs = self.triples(), self.datasets
        iv._declared_externals = self.externals
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
                if all(o.allow_missing for o in self.outputs.values()):
                    for o in self.outputs.values():
                        iv.record("io", op="skip-empty", rel=o.dataset, why=self.why)
                    return
                raise DeclError(
                    f"{self.primary}: {self.__name__} returned None and takes no "
                    f"{OUT_PARAM!r} parameter, so nothing was produced. Return the value "
                    f"to store, take `{OUT_PARAM}` and write to it, or pass "
                    f"allow_missing=True if this partition legitimately has none.")
            if self.split:
                self._commit_split(value)
            elif self.single:
                self._commit(next(iter(self.outputs.values())), value, part)
            else:
                self._commit_many(value, part)
        finally:
            (iv._part, iv._in_step, iv._node, iv._inputs, iv._outputs,
             iv._declared_externals) = prev
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


        from .core import _resolve_sel
        iv = self.pipeline
        params = inspect.signature(self.fn).parameters
        kw: dict = {}
        if self.part_keys and part:
            for key in self.part_keys:
                if key in params:
                    kw[key] = part[key]
        for name, r in self.by_param.items():
            paths = iv.reads(r.dataset, why=r.why,
                             where=_resolve_sel(r.sel(), part, r.dataset),
                             optional=r.optional, update_file_on_disk=r.is_own)
            if r.as_paths:
                kw[name] = list(paths)
            else:
                kw[name] = load_value(paths, _ext_of(paths, _sh.EXT)) if paths else None
        return kw

    def load(self, part: dict | None = None):


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
            iv._validate_schema(o.dataset, [got])
            return load_value([got.path], got.ext)

    def universe_parts(self) -> list[dict] | None:
        if self.universe is None:
            return None
        values = self.universe() if callable(self.universe) else self.universe
        out = []
        for v in values:
            if isinstance(v, dict):
                out.append({str(k): str(x) for k, x in v.items()})
            elif len(self.part_keys) == 1:
                out.append({self.part_keys[0]: str(v)})
            else:
                raise DeclError(
                    f"{self.primary} is partitioned by {self.part_keys!r}, so its "
                    f"universe= names every key per entry: "
                    f"[{{'league': 'nba', 'season': '2025'}}]. Got {v!r}.")
        return out

    def for_each(self, over=None) -> list[str]:

        if self.part_keys is None or self.split or self.fixed_part is not None:
            raise DeclError(
                f"{self.primary} is not built one partition at a time, so there is "
                f"nothing to iterate. Declare part='season' without split=.")
        iv = self.pipeline
        if over is None:
            over = self.universe_parts()
            if over is None:
                raise DeclError(
                    f"{self.primary}: for_each() with no argument builds the stage's "
                    f"declared universe=, and this stage declares none.")
        want = [self._coerce_part(p) for p in over]
        rebuild = [p for p in want
                   if iv.force or not self.may_skip or self.why_stale(**p) is not None]
        for p in rebuild:
            self.build(p)
        return [p[self.part_keys[0]] for p in rebuild] if len(self.part_keys) == 1 else rebuild

    def _coerce_part(self, value) -> dict:
        if isinstance(value, dict):
            if set(value) != set(self.part_keys):
                raise DeclError(f"{self.primary}: expected partition keys {self.part_keys}, got {sorted(value)}.")
            return self._validate_part({k: str(value[k]) for k in self.part_keys})
        if len(self.part_keys) == 1:
            return self._validate_part({self.part_keys[0]: str(value)})
        if isinstance(value, (tuple, list)) and len(value) == len(self.part_keys):
            return self._validate_part(dict(zip(self.part_keys, map(str, value))))
        raise DeclError(f"{self.primary}: multi-partition for_each() takes dictionaries or {len(self.part_keys)}-tuples.")

    def _validate_part(self, part: dict) -> dict:
        return part


def _externals(spec, dataset) -> tuple:


    if not spec:
        return ()
    if not isinstance(spec, dict):
        raise DeclError(
            f"{dataset}: external= is {{name: why}} — "
            f'external={{"espn/feeds": "ESPN\'s season files"}}.')
    return tuple((str(k), _decl._why(v, f"{dataset} external {k!r}"))
                 for k, v in spec.items())


def _part_spec(part, dataset) -> tuple:


    if part is None:
        return None, None
    if isinstance(part, str):
        return (part,), None
    if isinstance(part, (tuple, list)) and part and all(isinstance(k, str) and k for k in part):
        if len(set(part)) != len(part):
            raise DeclError(f"{dataset}: partition keys must be distinct, got {part!r}.")
        return tuple(part), None
    if isinstance(part, dict) and part:
        fixed = {str(k): str(v) for k, v in part.items()}
        return None, fixed
    raise DeclError(
        f"{dataset}: part= is key(s) this stage builds one shard per — part='season' or "
        "part=('league', 'season') — or "
        f"a literal shard it owns — part={{'source': 'ncaa'}}. Got {part!r}.")


def _universe(value, dataset, part_keys, split):
    if value is None:
        return None
    if not part_keys:
        raise DeclError(
            f"{dataset}: universe= names the partitions this stage builds, so it needs "
            f"part= to say what they are keyed on.")
    if split:
        raise DeclError(
            f"{dataset}: split=True builds every partition in one pass, so there is no "
            f"per-partition universe to enumerate.")
    if not callable(value) and not hasattr(value, "__iter__"):
        raise DeclError(
            f"{dataset}: universe= is the partitions to build \u2014 a sequence, or a "
            f"callable returning one where the answer depends on what is on disk. "
            f"Got {value!r}.")
    return value


def _version(value, dataset):
    if value is None:
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise DeclError(f"{dataset}: version= is a stable string or integer, got {value!r}.")
    return str(value)


def _ext_of(paths, default: str) -> str:
    suffixes = {Path(str(p)).suffix for p in paths}
    return suffixes.pop() if len(suffixes) == 1 else default
