from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .errors import DeclError, StateError

DIGEST_LEN = 16
PART_SEP = "__"
STAGE_ENV = "IV_STAGE_DIR"
EXT = ".parquet"

FINGERPRINT_OF = {
    ".parquet": lambda p: fingerprint(__import__("polars").read_parquet(str(p))),
    ".pkl": lambda p: _short("bytes:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()),
    ".json": lambda p: _short("bytes:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()),
    ".html": lambda p: _short("bytes:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()),
}

EMPTY_FP = "0" * DIGEST_LEN

_BAD_IN_VALUE = (".", "=", "/", "\\", PART_SEP)

_NUM = re.compile(r"(\d+)")

_HEX = re.compile(rf"^[0-9a-f]{{{DIGEST_LEN}}}$")


_local = threading.local()


@contextmanager
def snapshot():


    if getattr(_local, "cache", None) is not None:
        yield
        return
    _local.cache = {}
    try:
        yield
    finally:
        _local.cache = None


def _cache_put(dataset_dir, part_str: str, shard) -> None:
    cache = getattr(_local, "cache", None)
    if cache is None:
        return
    live = cache.get(str(dataset_dir))
    if live is not None:
        live[part_str] = shard


def _cache_drop(dataset_dir) -> None:
    cache = getattr(_local, "cache", None)
    if cache is not None:
        cache.pop(str(dataset_dir), None)


def _for_commit(dataset_dir, part_str: str) -> list[Shard]:
    """Use an active snapshot when it has already listed this dataset."""
    cache = getattr(_local, "cache", None)
    if cache is not None:
        live = cache.get(str(dataset_dir))
        if live is not None:
            shard = live.get(part_str)
            return [shard] if shard is not None else []
    return list_shards(dataset_dir).get(part_str, [])


def encode_part(part: dict[str, object] | None) -> str:
    if not part:
        return ""
    out = []
    for k, v in part.items():
        k, v = str(k), str(v)
        for bad in _BAD_IN_VALUE:
            if bad in k or bad in v:
                raise DeclError(
                    f"partition {k}={v!r} contains {bad!r}, which separates the parts of a "
                    f"shard name. Rename the value, or pick a different partition key.")
        if not k or not v:
            raise DeclError(f"empty partition key or value in {part!r}")
        out.append(f"{k}={v}")
    return PART_SEP.join(out)


def decode_part(part_str: str) -> dict[str, str]:
    got = _decode(part_str)
    if got is None:
        raise StateError(f"malformed partition string {part_str!r}")
    return got


def _decode(part_str: str) -> dict[str, str] | None:
    if not part_str:
        return {}
    out: dict[str, str] = {}
    for chunk in part_str.split(PART_SEP):
        k, sep, v = chunk.partition("=")
        if not sep or not k or not v:
            return None
        out[k] = v
    return out


def _nat(value: str) -> tuple:
    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t)
                 for t in _NUM.split(value) if t != "")


def sort_key(part_str: str) -> tuple:
    return (tuple(_nat(v) for v in decode_part(part_str).values()), part_str)


@dataclass(frozen=True)
class Shard:
    path: object
    part_str: str
    fp: str
    part: dict[str, str] = field(default_factory=dict)
    ext: str = EXT
    key: str = ""

    @property
    def name(self) -> str:
        return self.path.name


def shard_name(part: dict[str, object] | None, fp: str, ext: str = EXT,
               key: str = "") -> str:


    if not _HEX.match(fp or ""):
        raise DeclError(
            f"a fingerprint must be {DIGEST_LEN} hex chars, got {fp!r}. The shape is what "
            f"makes a shard name distinguishable from any other file in the directory.")
    if ext not in FINGERPRINT_OF:
        raise DeclError(
            f"no way to fingerprint a {ext!r} shard; known: {sorted(FINGERPRINT_OF)}. "
            f"Add it to FINGERPRINT_OF and say how its contents are digested.")
    if key and not _HEX.match(key):
        raise DeclError(f"a derivation key must be {DIGEST_LEN} hex chars, got {key!r}")
    return ".".join([p for p in (encode_part(part), key, fp) if p]) + ext


def parse_name(path) -> Shard | None:


    name = path.name
    ext = next((e for e in FINGERPRINT_OF if name.endswith(e)), None)
    if ext is None:
        return None
    segs = name[: -len(ext)].split(".")
    if not segs or len(segs) > 3 or not _HEX.match(segs[-1]):
        return None
    fp, rest = segs[-1], segs[:-1]
    part_str, key = "", ""
    if len(rest) == 2:
        part_str, key = rest
    elif len(rest) == 1:
        if _decode(rest[0]):
            part_str = rest[0]
        elif _HEX.match(rest[0]):
            key = rest[0]
        else:
            return None
    part = _decode(part_str)
    if part is None or (key and not _HEX.match(key)):
        return None
    return Shard(path=path, part_str=part_str, fp=fp, part=part, ext=ext, key=key)


def _short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:DIGEST_LEN]


def fingerprint(frame) -> str:
    schema = "|".join(f"{c}:{t}" for c, t in frame.schema.items())
    if frame.height == 0:
        return _short(f"empty|{schema}")
    return _short(f"{frame.height}|{schema}|{frame.hash_rows(seed=0).sum()}")


def fingerprint_of_file(path) -> str:
    ext = next((e for e in FINGERPRINT_OF if str(path).endswith(e)), None)
    if ext is None:
        raise DeclError(
            f"no way to fingerprint {path}; known: {sorted(FINGERPRINT_OF)}.")
    got = parse_name(path)
    if got is not None and got.fp == EMPTY_FP:
        return EMPTY_FP
    return FINGERPRINT_OF[ext](path)


def is_empty(shard: "Shard") -> bool:
    return shard.fp == EMPTY_FP


def commit_empty(dataset_dir, *, part: dict[str, object] | None,
                 key: str = "", ext: str = EXT) -> object:
    final = dataset_dir / shard_name(part, EMPTY_FP, ext, key)
    part_str = encode_part(part)
    superseded = [s for s in _for_commit(dataset_dir, part_str)
                  if s.name != final.name]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not final.exists():
        final.write_bytes(b"")
    for s in superseded:
        s.path.unlink()
    _cache_put(dataset_dir, part_str, parse_name(final))
    return final


def dataset_id(shards: Iterable[Shard]) -> str:
    body = "|".join(f"{s.part_str}:{s.fp}"
                    for s in sorted(shards, key=lambda s: s.part_str))
    return "data:" + _short(body) if body else "data:(empty)"


def list_shards(dataset_dir) -> dict[str, list[Shard]]:
    out: dict[str, list[Shard]] = {}
    try:
        for p in dataset_dir.iterdir():
            got = parse_name(p)
            if got is None:
                raise StateError(
                    f"{p} is in a dataset directory but is not a shard of it. Expected "
                    f"<part>.<key>.<fingerprint>[{"|".join(sorted(FINGERPRINT_OF))}] and nothing else — a shard is "
                    f"staged on local disk and moved in whole, so there is no in-flight file to "
                    f"allow for. A name this cannot read would silently drop a partition and "
                    f"every read of {dataset_dir.name} would come back short, so it stops here.")
            out.setdefault(got.part_str, []).append(got)
    except FileNotFoundError:
        pass
    return out


def current_shards(dataset_dir) -> dict[str, Shard]:
    cache = getattr(_local, "cache", None)
    if cache is not None:
        hit = cache.get(str(dataset_dir))
        if hit is not None:
            return hit
    out: dict[str, Shard] = {}
    for part_str, shards in list_shards(dataset_dir).items():
        if len(shards) > 1:
            names = ", ".join(sorted(s.name for s in shards))
            raise StateError(
                f"{dataset_dir} holds {len(shards)} shards for partition "
                f"{part_str or '(none)'}: {names}. A commit was interrupted. Run `iv gc` to "
                f"drop the superseded one — this cannot be resolved by guessing.")
        out[part_str] = shards[0]
    if cache is not None:
        cache[str(dataset_dir)] = out
    return out


OPS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
       "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}


def matches(value: str, rule: object) -> bool:
    if isinstance(rule, dict):
        for op, bound in rule.items():
            if op not in OPS:
                raise DeclError(
                    f"unknown range operator {op!r}; expected one of {sorted(OPS)}. "
                    f'A selector is data: where={{"season": {{"lt": "2021"}}}}.')
            if not OPS[op](_nat(value), _nat(str(bound))):
                return False
        return True
    values = rule if isinstance(rule, (list, tuple, set)) else [rule]
    return value in {str(v) for v in values}


def covers(where: dict[str, object] | None, part: dict[str, str]) -> bool:


    return all(k in part and matches(part[k], rule)
               for k, rule in (where or {}).items())


def select(shards: dict[str, Shard], where: dict[str, object] | None = None,
           *, dataset: str = "") -> list[Shard]:
    picked = list(shards.values())
    for key, rule in (where or {}).items():
        if callable(rule):
            raise DeclError(
                f"where={{{key!r}: <function>}} is not allowed. A selector has to be data "
                f'so it can be re-evaluated later: {{"lt": "2021"}}, or an explicit list. '
                f"See iv.shards.matches for why.")
        if not isinstance(rule, dict):
            want = [str(v) for v in (rule if isinstance(rule, (list, tuple, set)) else [rule])]
            have = {s.part[key] for s in picked if key in s.part}
            missing = [v for v in want if v not in have]
            if missing:
                raise StateError(
                    f"{dataset or 'dataset'} has no shard for {key}={', '.join(missing)}. "
                    f"Present: {', '.join(sorted(have)) or 'none'}. An explicit list is a "
                    f"coverage claim, so this is an error rather than a shorter read.")
        picked = [s for s in picked if covers({key: rule}, s.part)]
    return sorted(picked, key=lambda s: sort_key(s.part_str))


def stage(tag: object = "", stage_dir=None, ext: str = EXT) -> Path:
    d = Path(stage_dir or os.environ.get(STAGE_ENV) or tempfile.gettempdir()) / "iv-stage"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{os.getpid()}-{tag}{ext}"


def commit(staged, dataset_dir, *, part: dict[str, object] | None,
           key: str = "", on_commit=None) -> object:
    fp = fingerprint_of_file(staged)
    final = dataset_dir / shard_name(part, fp, Path(str(staged)).suffix, key)
    part_str = encode_part(part)
    superseded = [s for s in _for_commit(dataset_dir, part_str)
                  if s.name != final.name]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if final.exists():
        staged.unlink()
    else:
        _move(staged, final)
    for s in superseded:
        s.path.unlink()
    _cache_put(dataset_dir, part_str, parse_name(final))
    if on_commit:
        on_commit(final, not any(s.fp == fp for s in superseded))
    return final


def _move(src: Path, dst) -> None:
    upload = getattr(dst, "upload_from", None)
    if upload is None:
        shutil.move(str(src), str(dst))
        return
    upload(src)
    src.unlink()


def schemas_of(shards: Iterable[Shard]) -> dict[tuple, list[str]]:
    out: dict[tuple, list[str]] = {}
    for s in shards:
        cols = schema_of_file(s.path)
        out.setdefault(cols, []).append(s.part_str)
    return out


def schema_of_file(path) -> tuple:

    import polars as pl
    schema = pl.read_parquet_schema(str(path))
    return tuple((str(name), str(dtype)) for name, dtype in schema.items())


def gc(dataset_dir, *, keep: set[str] | None = None) -> list[str]:
    keep = keep or set()
    removed = []
    for shards in list_shards(dataset_dir).values():
        for s in shards:
            if s.name not in keep:
                s.path.unlink()
                removed.append(s.name)
    if removed:
        _cache_drop(dataset_dir)
    return sorted(removed)
