from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .errors import DeclError, StateError

DIGEST_LEN = 16
PART_SEP = "__"
INDEX_NAME = "_index.json"
STAGE_ENV = "IV_STAGE_DIR"
INDEX_VERSION = 1
EXT = ".parquet"

_BAD_IN_VALUE = (".", "=", "/", "\\", PART_SEP)

_NUM = re.compile(r"(\d+)")

_HEX = re.compile(rf"^[0-9a-f]{{{DIGEST_LEN}}}$")


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

    @property
    def name(self) -> str:
        return self.path.name


def shard_name(part: dict[str, object] | None, fp: str) -> str:
    if not _HEX.match(fp or ""):
        raise DeclError(
            f"a fingerprint must be {DIGEST_LEN} hex chars, got {fp!r}. The shape is what "
            f"makes a shard name distinguishable from any other file in the directory.")
    part_str = encode_part(part)
    return f"{part_str}.{fp}{EXT}" if part_str else f"{fp}{EXT}"


def parse_name(path) -> Shard | None:
    name = path.name
    if not name.endswith(EXT) or name == INDEX_NAME:
        return None
    segs = name[: -len(EXT)].split(".")
    if len(segs) not in (1, 2) or not _HEX.match(segs[-1]):
        return None
    if len(segs) == 1:
        return Shard(path=path, part_str="", fp=segs[0], part={})
    part = _decode(segs[0])
    return None if part is None else Shard(
        path=path, part_str=segs[0], fp=segs[1], part=part)


def _short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:DIGEST_LEN]


def fingerprint(frame) -> str:
    schema = "|".join(f"{c}:{t}" for c, t in frame.schema.items())
    if frame.height == 0:
        return _short(f"empty|{schema}")
    return _short(f"{frame.height}|{schema}|{frame.hash_rows(seed=0).sum()}")


def fingerprint_of_file(path) -> str:
    import polars as pl
    return fingerprint(pl.read_parquet(str(path)))


def dataset_id(shards: Iterable[Shard]) -> str:
    body = "|".join(f"{s.part_str}:{s.fp}"
                    for s in sorted(shards, key=lambda s: s.part_str))
    return "data:" + _short(body) if body else "data:(empty)"


def list_shards(dataset_dir) -> dict[str, list[Shard]]:
    out: dict[str, list[Shard]] = {}
    if not dataset_dir.exists():
        return out
    for p in dataset_dir.iterdir():
        if p.name == INDEX_NAME:
            continue
        got = parse_name(p)
        if got is None:
            raise StateError(
                f"{p} is in a dataset directory but is not a shard of it. Expected "
                f"<part>.<fingerprint>{EXT} or {INDEX_NAME}, and nothing else — a shard is "
                f"staged on local disk and moved in whole, so there is no in-flight file to "
                f"allow for. A name this cannot read would silently drop a partition and "
                f"every read of {dataset_dir.name} would come back short, so it stops here.")
        out.setdefault(got.part_str, []).append(got)
    return out


def current_shards(dataset_dir) -> dict[str, Shard]:
    out: dict[str, Shard] = {}
    for part_str, shards in list_shards(dataset_dir).items():
        if len(shards) > 1:
            names = ", ".join(sorted(s.name for s in shards))
            raise StateError(
                f"{dataset_dir} holds {len(shards)} shards for partition "
                f"{part_str or '(none)'}: {names}. A commit was interrupted. Run `iv gc` to "
                f"drop the superseded one — this cannot be resolved by guessing.")
        out[part_str] = shards[0]
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
        picked = [s for s in picked if key in s.part and matches(s.part[key], rule)]
    return sorted(picked, key=lambda s: sort_key(s.part_str))


def stage(tag: object = "", stage_dir=None) -> Path:
    d = Path(stage_dir or os.environ.get(STAGE_ENV) or tempfile.gettempdir()) / "iv-stage"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{os.getpid()}-{tag}{EXT}"


def commit(staged, dataset_dir, *, part: dict[str, object] | None) -> object:
    fp = fingerprint_of_file(staged)
    final = dataset_dir / shard_name(part, fp)
    part_str = encode_part(part)
    superseded = [s for s in list_shards(dataset_dir).get(part_str, [])
                  if s.name != final.name]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if final.exists():
        staged.unlink()
    else:
        _move(staged, final)
    for s in superseded:
        s.path.unlink()
    return final


def _move(src: Path, dst) -> None:
    upload = getattr(dst, "upload_from", None)
    if upload is None:
        shutil.move(str(src), str(dst))
        return
    upload(src)
    src.unlink()


def gc(dataset_dir, *, keep: set[str] | None = None) -> list[str]:
    keep = keep or set()
    removed = []
    for shards in list_shards(dataset_dir).values():
        for s in shards:
            if s.name not in keep:
                s.path.unlink()
                removed.append(s.name)
    return sorted(removed)


def read_index(dataset_dir) -> dict:
    p = dataset_dir / INDEX_NAME
    if not p.exists():
        return {}
    try:
        got = json.loads(p.read_text())
    except (ValueError, OSError) as e:
        raise StateError(
            f"{p} is unreadable: {e}. It records what each shard was built from, so a "
            f"damaged one cannot be told apart from a dataset that was never built. "
            f"Delete it to force a rebuild of this dataset, knowingly.") from e
    if not isinstance(got, dict) or got.get("v") != INDEX_VERSION:
        raise StateError(
            f"{p} is version {got.get('v') if isinstance(got, dict) else '?'}, and this is "
            f"version {INDEX_VERSION}. Delete it to rebuild this dataset.")
    return got


def write_entry(dataset_dir, part_str: str, entry: dict) -> None:
    got = read_index(dataset_dir) or {"v": INDEX_VERSION, "shards": {}}
    got.setdefault("shards", {})[part_str] = entry
    (dataset_dir / INDEX_NAME).write_text(json.dumps(got, indent=1, sort_keys=True))
