"""One parquet file is the unit. A dataset is a directory of them.

    processed/box_features/               partitioned
        season=2025.a3f21c8e4f10bb92.parquet
        season=2026.7b09d4118ad10e77.parquet
        _index.json
    processed/xpm/                        one shard, so no partition segment
        9c1f2a0322ab61de.parquet

A shard's name is its partition and a fingerprint OF ITS DATA, and nothing else:

    <part> . <fingerprint> . parquet

THE DATA IDENTIFIES THE DATA. That is the whole design, and the discipline is in what the
name leaves out. No code hash, no version, no digest of what it was built from — because
a dependant does not care how a shard came to exist, only what is in it. A version bump
that changes no numbers changes no filename, so nothing downstream stirs and nothing on
disk is rewritten. Rebuild an artifact from different inputs and get identical rows, and
the same is true.

What decides whether a stage RE-RUNS is a different question with a different answer, and
it lives in `core2.py`: a stage compares its inputs' fingerprints now against the ones it
recorded last time. Keeping that out of the filename is what stops a code change from
propagating past the one stage it actually affects.

A staleness check is therefore a directory listing and ZERO file reads. A fingerprint is
computed exactly once per shard, at the moment it is written, from a LOCAL staged file.

THE DIRECTORY IS OURS, SO ANYTHING UNRECOGNISED IN IT IS A HARD ERROR. `list_shards`
raises rather than skipping a file it cannot parse. Skipping would mean a shard whose name
got mangled quietly drops out of the dataset and every read downstream is silently SHORT —
the same failure as a join that lands empty and falls back, and the one that costs a season
of predictions before anyone notices. `_index.json` is the only other name that may appear;
a shard being written is staged on local disk and moved in whole, so there is no in-flight
file to make an exception for.

ORDERING IS BY PARSED VALUE, NEVER BY FILENAME STRING. `game_id=9` sorts before
`game_id=10`, with nobody having to remember to zero-pad. Row order is a model input in the
pipelines this serves, so `select()` returning a stable, semantic order is a guarantee
rather than a convenience.

THE SEPARATOR IS `.` AND NOT `_`, which is the one place this deviates from how it was
sketched: `_` appears inside real partition values (`dataset=player_box`) and doubled it is
what joins a multi-key part, so a name split on it would parse two ways. `.` is already
forbidden inside a value.
"""
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

DIGEST_LEN = 16                  # hex chars; an equality check, not a signature
PART_SEP = "__"                  # between key=value pairs within one part
INDEX_NAME = "_index.json"
STAGE_ENV = "IV_STAGE_DIR"       # override where shards are staged before committing
INDEX_VERSION = 1
EXT = ".parquet"

# A part value has to survive being a filename segment and being parsed back out, so the
# separators are forbidden inside it. Raising is the point: the alternative is an escaping
# scheme, and a name you cannot read by eye stops being worth putting on disk.
_BAD_IN_VALUE = (".", "=", "/", "\\", PART_SEP)

_NUM = re.compile(r"(\d+)")

# A fingerprint segment is exactly this, which is what makes a shard name SELF-IDENTIFYING:
# a dataset directory can hold a README or a hand-made `.bak` and neither can be mistaken
# for an unpartitioned shard. Without it, every one-segment filename could be.
_HEX = re.compile(rf"^[0-9a-f]{{{DIGEST_LEN}}}$")


# ── parts ─────────────────────────────────────────────────────────────────────

def encode_part(part: dict[str, object] | None) -> str:
    """`{"season": 2026}` -> `"season=2026"`. Key ORDER is preserved and is significant."""
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
    """`"season=2026"` -> `{"season": "2026"}`. The inverse of `encode_part`."""
    got = _decode(part_str)
    if got is None:
        raise StateError(f"malformed partition string {part_str!r}")
    return got


def _decode(part_str: str) -> dict[str, str] | None:
    """`decode_part`, but `None` instead of raising.

    `parse_name` is asked about whatever is in a directory, so "this is not a partition
    string" has to be an answer it can give rather than an exception it throws.
    """
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
    """A sort key that compares digit runs numerically.

    `game_id=9` before `game_id=10`, and an ISO date still sorts as text. Every element is
    the same 3-tuple shape so tuples of them compare without ever hitting int-vs-str.
    """
    return tuple((0, int(t), "") if t.isdigit() else (1, 0, t)
                 for t in _NUM.split(value) if t != "")


def sort_key(part_str: str) -> tuple:
    """Order shards by their partition VALUES, in declared key order.

    The part string is the tiebreak, so the order is total: two shards can never compare
    equal unless they are the same partition, and an unstable order is exactly the input
    that makes a float sum or a positional slice irreproducible.
    """
    return (tuple(_nat(v) for v in decode_part(part_str).values()), part_str)


# ── names ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Shard:
    """One file, and everything its name says about it."""
    path: object                       # Path or CloudPath
    part_str: str
    fp: str
    part: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name


def shard_name(part: dict[str, object] | None, fp: str) -> str:
    """Build the filename: which partition, and what is in it."""
    if not _HEX.match(fp or ""):
        raise DeclError(
            f"a fingerprint must be {DIGEST_LEN} hex chars, got {fp!r}. The shape is what "
            f"makes a shard name distinguishable from any other file in the directory.")
    part_str = encode_part(part)
    return f"{part_str}.{fp}{EXT}" if part_str else f"{fp}{EXT}"


def parse_name(path) -> Shard | None:
    """Read a shard's name back. `None` if the file is not one of ours."""
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


# ── fingerprints ──────────────────────────────────────────────────────────────

def _short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:DIGEST_LEN]


def fingerprint(frame) -> str:
    """A digest of the DATA: its height, its schema, and every value in it.

    Order-INSENSITIVE. polars sums UInt64 with wraparound, so the sum of the per-row hashes
    is a commutative digest mod 2**64: rewriting the same rows in a different order lands on
    the same value. That is the right default because a reordered rewrite is not new data,
    and a fetcher that reorders on every run would otherwise invalidate the world nightly.

    NOT a moment summary. `H(n, mean, std)` was measured against this and costs the same to
    within 3ms on a 614 MB frame — the read dominates both, and parquet footers carry
    min/max/nulls, never moments, so neither can skip it. What it buys for that is a hole:
    mean and std are undefined for a string column, so `team: LAS,NYL -> SEA,CHI` reads as
    no change at all. `position` and `team` are model inputs here.

    Schema is in the digest because a rename or a dtype change is a change even when every
    value survives it.
    """
    schema = "|".join(f"{c}:{t}" for c, t in frame.schema.items())
    if frame.height == 0:
        return _short(f"empty|{schema}")
    return _short(f"{frame.height}|{schema}|{frame.hash_rows(seed=0).sum()}")


def fingerprint_of_file(path) -> str:
    """`fingerprint` of a file on disk, which is what a READER would get back.

    The committed digest is taken from the staged file rather than the frame in memory, so
    it describes what comes back out of parquet — round-trip and all — and not what the
    writer happened to be holding. Those differ for any dtype parquet does not preserve
    exactly, and it is the reader's answer that downstream correctness rests on.
    """
    import polars as pl
    return fingerprint(pl.read_parquet(str(path)))


def dataset_id(shards: Iterable[Shard]) -> str:
    """The id of a dataset, or of a SELECTION of it, as its dependants see it.

    A selection has the digest of just the shards selected, which is what makes a
    walk-forward stage precise: a cohort fit on seasons up to T folds exactly those seasons,
    so next year's data cannot move it and a correction to an old one still does.
    """
    body = "|".join(f"{s.part_str}:{s.fp}"
                    for s in sorted(shards, key=lambda s: s.part_str))
    return "data:" + _short(body) if body else "data:(empty)"


# ── listing and selection ─────────────────────────────────────────────────────

def list_shards(dataset_dir) -> dict[str, list[Shard]]:
    """Every shard on disk, grouped by part. A LIST per part, because duplicates are real.

    A crash between writing a new shard and dropping the one it replaced leaves two files
    for one partition. That is a fact about the directory, so it is reported rather than
    resolved here — `current_shards` is where the policy lives.
    """
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
    """One shard per part, or an error naming the ambiguity.

    RAISES rather than picking. Two shards for one partition means an interrupted commit,
    and the two differ in content by definition — choosing the newer by mtime would be a
    silent, unreproducible answer to a question the directory cannot actually answer.
    """
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


# The comparisons a range selector may use. Ordered by PARSED value, so `game_id` 9 comes
# before 10 and a season compares as a number rather than as text.
OPS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
       "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}


def matches(value: str, rule: object) -> bool:
    """Does one partition value satisfy a selector?

    A selector is DATA, never a callable, and that is the whole point of it. A lambda
    cannot be replayed — it is a closure in the stage's body, and the staleness check runs
    from another process — so a selection made with one could only ever be remembered as
    the shards it happened to match. A partition appearing later inside the same range
    would then go unnoticed, and the stage would sit there built from less than it should
    be, reporting itself current. Written as data, the rule goes into the record and is
    re-evaluated exactly.
    """
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
    """The shards a `where=` names, SORTED BY PARSED PARTITION VALUE.

    `where={"season": ["2019", "2020"]}` — an explicit list. Every value must be present or
    it raises: a stage that asked for twenty seasons and silently got nineteen is the
    failure this exists to prevent, and an empty read is indistinguishable from thin data
    after the fact.

    `where={"season": {"lt": "2021"}}` — a range, which may legitimately match nothing,
    because "no season qualifies yet" is a real state early in a walk-forward loop.
    """
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


# ── committing ────────────────────────────────────────────────────────────────

def stage(tag: object = "", stage_dir=None) -> Path:
    """A LOCAL path to write a shard to before committing it.

    Local always, even for a dataset that lives in a bucket. The fingerprint has to read the
    rows, and reading a file that was just uploaded means paying for the data twice; staged
    locally the sequence is write local, hash local, upload ONCE straight to the final name.
    Outside any dataset directory, so "a dataset holds shards and its index, full stop" is an
    enforced invariant with no exception carved out of it.
    """
    d = Path(stage_dir or os.environ.get(STAGE_ENV) or tempfile.gettempdir()) / "iv-stage"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{os.getpid()}-{tag}{EXT}"


def commit(staged, dataset_dir, *, part: dict[str, object] | None) -> object:
    """Fingerprint the staged file, move it in under its name, drop what it replaced.

    THE FINGERPRINT IS COMPUTED HERE and is not a parameter, so the caller cannot get it
    wrong and it always describes the bytes that actually landed.

    Identical data means an identical name, so a rebuild that changed nothing finds its own
    shard already in place, drops the staged copy and touches the dataset not at all. Over a
    bucket that is the difference between a no-op and a copy per file.

    NEW FIRST, THEN THE OLD ONE. The window between holds two shards for the partition, which
    `current_shards` reports loudly; the other order holds ZERO, which reads as "the data is
    gone". Neither is atomic over an object store, so the choice is which failure a crash
    leaves behind, and a duplicate is recoverable where an absence is not. A PARTIAL file is
    not among the options either way: what lands is a complete local file, moved whole.
    """
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
    """Local file -> wherever the dataset lives, in one operation.

    `upload_from` is cloudpathlib's, and its absence is what says the destination is an
    ordinary directory. `shutil.move` rather than `rename` for the local case, because the
    staging directory is frequently on another filesystem and `rename` cannot cross one.
    """
    upload = getattr(dst, "upload_from", None)
    if upload is None:
        shutil.move(str(src), str(dst))
        return
    upload(src)
    src.unlink()


def gc(dataset_dir, *, keep: set[str] | None = None) -> list[str]:
    """Drop shards not named in `keep`. Returns what was removed."""
    keep = keep or set()
    removed = []
    for shards in list_shards(dataset_dir).values():
        for s in shards:
            if s.name not in keep:
                s.path.unlink()
                removed.append(s.name)
    return sorted(removed)


# ── the index ─────────────────────────────────────────────────────────────────

def read_index(dataset_dir) -> dict:
    """What each shard was built FROM. See `core2` for why losing it is safe.

    MISSING is normal and returns `{}` — a dataset that has never been built has no record,
    and the shard rebuilds. CORRUPT is not normal and raises. The two used to be the same
    answer, which meant a truncated write or a half-synced file read as "never built" and
    quietly rebuilt the world with no sign anything was wrong.

    A record written by an older layout also raises rather than being ignored, because the
    fix is a migration, not a silent full rebuild.
    """
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
    """Record what a shard was built from.

    RAISES on failure. This is not logging: without the record the shard cannot be shown
    current, so a silent failure here means the stage rebuilds every run forever with
    nothing to explain it.
    """
    got = read_index(dataset_dir) or {"v": INDEX_VERSION, "shards": {}}
    got.setdefault("shards", {})[part_str] = entry
    (dataset_dir / INDEX_NAME).write_text(json.dumps(got, indent=1, sort_keys=True))
