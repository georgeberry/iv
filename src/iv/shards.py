"""One parquet file is the unit. A dataset is a directory of them.

    processed/box_features/               partitioned
        season=2025.a3f21c8e.4f10bb92.parquet
        season=2026.7b09d411.4f10bb92.parquet
        _index.json
    processed/xpm/                        one shard, so no partition segment
        9c1f2a03.8ad10e77.parquet
    raw/sr_college/                       policy="settled": no digests at all
        school=duke__season=2019.parquet

A shard's name carries everything a decision needs, and that is the whole design:

    <part> . <content> . <recipe> . parquet

`content` is a digest of the rows. It is what DEPENDANTS fold, so a rebuild that produces
identical rows produces an identical name and nothing downstream stirs — early cutoff,
for free, with no comparison to make.

`recipe` is a digest of the metadata and the input ids this shard was built from. It is
what THIS shard's staleness is decided by: recompute the recipe from the inputs as they
stand, and ask whether a file with that recipe is on disk. So `is_current` over a whole
pipeline is one directory listing per dataset and ZERO file reads.

Two consequences worth stating, because they are why the name carries both:

**Nothing load-bearing is shared between writers.** `_index.json` holds lineage, timings
and the `by`/`at` provenance that makes `iv why` readable — and it is ADVISORY. If it is
lost, raced, or stale, the pipeline still decides correctly, because the decision is the
filename. The previous design kept one small state file per artifact precisely because a
shared file is where parallel runs go wrong; here there is no shared file on the path that
matters.

**Sub-file reads stop existing.** "Seasons up to T" is a choice of filenames, not a filter
over rows, so a stage that must not see the future physically cannot open it. The bound is
visible at the call site instead of applied two layers down.

ORDERING IS BY PARSED VALUE, NEVER BY FILENAME STRING. `game_id=9` sorts before
`game_id=10`, with nobody having to remember to zero-pad. Row order is a model input in
the pipelines this serves, so `select()` returning a stable, semantic order is a guarantee
rather than a convenience.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Iterable

from .errors import DeclError, StateError

DIGEST_LEN = 16                  # hex chars; an equality check, not a signature
PART_SEP = "__"                  # between key=value pairs within one part
INDEX_NAME = "_index.json"
INDEX_VERSION = 1
EXT = ".parquet"
NO_PART = "_"                    # the part string of a settled dataset with no partition

# A part value has to survive being a filename segment and being parsed back out, so the
# separators are forbidden inside it. Raising here is the point: the alternative is an
# escaping scheme, and a name you cannot read by eye stops being worth putting on disk.
_BAD_IN_VALUE = (".", "=", "/", "\\", PART_SEP)

_NUM = re.compile(r"(\d+)")

# A digest segment is exactly this, which is what makes a shard name SELF-IDENTIFYING: a
# dataset directory can hold a README, an `_index.json` or a hand-made `.bak` without any
# of them parsing as an unpartitioned shard. Without it, every two-segment filename does.
_HEX = re.compile(rf"^[0-9a-f]{{{DIGEST_LEN}}}$")


# ── parts ─────────────────────────────────────────────────────────────────────

def encode_part(part: dict[str, object] | None) -> str:
    """`{"season": 2026}` -> `"season=2026"`. Key ORDER is preserved and is significant.

    The caller's dict order is the declared key order, so a two-key part reads the way it
    was written (`school=duke__season=2019`) rather than in whatever order a set produced.
    """
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

    `parse_name` walks whatever happens to be in a directory, so "this is not a partition
    string" has to be an answer rather than an exception — a stray `notes.backup.parquet`
    alongside the shards must not take down a listing.
    """
    if not part_str or part_str == NO_PART:
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
    content: str = ""                  # "" for a settled shard: existence is the identity
    recipe: str = ""
    part: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name


def shard_name(part: dict[str, object] | None, content: str = "",
               recipe: str = "", *, digested: bool = True) -> str:
    """Build the filename. `digested=False` is a settled dataset: the part, and nothing else."""
    part_str = encode_part(part)
    if not digested:
        return f"{part_str or NO_PART}{EXT}"
    for label, d in (("content", content), ("recipe", recipe)):
        if not _HEX.match(d or ""):
            raise DeclError(
                f"a tracked shard's {label} must be {DIGEST_LEN} hex chars, got {d!r}. "
                f"The shape is what makes a shard name distinguishable from any other "
                f"file that happens to sit in the directory.")
    return f"{part_str}.{content}.{recipe}{EXT}" if part_str else f"{content}.{recipe}{EXT}"


def parse_name(path, *, digested: bool = True) -> Shard | None:
    """Read a shard's name back. `None` if the file is not one of ours.

    `digested` comes from the DATASET's policy, not from the name, and that is deliberate:
    `9c1f2a03.parquet` is a settled part on one dataset and would be an unpartitioned
    tracked shard on another. Deciding by policy means there is no name that parses two
    ways, so there is no case where a wrong guess quietly becomes a wrong id.
    """
    name = path.name
    if not name.endswith(EXT) or name == INDEX_NAME:
        return None
    stem = name[: -len(EXT)]
    if not stem:
        return None
    if not digested:
        part = _decode(stem)
        return None if part is None else Shard(
            path=path, part_str="" if stem == NO_PART else stem, part=part)
    segs = stem.split(".")
    if len(segs) not in (2, 3) or not all(_HEX.match(x) for x in segs[-2:]):
        return None
    if len(segs) == 2:                       # <content>.<recipe> — no partition
        return Shard(path=path, part_str="", content=segs[0], recipe=segs[1], part={})
    part = _decode(segs[0])                  # <part>.<content>.<recipe>
    return None if part is None else Shard(
        path=path, part_str=segs[0], content=segs[1], recipe=segs[2], part=part)


# ── digests ───────────────────────────────────────────────────────────────────

def _short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:DIGEST_LEN]


def content_digest(frame) -> str:
    """A digest of the rows, ORDER-SENSITIVE, with the schema folded in.

    Order-sensitive because row order is an input downstream — minibatches are contiguous
    slices of a file, and an order-insensitive digest would let a reorder pass as no
    change at all. Schema included because a rename or a dtype change is a change even
    when every value survives it.

    Not the file's bytes: parquet is not reproducible across writes (compression and
    footer metadata move on their own), so a byte digest would report a change on every
    rebuild and defeat the early cutoff this exists to provide.
    """
    schema = "|".join(f"{c}:{t}" for c, t in frame.schema.items())
    h = hashlib.sha256()
    h.update(f"{frame.height}|{schema}|".encode())
    if frame.height:
        rows = frame.hash_rows(seed=0)
        try:
            h.update(rows.to_numpy().tobytes())
        except (ImportError, ModuleNotFoundError):
            for v in rows.to_list():
                h.update(str(v).encode())
    return h.hexdigest()[:DIGEST_LEN]


def content_digest_of_file(path) -> str:
    """`content_digest` of a file on disk. MUST agree with the in-memory form.

    The caller decides which to use, and it is a real cost decision rather than a detail:
    over a bucket this is a download of something that was just uploaded. A caller holding
    the frame should hash the frame.
    """
    import polars as pl
    return content_digest(pl.read_parquet(str(path)))


def recipe_digest(meta: str, inputs: dict[str, str]) -> str:
    """What this shard was BUILT FROM, as one string: the metadata and the input ids.

    Sorted by input name so two runs that discovered the same inputs in different orders
    agree. This is the value that decides whether the shard rebuilds; `content` is the
    value that decides whether anything downstream does.
    """
    body = "|".join([f"meta={meta}"] + [f"{k}={inputs[k]}" for k in sorted(inputs)])
    return _short(body)


def dataset_id(shards: Iterable[Shard]) -> str:
    """The id of a dataset, or of a SELECTION of it, as its dependants see it.

    A selection has the digest of just the shards selected, which is what makes a
    walk-forward stage precise: a cohort fit on seasons up to T folds exactly those
    seasons, so next year's data cannot move it and a correction to an old one still does.

    A settled shard contributes its part name and no content, because the question a
    fetch-once archive answers is coverage, not change.
    """
    body = "|".join(f"{s.part_str}:{s.content}"
                    for s in sorted(shards, key=lambda s: s.part_str))
    return "ds:" + _short(body) if body else "ds:(empty)"


# ── listing and selection ─────────────────────────────────────────────────────

def list_shards(dataset_dir, *, digested: bool = True) -> dict[str, list[Shard]]:
    """Every shard on disk, grouped by part. A LIST per part, because duplicates are real.

    A crash between writing a new shard and dropping the one it replaced leaves two files
    for one partition. That is a fact about the directory, so it is reported rather than
    resolved here — `current_shards` is where the policy lives.
    """
    out: dict[str, list[Shard]] = {}
    if not dataset_dir.exists():
        return out
    for p in dataset_dir.iterdir():
        sh = parse_name(p, digested=digested)
        if sh is not None:
            out.setdefault(sh.part_str, []).append(sh)
    return out


def current_shards(dataset_dir, *, digested: bool = True) -> dict[str, Shard]:
    """One shard per part, or an error naming the ambiguity.

    RAISES rather than picking. Two shards for one partition means an interrupted commit,
    and the two differ in content by definition — choosing the newer one by mtime would be
    a silent, unreproducible answer to a question the directory cannot actually answer.
    `gc()` is the fix, and it should be run knowingly.
    """
    found = list_shards(dataset_dir, digested=digested)
    out: dict[str, Shard] = {}
    for part_str, shards in found.items():
        if len(shards) > 1:
            names = ", ".join(sorted(s.name for s in shards))
            raise StateError(
                f"{dataset_dir} holds {len(shards)} shards for partition "
                f"{part_str or '(none)'}: {names}. A commit was interrupted. Run `iv gc` "
                f"to drop the superseded one, or delete it by hand — this cannot be "
                f"resolved by guessing which is current.")
        out[part_str] = shards[0]
    return out


def select(shards: dict[str, Shard], where: dict[str, object] | None = None,
           *, dataset: str = "") -> list[Shard]:
    """The shards a `where=` names, SORTED BY PARSED PARTITION VALUE.

    `where={"season": ["2019", "2020"]}` — an explicit list, and every value must be
    present or it raises. That is the coverage check, and it belongs here: a stage that
    asked for twenty seasons and silently got nineteen is the failure this repo keeps
    finding, and an empty join is indistinguishable from a thin one after the fact.

    `where={"season": fn}` — a predicate over the parsed value. A predicate is allowed to
    match nothing; the caller decides whether that is legal, because "no season qualifies
    yet" is a real state early in a walk-forward loop.
    """
    picked = list(shards.values())
    for key, want in (where or {}).items():
        if callable(want):
            picked = [s for s in picked if key in s.part and want(s.part[key])]
            continue
        want_list = [str(v) for v in (want if isinstance(want, (list, tuple, set)) else [want])]
        have = {s.part[key] for s in picked if key in s.part}
        missing = [v for v in want_list if v not in have]
        if missing:
            raise StateError(
                f"{dataset or 'dataset'} has no shard for {key}="
                f"{', '.join(missing)}. Present: {', '.join(sorted(have)) or 'none'}. "
                f"An explicit list is a coverage claim, so this is an error rather than a "
                f"quietly shorter read.")
        picked = [s for s in picked if key in s.part and s.part[key] in set(want_list)]
    return sorted(picked, key=lambda s: sort_key(s.part_str))


# ── committing ────────────────────────────────────────────────────────────────

def commit(tmp_path, dataset_dir, *, part: dict[str, object] | None,
           content: str, recipe: str, digested: bool = True) -> object:
    """Move a just-written temp file into its final name and drop what it replaced.

    NEW FIRST, THEN THE OLD ONE. The window in between holds two shards for the partition,
    which `current_shards` reports loudly; the other order holds ZERO, which reads as "the
    data is gone". Neither is atomic over a bucket — object stores have no rename — so the
    choice is which failure a crash leaves behind, and a duplicate is recoverable where an
    absence is not.
    """
    name = shard_name(part, content, recipe, digested=digested)
    final = dataset_dir / name
    part_str = encode_part(part)
    superseded = [s for s in list_shards(dataset_dir, digested=digested).get(part_str, [])
                  if s.name != name]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if str(tmp_path) != str(final):
        if final.exists():
            # The identical shard is already here: same part, same content, same recipe.
            # Nothing to do but drop the copy we just made.
            tmp_path.unlink()
        else:
            tmp_path.rename(final)
    for s in superseded:
        s.path.unlink()
    return final


def gc(dataset_dir, *, digested: bool = True, keep: set[str] | None = None) -> list[str]:
    """Drop shards no longer referenced. Returns the names removed.

    `keep` names the shard filenames that are current. Anything else in the directory that
    parses as one of ours is a leftover from an interrupted commit.
    """
    keep = keep or set()
    removed = []
    for shards in list_shards(dataset_dir, digested=digested).values():
        for s in shards:
            if s.name not in keep:
                s.path.unlink()
                removed.append(s.name)
    return sorted(removed)


# ── the index: advisory, never load-bearing ───────────────────────────────────

def read_index(dataset_dir) -> dict:
    """Lineage and timings for `iv why`. NEVER consulted to decide staleness.

    Returns `{}` on anything unreadable, and that is safe here in a way it would not be
    for state: a decision made from an empty index would be a decision made from the
    filenames, which is the decision anyway. Nothing here can make a stale shard look
    current.
    """
    p = dataset_dir / INDEX_NAME
    if not p.exists():
        return {}
    try:
        got = json.loads(p.read_text())
    except (ValueError, OSError):
        return {}
    return got if isinstance(got, dict) and got.get("v") == INDEX_VERSION else {}


def write_entry(dataset_dir, part_str: str, entry: dict) -> None:
    """Record what a shard was built from. Best effort; a failure never fails the build."""
    try:
        got = read_index(dataset_dir) or {"v": INDEX_VERSION, "shards": {}}
        got.setdefault("shards", {})[part_str or NO_PART] = entry
        (dataset_dir / INDEX_NAME).write_text(json.dumps(got, indent=1, sort_keys=True))
    except OSError:
        pass
