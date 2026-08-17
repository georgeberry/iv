"""The id, the state file, and the one implementation of "is this stale".

THE RULE, and it is the whole of it:

    id(A) = H( fingerprint(A's data), A's metadata, the ids of A's inputs )

    stale(A)  <=>  recomputed id(A) != stored id(A)

A root — a file nothing in the pipeline writes — has no inputs and no metadata of its own,
so its id IS its data fingerprint. That is where the recursion bottoms out. Every derived
artifact folds its inputs' ids into its own, so a root that moves moves the whole chain
below it, and a root rewritten with identical data moves nothing.

WHY THE ID CONTAINS THE UPSTREAM. Identify an artifact by its version tag and its own
content alone and a dependant cannot tell that its input was rebuilt from different data.
That is not hypothetical: it is how a walk-forward evaluation in the repo this came from
went from 7 seasons to 24 with no new game, while every downstream stage read tag-ok,
skipped, and served the old numbers. With the upstream ids inside the id, the failure mode
is not expressible.

WHY THE FINGERPRINT IS OF THE DATA. A fetcher rewrites its output every run. Hash the
bytes and every no-op refetch invalidates the world; use mtime and it is worse. Hash the
rows and a refetch that changed nothing changes nothing.

WHAT STAYS MANUAL. A builder whose logic changed produces different output from identical
inputs, and no fingerprint of the inputs can see that. That is what the version axes are
for: they sit in the metadata term, so a bump moves every id below it. The alternative is
hashing source code and its transitive imports, which is a bigger promise than it looks.

ONE IMPLEMENTATION. `is_current(rel)` is `why_stale(rel) is None`. Three hand-written
copies of a staleness rule will disagree, and the display will report everything current
while two artifacts are stale.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from . import fingerprint as fp_mod
from . import record as rec
from .errors import StateError
from .fingerprint import ABSENT

STATE_VERSION = 1

# The closed vocabulary. Each entry changes what the id MEANS, not merely what is done
# with it — which is why they live at the write site rather than in a list somewhere.
POLICIES = ("tracked", "manual", "settled", "exempt", "clock")

_cache: dict | None = None


# ── the artifact spec, as declared at its write site ──────────────────────────

@dataclass(frozen=True)
class Spec:
    """Everything a write site said about the artifact it produces."""
    why: str
    fp: object = "data"                          # str name or callable
    versions: tuple[str, ...] = ("data",)
    policy: str = "tracked"
    terminal: bool = False
    scope: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(
                f"unknown policy {self.policy!r}; expected one of {POLICIES}")


# ── the id ────────────────────────────────────────────────────────────────────

def metadata_term(versions: tuple[str, ...] | list[str], policy: str) -> str:
    """The non-data half of the id: the selected version axes, and the policy.

    Axis VALUES come from the live config, axis NAMES from the write site. So a bump in
    `pyproject.toml` moves the id of everything that selected that axis, and nothing else.
    """
    from .config import get
    cfg = get()
    parts = []
    for axis in sorted(versions):
        if axis not in cfg.versions:
            raise StateError(
                f"version axis {axis!r} is not configured. "
                f"Known axes: {sorted(cfg.versions) or '(none)'} — "
                f"add it under [tool.dagio.versions].")
        parts.append(f"{axis}-{cfg.versions[axis]}")
    term = "+".join(parts) if parts else "unversioned"
    if policy == "clock":
        # The clock is the input. Nothing upstream can express "rebuild once a day".
        term += f"|date={_dt.date.today().isoformat()}"
    return f"{term}|policy={policy}"


def compute_id(fp_value: str, meta: str, inputs: dict[str, str], policy: str) -> str:
    """Fold the three terms into one id.

    `exempt` drops the input term: a walk-forward artifact fit only on completed periods
    has declared inputs that move nightly and cannot reach it. The metadata is then the
    whole rule, which is exactly the trade — a new period needs a version bump to appear.
    """
    body = [f"fp={fp_value}", f"meta={meta}"]
    if policy != "exempt":
        body += [f"{k}={inputs[k]}" for k in sorted(inputs)]
    return hashlib.sha256("|".join(body).encode()).hexdigest()[:fp_mod.DIGEST_LEN]


# ── the state file ────────────────────────────────────────────────────────────

def _path():
    from .config import get
    return get().state_path


def load() -> dict:
    """The whole state file.

    Fails LOUD on malformed JSON. Returning `{}` there would make invalidation a no-op and
    every builder rebuild forever, which reads as "the cache does not work" rather than as
    "the state file is corrupt". A missing file is the only legal empty state.
    """
    global _cache
    if _cache is not None:
        return _cache
    p = _path()
    with rec.bookkeeping():
        if not p.exists():
            _cache = {"version": STATE_VERSION, "artifacts": {}}
            return _cache
        try:
            raw = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise StateError(
                f"state file {p} is unreadable: {e}. Fix or delete it — deleting costs a "
                f"full rebuild, which is recoverable; guessing is not.") from e
    if raw.get("version") != STATE_VERSION:
        raise StateError(
            f"state file {p} is version {raw.get('version')}, this dagio writes "
            f"{STATE_VERSION}. Delete it to rebuild from scratch.")
    _cache = raw
    return _cache


def save(data: dict) -> None:
    p = _path()
    body = json.dumps(data, indent=2, sort_keys=True)
    with rec.bookkeeping():
        if isinstance(p, Path):
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(body)
            os.replace(tmp, p)              # atomic, so a crash cannot leave it torn
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body)


def record_of(rel: str) -> dict | None:
    return load()["artifacts"].get(rel)


def reset() -> None:
    """Forget the in-process copies. Every stage is its own process in the normal case, so
    this exists for tests and for a long-lived host."""
    global _cache
    _cache = None
    _COLLECTION.clear()


# ── ids of things ─────────────────────────────────────────────────────────────

def id_of(rel: str, how: object = "data") -> str:
    """The id of an artifact AS SEEN BY ITS DEPENDANTS.

    Derived — something in the pipeline writes it — so its id is the one stamped when it
    was built: a dict lookup, no file read. A staleness check on a current pipeline
    therefore touches only the raw feeds.

    Otherwise it is a root, and its id is its data fingerprint, computed live. That is
    where the recursion bottoms out, and the absence of a record is exactly how a root is
    recognised — nothing declares it.
    """
    entry = record_of(rel)
    if entry is not None:
        return entry["id"]
    from .paths import fields, resolve
    if fields(rel):
        return collection_id(rel, how)
    p = resolve(rel)
    with rec.bookkeeping():
        if not p.exists():
            return ABSENT
        return fp_mod.compute(p, how)


_COLLECTION: dict[tuple[str, object], str] = {}


def instances_of(template: str) -> list[str]:
    """Every concrete rel path a template covers: on disk, plus anything stamped.

    Both halves are needed. A root feed exists only on disk. A per-partition artifact the
    pipeline WRITES exists in the state file, and if it has since been deleted the
    collection must still notice it is gone rather than quietly shrinking.
    """
    from .config import get
    from .paths import fields
    from .static import _pattern
    pattern = template
    for f in fields(template):
        pattern = pattern.replace("{" + f + "}", "*")
    root = get().data_root
    with rec.bookkeeping():
        on_disk = {str(p)[len(str(root)) + 1:] for p in root.glob(pattern)}
    rx = _pattern(template)
    stamped = {rel for rel in load()["artifacts"] if rx.match(rel)}
    return sorted(on_disk | stamped)


def collection_id(template: str, how: object = "data") -> str:
    """The id of a whole per-partition feed, e.g. `raw/box/{season}.parquet`.

    A stage that reads one file per partition depends on ALL of them, so the artifact it
    produces is keyed on the collection: every instance's id, folded together. A file
    appearing, vanishing, or changing all move it.

    It folds each instance's ID, not its fingerprint — so an instance the pipeline writes
    contributes its stamped id, exactly as a single-file input would. Fingerprinting here
    while `id_of` reads the stamp elsewhere makes the collection and its members disagree,
    which shows up as an outer guard that says "stale" over a cache that says "nothing to
    do".

    THE COST IS REAL, and it is why `fp=` is on `reads()`. A collection of ROOTS is
    fingerprinted on every check, so a twenty-one-season feed of 220 MiB files wants
    `fp="rows"` (a footer read) rather than the default full data hash.
    """
    ckey = (template, how if isinstance(how, str) else id(how))
    cached = _COLLECTION.get(ckey)
    if cached is not None:
        return cached
    found = instances_of(template)
    if not found:
        out = ABSENT
    else:
        body = "|".join(f"{rel}={id_of(rel, how)}" for rel in found)
        out = "coll:" + hashlib.sha256(body.encode()).hexdigest()[:fp_mod.DIGEST_LEN]
    _COLLECTION[ckey] = out
    return out


def input_ids(inputs: dict[str, object]) -> dict[str, str]:
    """`{rel: fp strategy}` -> `{rel: id}`. The strategy is only consulted for roots."""
    return {rel: id_of(rel, how) for rel, how in inputs.items()}


# ── stamping ──────────────────────────────────────────────────────────────────

def stamp(rel: str, *, spec: Spec, inputs: dict[str, object], by: str,
          fp_value: str | None = None) -> str:
    """Fingerprint the artifact, fold everything into one id, and write the record.

    Called only on a write that completed. A raise inside the write leaves no record, so a
    half-written artifact reads as stale rather than as fresh — the failure the shell-level
    `run_if_stale` this replaces used to have, where the stamp did not depend on the build
    succeeding.

    `inputs` maps each input's rel path to the fingerprint strategy to use IF it turns out
    to be a root. Both the ids and the strategies are stored: the ids are what the next
    check compares against, the strategies are what lets it recompute a root's id at all.
    """
    from .paths import resolve
    p = resolve(rel)
    with rec.bookkeeping():
        if not p.exists():
            raise StateError(
                f"{rel} was declared written but is not on disk. Nothing is stamped — "
                f"a stamp means one thing only: THIS code produced THIS file.")
        value = fp_value if fp_value is not None else fp_mod.compute(p, spec.fp)

    ids = input_ids(inputs)
    meta = metadata_term(spec.versions, spec.policy)
    new_id = compute_id(value, meta, ids, spec.policy)

    data = load()
    data["artifacts"][rel] = {
        "id": new_id,
        "fp": value,
        "meta": meta,
        "fp_how": spec.fp if isinstance(spec.fp, str) else "<callable>",
        "versions": sorted(spec.versions),
        "policy": spec.policy,
        "terminal": spec.terminal,
        "why": spec.why,
        "in": {rel_i: {"id": ids[rel_i],
                       "fp": inputs[rel_i] if isinstance(inputs[rel_i], str) else "<callable>"}
               for rel_i in sorted(ids)},
        "by": by,
        "at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
    }
    save(data)
    return new_id


# ── the check ─────────────────────────────────────────────────────────────────

def why_stale(rel: str, declared_inputs: dict[str, object] | None = None) -> str | None:
    """None if current, else one line naming the component that moved.

    The recomputation uses the artifact's STORED fingerprint — the file itself has not
    been rewritten since it was stamped, and re-reading it on every check would cost a
    full pass over every artifact in the pipeline to learn nothing.

    `declared_inputs`, when given, is what the code reads NOW, from the static scan. It is
    the only way to notice an input that was ADDED: the stored record has no entry for a
    path the last build never read, so there would be nothing to compare against.
    """
    from .paths import fields, resolve
    if fields(rel):
        # A partitioned artifact: the template is one node in the graph but many files on
        # disk. It is current only if every instance is, and the reason names the instance.
        found = instances_of(rel)
        if not found:
            return "not on disk (no partition of it exists)"
        for inst in found:
            reason = why_stale(inst, declared_inputs)
            if reason is not None:
                return f"{inst}: {reason}"
        return None

    p = resolve(rel)
    with rec.bookkeeping():
        exists = p.exists()
    if not exists:
        return "not on disk"

    entry = record_of(rel)
    if entry is None:
        return "never stamped"

    policy = entry.get("policy", "tracked")
    if policy == "settled":
        return None                         # the question is coverage, not staleness

    stored_inputs = entry.get("in") or {}
    wanted = declared_inputs if declared_inputs is not None else {
        k: v.get("fp", "data") for k, v in stored_inputs.items()
    }

    if declared_inputs is not None and policy != "exempt":
        added = sorted(set(wanted) - set(stored_inputs))
        removed = sorted(set(stored_inputs) - set(wanted))
        if added:
            return f"input added: {added[0]} (the code reads something it did not before)"
        if removed:
            return f"input removed: {removed[0]} (the code no longer reads it)"

    ids_now = input_ids(wanted)
    meta_now = metadata_term(tuple(entry.get("versions") or ()), policy)
    id_now = compute_id(entry["fp"], meta_now, ids_now, policy)
    if id_now == entry["id"]:
        return None

    # The id moved. Say WHICH component, because "the id changed" is not actionable.
    if meta_now != entry.get("meta"):
        return f"version bumped: {entry.get('meta')} -> {meta_now}"
    for k in sorted(ids_now):
        was = (stored_inputs.get(k) or {}).get("id")
        if was != ids_now[k]:
            return f"input moved: {k}  {was} -> {ids_now[k]}"
    return f"id moved: {entry['id']} -> {id_now}"


def is_current(rel: str, declared_inputs: dict[str, object] | None = None) -> bool:
    return why_stale(rel, declared_inputs) is None
