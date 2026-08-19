"""The id, the state file, and the one implementation of "is this stale".

THE RULE, and it is the whole of it:

    id(A) = H( fingerprint(A's data), A's metadata, the ids of A's inputs )

    stale(A)  <=>  recomputed id(A) != stored id(A)

A root — a file nothing in the pipeline writes — has no inputs and no metadata of its own,
so its id IS its data fingerprint. That is where the recursion bottoms out. Every derived
artifact folds its inputs' ids into its own, so a root that moves moves the whole chain
below it, and a root rewritten with identical data moves nothing.

WHY THE ID CONTAINS THE UPSTREAM. Identify an artifact by its version and its own content
alone and a dependant cannot tell that its input was rebuilt from different data. That is
not hypothetical: it is how a walk-forward evaluation in the repo this came from went from
7 seasons to 24 with no new game, while every downstream stage read version-ok, skipped,
and served the old numbers. With the upstream ids inside the id, the failure mode is not
expressible.

WHY THE FINGERPRINT IS OF THE DATA. A fetcher rewrites its output every run. Hash the
bytes and every no-op refetch invalidates the world; use mtime and it is worse. Hash the
rows and a refetch that changed nothing changes nothing.

WHAT `data_version` IS FOR. A builder whose logic changed produces different output from
identical inputs, and no fingerprint of the inputs can see that. `data_version` sits in
the metadata term of every id, so bumping it moves everything. `step(code=True)` narrows
that to one function where you want it; the global remains the honest blunt instrument,
because the alternative is hashing source code and its transitive imports, which is a
bigger promise than it looks.

ONE IMPLEMENTATION. `is_current` is `why_stale(...) is None`. Three hand-written copies of
a staleness rule will disagree, and the display will report everything current while two
artifacts are stale.

ONE FILE PER ARTIFACT. The stamps used to live in a single `state.json`, and stamping was
read-modify-write over the whole of it. Two stages running at once therefore each loaded
the map, added their own artifact, and wrote the map back — so whichever finished last
erased the other's record. Measured on a real parallel run: eight artifacts written, four
stamped. The failure direction was safe (an unstamped artifact rebuilds) but the cache did
not work, and the two writers also raced on one shared `.tmp` name.

So the state is a DIRECTORY, one small JSON per artifact, and a stamp touches exactly the
file it is about. Two stages cannot collide unless they write the same artifact, which is
already an error the graph check reports. It also fits an object store, where there is no
lock to take and a read-modify-write over a shared key is not something you can make safe.

Reading is still ONE bulk load per process — a glob and a concurrent read — because the
question `iv status` asks is about every artifact at once, and twenty serial round trips
over a bucket is the loop this package exists to avoid.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from . import fingerprint as _fp
from .errors import StateError
from .fingerprint import ABSENT

STATE_VERSION = 3
LEGACY_VERSION = 2                   # the single `state.json`, migrated on first touch

# The closed vocabulary. Each entry changes what the id MEANS, not merely what is done
# with it — which is why they live at the write site rather than in a list somewhere.
POLICIES = ("tracked", "manual", "settled", "clock")


@dataclass(frozen=True)
class Spec:
    """Everything a write site said about the artifact it produces."""
    why: str
    fp: object = "data"
    policy: str = "tracked"
    terminal: bool = False
    code: str = ""                       # from step(code=True); "" when not asked for
    version: str = ""                    # an extra version only THIS artifact answers to

    def __post_init__(self):
        if self.policy not in POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}; expected one of {POLICIES}")


class State:
    """One pipeline's state file, and the id arithmetic over it."""

    def __init__(self, iv) -> None:
        self.iv = iv
        self._cache: dict | None = None
        self._collections: dict[tuple, str] = {}
        self._roots: dict[tuple, str] = {}
        self._groups: dict[tuple, dict[str, str] | None] = {}

    @property
    def dir(self):
        """The DIRECTORY the stamps live in, one file per artifact.

        Under the OUT root by default, because the state describes what THIS pipeline
        built — a local run writing into a scratch tree must not stamp the shared one.
        `state_path=` overrides it outright, which is what a shadow run over someone
        else's data wants.

        A path ending in `.json` names the OLD single file, so it is read as the directory
        beside it: `state_path=".../shadow.json"` keeps working and its records land in
        `.../shadow/`. That is the same rule `legacy_path` inverts, so a config written for
        the single file migrates itself.
        """
        p = self.iv.state_path_override or (self.iv.out_root / self.iv.state_rel)
        s = str(p)
        return p.parent / p.name[:-len(".json")] if s.endswith(".json") else p

    @property
    def legacy_path(self):
        """Where a pre-directory `state.json` would be, if this tree has one."""
        d = self.dir
        return d.parent / (d.name + ".json")

    @property
    def renamed_dir(self):
        """Where the stamps lived when this package was called `invalidator`.

        Only meaningful for the DEFAULT layout — an explicit `state_path=` names a tree
        that was never under the old name.
        """
        d = self.dir
        return d.parent.parent / ".invalidator" / d.name if d.parent.name == ".iv" else None

    # ── the id ────────────────────────────────────────────────────────────────

    def metadata_term(self, policy: str, code: str = "", version: str = "") -> str:
        """The non-data half of the id: the global version, the policy, and the two
        narrower things an artifact can additionally answer to.

        `version` is an opaque extra string a step declares when something beyond the data
        governs it — a model version, a vendor API version, a hand-tuned table. Only the
        artifacts that NAME it move when it changes, which is the whole point: a model
        bump should not rebuild a feature pipeline that took four minutes and cannot have
        been affected.
        """
        term = f"v={self.iv.data_version}|policy={policy}"
        if version:
            term += f"|version={version}"
        if code:
            term += f"|code={code}"
        if policy == "clock":
            # The clock is the input. Nothing upstream can express "rebuild once a day".
            term += f"|date={_dt.date.today().isoformat()}"
        return term

    @staticmethod
    def compute_id(fp_value: str, meta: str, inputs: dict[str, str], policy: str) -> str:
        """Fold the three terms into one id.

        NOTHING IS EXEMPT. There was a policy that dropped the input term for artifacts
        fit only on completed periods, and it was the wrong answer to a real problem: it
        could not tell "the live period moved" (irrelevant) from "an old period was
        corrected" (very relevant), so it traded a false rebuild for a missed one, and a
        new period needed a version bump to appear at all.

        The right answer is that such an artifact does not depend on a whole file, it
        depends on the PERIODS IT WAS BUILT FROM. That is `PartitionCache(scoped=...)`
        with a bound per partition, which is precise in both directions.
        """
        body = [f"fp={fp_value}", f"meta={meta}"]
        body += [f"{k}={inputs[k]}" for k in sorted(inputs)]
        return hashlib.sha256("|".join(body).encode()).hexdigest()[:_fp.DIGEST_LEN]

    # ── the stamps ────────────────────────────────────────────────────────────

    def records(self) -> dict[str, dict]:
        """Every stamp, by rel path. One bulk load per process.

        Lazily reading one record per question would be N round trips over a bucket to
        answer what a glob and a concurrent read answer in two, and `iv status` asks about
        every artifact at once. A process therefore sees the state as of its first look —
        which is what the single file did too, and is fine because a stage cannot be
        downstream of a stamp that has not happened yet.

        Fails LOUD on a malformed record. Returning `{}` there would make invalidation a
        no-op and every builder rebuild forever, which reads as "the cache does not work"
        rather than as "the state is corrupt". A missing directory is the only legal empty
        state.
        """
        if self._cache is not None:
            return self._cache
        self._migrate()
        d = self.dir
        with self.iv.bookkeeping():
            files = sorted(d.glob("*.json")) if d.exists() else []
            if len(files) > 1:
                # Small files, one per artifact: serially over a bucket this is the same
                # per-file round trip that `prefetch` exists to kill.
                import concurrent.futures as cf
                with cf.ThreadPoolExecutor(max_workers=min(16, len(files))) as ex:
                    loaded = list(ex.map(_read_record, files))
            else:
                loaded = [_read_record(f) for f in files]
        self._cache = {raw["rel"]: raw for raw in loaded}
        return self._cache

    def _write_record(self, rel: str, entry: dict) -> None:
        """Stamp ONE artifact. Touches only that artifact's file.

        The temp name carries the pid and thread id: two stages stamping at the same
        moment used to write the same `state.json.tmp`, so one could rename the other's
        half-written file into place.
        """
        p = self._record_path(rel)
        # `state_version`, not `version`: an artifact's OWN `version=` is a field of the
        # entry, and one envelope key shadowing it wrote every record as version "".
        body = json.dumps({"state_version": STATE_VERSION, "rel": rel, **entry},
                          indent=2, sort_keys=True)
        with self.iv.bookkeeping():
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(p, Path):
                tmp = p.with_name(f"{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
                tmp.write_text(body)
                os.replace(tmp, p)      # atomic, so a crash cannot leave it torn
            else:
                p.write_text(body)

    def _record_path(self, rel: str):
        return self.dir / record_filename(rel)

    def _absorb(self, old_dir, d) -> None:
        """Take the stamps out of `old_dir` and leave nothing behind.

        A MOVE, not a copy. Two directories both called `state`, both holding stamps, one
        of them dead — that is the single hazard this whole package is against, and
        leaving it around to be safe just relocates the confusion. So: copy every record,
        verify the bytes landed, and only then delete the source.

        Idempotent, because it may find a half-finished move: a record already present at
        the destination is left alone and its source is still removed.
        """
        moved = kept = 0
        d.mkdir(parents=True, exist_ok=True)
        for f in (old_dir.glob("*.json") if old_dir.exists() else ()):
            body = f.read_bytes()
            dest = d / f.name
            if dest.exists():
                kept += 1
            else:
                dest.write_bytes(body)
                if dest.read_bytes() != body:
                    raise StateError(
                        f"copying {f.name} to {dest} did not land the same bytes. "
                        f"Nothing was deleted; fix the destination and re-run.")
                moved += 1
            f.unlink()
        # `state.json.migrated` is debris from the single-file -> directory move, kept
        # then so a stale file was not silently destroyed. The directory it was migrated
        # INTO has now moved as well, so it is dead twice over and it is the only thing
        # keeping the old tree alive. Nothing else here is ours to delete.
        leftover = old_dir.parent / (old_dir.name + ".json.migrated")
        if leftover.exists():
            leftover.unlink()
        for empty in (old_dir, old_dir.parent):
            try:
                empty.rmdir()
            except Exception:           # noqa: BLE001 — cloudpathlib does not raise OSError
                break                   # something we did not put there; leave it alone
        if moved or kept:
            print(f"  moved {moved + kept} stamp(s) out of {old_dir.parent.name}/ into "
                  f"{d.parent.name}/ — the package was renamed, the state did not change "
                  f"shape. The old directory is gone.")

    def _migrate(self) -> None:
        """A pre-directory `state.json` -> one file per artifact, once.

        Worth doing rather than telling people to delete it: the records are unchanged in
        shape, only in layout, and the alternative is a full rebuild of a real pipeline for
        a bookkeeping change. Two processes racing here both write the same bytes to the
        same per-artifact names, so the race is harmless.
        """
        d, legacy = self.dir, self.legacy_path
        # The package was renamed `invalidator` -> `iv`, and the stamps moved with it.
        # Copying them is the whole point: a bookkeeping change must not cost a rebuild
        # of a real pipeline — for wvorp that is the 287s xPM fit and everything below
        # it, for a name.
        old_dir = self.renamed_dir
        with self.iv.bookkeeping():
            # Also when only the debris is left: the stamps can have moved on an earlier
            # run while `state.json.migrated` kept the old tree standing.
            if old_dir is not None and (
                    old_dir.exists()
                    or (old_dir.parent / (old_dir.name + ".json.migrated")).exists()):
                self._absorb(old_dir, d)
            if d.exists() or not legacy.exists():
                return
            try:
                raw = json.loads(legacy.read_text())
            except (json.JSONDecodeError, OSError) as e:
                raise StateError(
                    f"state file {legacy} is unreadable: {e}. Fix or delete it — deleting "
                    f"costs a full rebuild, which is recoverable; guessing is not.") from e
        if raw.get("version") != LEGACY_VERSION:
            raise StateError(
                f"state file {legacy} is version {raw.get('version')}, and the only single "
                f"file this iv can move to the new layout is version "
                f"{LEGACY_VERSION}. Delete it to rebuild from scratch.")
        artifacts = raw.get("artifacts") or {}
        for rel, entry in artifacts.items():
            self._write_record(rel, entry)
        with self.iv.bookkeeping():
            try:
                legacy.rename(legacy.parent / (legacy.name + ".migrated"))
            except Exception:
                # Best effort. Leaving it behind is harmless — the directory now exists,
                # so this never runs again — but a stale file beside a live one misleads.
                pass
        if artifacts:
            print(f"  migrated {len(artifacts)} stamp(s) from {legacy.name} to {d.name}/")

    def record_of(self, rel: str) -> dict | None:
        return self.records().get(rel)

    def reset(self) -> None:
        self._cache = None
        self._collections.clear()
        self._roots.clear()

    # ── ids of things ─────────────────────────────────────────────────────────

    def id_of(self, rel: str, how: object = "data") -> str:
        """The id of an artifact AS SEEN BY ITS DEPENDANTS.

        Derived — something in the pipeline writes it — so its id is the one stamped when
        it was built: a dict lookup, no file read. A staleness check on a current pipeline
        therefore touches only the roots.

        Otherwise it is a root, and its id is its data fingerprint, computed live. That is
        where the recursion bottoms out, and the absence of a record is exactly how a root
        is recognised — nothing declares it.
        """
        entry = self.record_of(rel)
        if entry is not None:
            return entry["id"]
        from .paths import fields
        if fields(rel):
            return self.collection_id(rel, how)
        # Memoised for the life of the process, and it is not an optimisation at the
        # margin: a per-partition key folds the same whole-artifact inputs into every
        # partition, so one 21-season panel re-hashed its three crosswalks 63 times over
        # a bucket. A root cannot move mid-run — anything this pipeline writes has a
        # record, and `stamp` clears the cache — so the second read has the same answer.
        ckey = (rel, how if isinstance(how, str) else id(how))
        if ckey in self._roots:
            return self._roots[ckey]
        p = self.iv.resolve(rel)
        with self.iv.bookkeeping():
            if not p.exists():
                return ABSENT
            self._roots[ckey] = _fp.compute(p, how)
            return self._roots[ckey]

    def slice_id(self, rel: str, label: str) -> str:
        """The identity of ONE SLICE of an artifact: a digest of the rows it selects.

        An artifact can be two populations that different stages own. wvorp's
        `game_predictions.parquet` is written for PLAYED games by `predict_games` and for
        UPCOMING ones by `predict_upcoming_games`, and `build_preseason` reads only the
        played half — it takes a calibration sigma off completed seasons and cannot see an
        unplayed game.

        `slice=` already stopped that being a graph EDGE. It did not stop being a
        staleness trigger: the read recorded the whole-artifact id, so appending tomorrow's
        fixtures moved it and `preseason_*` came out of every refresh stale against rows
        that had not changed. A label that filters what you READ and not what you DEPEND
        ON is only half a slice.

        Computed through the SAME predicate `iv.frame` filters with, so the label cannot
        drift from the measurement.
        """
        from . import fingerprint as _f
        pred = self.iv.slices.get(label)
        if pred is None:
            raise StateError(
                f"input {rel} was stamped under slice {label!r}, which this Invalidator "
                f"does not define. Known: {sorted(self.iv.slices) or 'none'}. Add it to "
                f"slices={{...}} or rebuild the artifact.")
        p = self.iv.resolve(rel)
        with self.iv.bookkeeping():
            if not p.exists():
                return _f.ABSENT
            return "slice:" + _f.frame_digest(pred(_f.read_frame(p)))

    def group_ids(self, rel: str, by: str) -> dict[str, str]:
        """`{period: digest}` for an artifact that holds many periods. Memoised.

        The read is the expensive part — one round trip to a bucket — and a walk-forward
        artifact asks about the same file once per partition. Twenty cohorts times a full
        read of `box_features` on every staleness check is not affordable, so the whole map
        is computed once per `(rel, by)` per process.
        """
        memo = self._groups.setdefault((rel, by), None)
        if memo is not None:
            return memo
        from . import fingerprint as _f
        p = self.iv.resolve(rel)
        with self.iv.bookkeeping():
            out = {} if not p.exists() else _f.group_digests(_f.read_frame(p), by)
        self._groups[(rel, by)] = out
        return out

    def upto_id(self, rel: str, by: str, bound: str) -> str:
        """The identity of everything in `rel` at or before `bound`.

        This is what replaces a whole-file id for a reader that only sees the past. It
        moves when a period AT OR BEFORE the bound changes, and does not move when a later
        one does — which is the entire point.
        """
        g = self.group_ids(rel, by)
        if not g:
            return _fp.ABSENT
        body = "|".join(f"{k}={g[k]}" for k in sorted(g) if k <= bound)
        return "upto:" + _fp._short(body) if body else "upto:(empty)"

    def input_ids(self, inputs: dict[str, object],
                  assume_unchanged: dict[str, dict] | None = None,
                  slices: dict[str, str] | None = None) -> dict[str, str]:
        """`{rel: fp strategy}` -> `{rel: id}`. The strategy is only used for roots.

        `assume_unchanged` is the cheap mode: a ROOT's id is taken from the record rather
        than computed from the file. Fingerprinting roots is the ONLY I/O a check does —
        6.7s for a 21-file collection over a bucket — and most of the time the question is
        about code or a version, which costs nothing to answer.

        A derived input is a record lookup either way, so nothing is given up there.
        """
        out = {}
        sl = slices or {}
        for rel, how in inputs.items():
            if rel in sl:
                out[rel] = self.slice_id(rel, sl[rel])
                continue
            if assume_unchanged is not None and self.record_of(rel) is None:
                stored = (assume_unchanged.get(rel) or {}).get("id")
                if stored is not None:
                    out[rel] = stored                # a root, taken on trust
                    continue
            out[rel] = self.id_of(rel, how)
        return out

    def prefetch(self, inputs: dict[str, object]) -> None:
        """Fingerprint many roots at once, into the memo `id_of` already reads.

        A per-partition key asks for one root per partition, and over a bucket each of
        those is a fresh connection and a footer read: twenty-one seasons cost two
        MINUTES serially and seconds together. That is the same shape as every other
        one-file-per-season loop in a pipeline — the bug is the loop, not the file.

        A warm-up, never a second source of truth: it computes exactly what `id_of`
        would, and anything it skips `id_of` still answers.
        """
        from .paths import fields
        todo = [(rel, how) for rel, how in inputs.items()
                if not fields(rel) and self.record_of(rel) is None
                and (rel, how if isinstance(how, str) else id(how)) not in self._roots]
        if len(todo) < 2:
            return
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=min(16, len(todo))) as ex:
            list(ex.map(lambda t: self.id_of(*t), todo))

    def instances_of(self, template: str) -> list[str]:
        """Every concrete rel path a template covers: on disk, plus anything stamped.

        Both halves are needed. A root feed exists only on disk. A per-partition artifact
        the pipeline WRITES exists in the state file, and if it has since been deleted the
        collection must still notice it is gone rather than quietly shrinking.
        """
        from .paths import fields
        from .static import path_pattern
        pattern = template
        for f in fields(template):
            pattern = pattern.replace("{" + f + "}", "*")
        with self.iv.bookkeeping():
            # Both roots, in either mode: writes land in out_root whether or not reads
            # fall back to it, so a partition built by this run has to be visible next to
            # the ones that were already there.
            on_disk = set()
            for root in {str(self.iv.data_root): self.iv.data_root,
                         str(self.iv.out_root): self.iv.out_root}.values():
                on_disk |= {str(p)[len(str(root)) + 1:] for p in root.glob(pattern)}
        rx = path_pattern(template)
        # The glob is a cheap PREFILTER, never the answer. `*` per field is greedy and
        # blind to a repeated field, so `raw/{league}/{dataset}/{dataset}_{season}` globs
        # as `raw/*/*/*_*` and swept up `raw/rosters/history/bak_roster_2026_2026-08-02`
        # — a backup file, in a tree the template does not describe, reported forever as
        # an unstamped partition of a feed it has nothing to do with. The pattern, with
        # its back-references, is what decides membership; it already governed `stamped`
        # and simply was not applied here.
        on_disk = {rel for rel in on_disk if rx.match(rel)}
        stamped = {rel for rel in self.records() if rx.match(rel)}
        return sorted(on_disk | stamped)

    def collection_id(self, template: str, how: object = "data") -> str:
        """The id of a whole per-partition feed, e.g. `raw/box/{season}.parquet`.

        It folds each instance's ID, not its fingerprint — so an instance the pipeline
        writes contributes its stamped id, exactly as a single-file input would.
        Fingerprinting here while `id_of` reads the stamp elsewhere makes the collection
        and its members disagree, which shows up as an outer guard saying "stale" over a
        cache saying "nothing to do".

        THE COST IS REAL, and it is why `fp=` is on `reads()`. A collection of ROOTS is
        fingerprinted on every check, so a twenty-one-season feed of 220 MiB files wants
        `fp="rows"` (a footer read) rather than the default full data hash.
        """
        ckey = (template, how if isinstance(how, str) else id(how))
        if ckey in self._collections:
            return self._collections[ckey]
        found = self.instances_of(template)
        if not found:
            out = ABSENT
        elif how == "present":
            # Coverage, not content. A fetch-once archive answers "is it COMPLETE?", and
            # the member list already answers that — so the whole collection costs one
            # listing rather than 4,470 footer reads over a bucket, which is the size of
            # the archive that forced this.
            out = "coll:" + hashlib.sha256("|".join(found).encode()).hexdigest()[:_fp.DIGEST_LEN]
        else:
            # Concurrently: a collection is one-file-per-partition by construction, so
            # folding its members serially is the loop this whole module exists to kill.
            self.prefetch({rel: how for rel in found})
            body = "|".join(f"{rel}={self.id_of(rel, how)}" for rel in found)
            out = "coll:" + hashlib.sha256(body.encode()).hexdigest()[:_fp.DIGEST_LEN]
        self._collections[ckey] = out
        return out

    # ── stamping ──────────────────────────────────────────────────────────────

    def _live_contributions(self, rel: str) -> dict[str, dict]:
        """What OTHER writers have put into `rel`, minus any that no longer write it."""
        entry = self.record_of(rel) or {}
        contrib = entry.get("contrib")
        if contrib is None:
            # A record written before contributions were split. Its whole input map is
            # its writer's share, which is exactly right for the single-writer case that
            # every such record is.
            got = entry.get("in")
            contrib = {entry.get("by", "?"): got} if got else {}
        from . import static as _static
        writers = _static.writers_of(self.iv, rel)
        if not writers:
            return dict(contrib)                    # the scan cannot see it; keep all
        return {w: c for w, c in contrib.items() if w in writers}

    def stamp(self, rel: str, *, spec: Spec, inputs: dict[str, object], by: str,
              fp_value: str | None = None, parts: dict[str, str] | None = None,
              prior: set[str] | None = None,
              slices: dict[str, str] | None = None) -> str:
        """Fingerprint the artifact, fold everything into one id, write the record.

        Called only on a write that completed. A raise inside the write leaves no record,
        so a half-written artifact reads as stale rather than as fresh.

        `inputs` maps each input's rel path to the strategy to use IF it is a root. Both
        the ids and the strategies are stored: the ids are what the next check compares
        against, the strategies are what lets it recompute a root's id at all.

        `prior` names the inputs read with `prior=True` — deliberately the PREVIOUS run's
        copy. They are recorded, because they are lineage, and excluded from the
        comparison, because they cannot pass it: the producer runs LATER in the same
        pipeline, so the value moves after the stamp, every run, for ever. wvorp's
        `rookie_prior` reads the roster snapshot at stage [5c] and `fetch_rosters`
        rewrites it at [8b2]; comparing them made it stale at the end of every run it had
        just been rebuilt in. Saying "a later producer is not an ordering bug" and then
        treating that producer as a staleness trigger are the same fact answered two
        different ways.

        SEVERAL STAGES CAN WRITE ONE ARTIFACT, and then a stamp CONTRIBUTES rather than
        replaces. wvorp's `college_features.parquet` is built by one stage and then
        rewritten by two more, each folding its own block in; the last of those is not
        the only thing that made the file, and a record holding only its reads says the
        college feed had nothing to do with the table. So contributions are kept per
        writer and the identity is the fold of all of them — which is just the honest
        reading of "an artifact's inputs are what went into it".

        A writer that no longer declares this artifact is dropped, so deleting a link
        from a chain removes its contribution rather than freezing it in. Nothing is
        dropped when the scan cannot see the artifact at all, because then absence is
        not evidence.
        """
        p = self.iv.resolve_out(rel)
        with self.iv.bookkeeping():
            if not p.exists():
                raise StateError(
                    f"{rel} was declared written but is not on disk. Nothing is stamped — "
                    f"a stamp means one thing only: THIS code produced THIS file.")
            value = fp_value if fp_value is not None else _fp.compute(p, spec.fp)

        lagged, sl = prior or set(), slices or {}
        mine = {k: {"id": v, "fp": inputs[k] if isinstance(inputs[k], str) else "<callable>",
                    **({"prior": True} if k in lagged else {}),
                    **({"slice": sl[k]} if k in sl else {})}
                for k, v in self.input_ids(inputs, slices=sl).items()}
        contrib = {**self._live_contributions(rel), by: mine}
        ids = {k: v["id"] for c in contrib.values() for k, v in c.items()
               if not v.get("prior")}
        meta = self.metadata_term(spec.policy, spec.code, spec.version)
        new_id = self.compute_id(value, meta, ids, spec.policy)

        entry = {
            "id": new_id,
            "fp": value,
            "meta": meta,
            "fp_how": spec.fp if isinstance(spec.fp, str) else "<callable>",
            "policy": spec.policy,
            "version": spec.version,
            "terminal": spec.terminal,
            "why": spec.why,
            # `in` is the union, flattened, and it is what the staleness check walks.
            # `contrib` is the same thing split by writer, which is what lets the next
            # stamp replace one stage's share without touching another's.
            "in": {k: v for c in contrib.values() for k, v in sorted(c.items())},
            "contrib": {w: dict(sorted(c.items())) for w, c in sorted(contrib.items())},
            "by": by,
            "at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        if parts is not None:
            entry["parts"] = parts
        self._write_record(rel, entry)
        self.records()[rel] = {"state_version": STATE_VERSION, "rel": rel, **entry}
        # A write is the one thing that can move a file's identity mid-process, so both
        # memos go. `updates` rewrites a path this run already fingerprinted, and a new
        # partition changes what its collection contains.
        self._roots.clear()
        self._collections.clear()
        return new_id

    # ── the check ─────────────────────────────────────────────────────────────

    def why_stale(self, rel: str,
                  code: str | None = None,
                  version: str | None = None,
                  fingerprint: bool = True) -> str | None:
        """None if current, else one line naming the component that moved.

        The recomputation uses the artifact's STORED fingerprint — the file has not been
        rewritten since it was stamped, and re-reading it on every check would cost a full
        pass over every artifact in the pipeline to learn nothing.

        `code` and `version` come from the CODE rather than from the record, because the
        record can only describe the last build. Both are DECLARATIONS — facts about the
        source as written — which is why they can be trusted here, unlike a static guess at
        which reads will execute. None means "nobody could tell me" and the stored value
        stands, so turning a flag off stops keying on it rather than invalidating.

        `fingerprint=False` takes ROOTS on trust, using the id in the record instead of
        reading the file. Fingerprinting a root is the only I/O a check does, and it is the
        expensive part — so this answers "is it stale for a reason I can see without
        touching the data?", which is the right question when you want to know what an
        edit broke. A live run always fingerprints.
        """
        from .paths import fields
        if fields(rel):
            # A partitioned artifact: one node in the graph, many files on disk. Current
            # only if every instance is, and the reason names the instance.
            found = self.instances_of(rel)
            if not found:
                # An arm of an if/else this league never takes is not a missing artifact.
                # `_fetch_write` writes a flat path on the parent league and a nested one
                # on a sub-league; a W run only ever takes the first, so the second is
                # empty by construction and saying so every night is noise.
                from .static import branch_siblings
                with self.iv.bookkeeping():
                    taken = any(self.instances_of(sib)
                                for sib in branch_siblings(self.iv, rel))
                if taken:
                    return None
                return "not on disk (no partition of it exists)"
            for inst in found:
                if self.record_of(inst) is None:
                    # A partition on disk that NOTHING here has ever stamped is a ROOT,
                    # by observation rather than inference. Its identity is its bytes,
                    # and that is what dependants already fold — `processed/panel/
                    # player_box.parquet` reads this feed as `coll:…`, a digest of
                    # fingerprints, so an unstamped season contributes exactly as a
                    # stamped one does and nothing downstream is stale.
                    #
                    # The archive is most of the tree: wvorp's refresh fetches ONE season,
                    # so 90 of 93 partitions of `raw/{dataset}/{dataset}_{season}` were
                    # written once in a backfill and no future run will ever stamp them.
                    # Calling them stale every night is a report nobody can act on.
                    #
                    # A partition that HAS a record is still checked, which is what keeps
                    # "did the fetcher actually run?" answerable for the live season: it
                    # was stamped yesterday, so a failure to write it today shows up as a
                    # moved fingerprint rather than as silence.
                    continue
                reason = self.why_stale(inst, code, version, fingerprint)
                if reason is not None:
                    return f"{inst}: {reason}"
            return None

        p = self.iv.resolve_out(rel)
        with self.iv.bookkeeping():
            exists = p.exists()
        if not exists:
            return "not on disk"

        entry = self.record_of(rel)
        if entry is None:
            return "never stamped"

        policy = entry.get("policy", "tracked")
        if policy == "settled":
            return None                     # the question is coverage, not staleness

        stored_inputs = entry.get("in") or {}
        # A `prior=True` input is lineage, not a trigger. Its producer runs LATER in the
        # same pipeline, so its value moves after the stamp on every run — comparing it
        # makes the artifact permanently stale one step behind itself. See `stamp`.
        wanted = {k: v.get("fp", "data") for k, v in stored_inputs.items()
                  if not v.get("prior")}
        sliced = {k: v["slice"] for k, v in stored_inputs.items()
                  if v.get("slice") and k in wanted}

        # THE RECORDED SET GOVERNS, NOT THE DECLARED ONE.
        #
        # It is tempting to treat the static scan as authoritative and call an artifact
        # stale when the code reads something the record has never seen. The trouble is
        # that the two sets are derived by different mechanisms — one by reading source,
        # one by watching a process — and a rebuild cannot reconcile a disagreement about
        # DERIVATION, only about data. So any blind spot becomes a PERMANENT REBUILD:
        # correct output, no error, and the artifact never skips. `tests/test_integration
        # .py::test_a_read_the_scan_cannot_see_does_not_loop` demonstrates it with a dict
        # dispatch. It loops the other way too, on a declared-but-untaken branch.
        #
        # HONESTY ABOUT THE EVIDENCE: the case that prompted this was NOT an unfollowable
        # call. It was `bookkeeping()` failing to suppress input registration, so a
        # manifest's self-inspection read became a data edge. That is fixed. Whether real
        # unfollowable calls are common enough to matter is still unmeasured — `drift`
        # against a real trace is how to find out, and this can be revisited with data.
        #
        # So staleness is decided by the ids of the inputs the last build ACTUALLY read.
        # A declared-vs-recorded gap is real and worth knowing about, but it is a
        # reporting matter: `iv drift` says so against a trace, where the answer
        # is measured rather than inferred.
        #
        # The cost is that ADDING a read does not by itself invalidate. That is what
        # `data_version`, `version=` and `step(code=True)` are for — and it is the rule
        # this pipeline already lives by.
        ids_now = self.input_ids(wanted, None if fingerprint else stored_inputs,
                                 slices=sliced)
        stored_meta = entry.get("meta", "")
        code_now = _part_of(stored_meta, "code") if code is None else code
        version_now = _part_of(stored_meta, "version") if version is None else version
        meta_now = self.metadata_term(policy, code_now, version_now)
        id_now = self.compute_id(entry["fp"], meta_now, ids_now, policy)
        if id_now == entry["id"]:
            return None

        # The id moved. Say WHICH component, because "the id changed" is not actionable.
        if meta_now != stored_meta:
            was_v, now_v = _part_of(stored_meta, "v"), self.iv.data_version
            if was_v != now_v:
                return f"data_version bumped: {was_v} -> {now_v}"
            was_v2 = _part_of(stored_meta, "version")
            if was_v2 != version_now:
                return (f"version bumped: {was_v2 or '(none)'} -> "
                        f"{version_now or '(none)'}")
            was_c = _part_of(stored_meta, "code")
            if was_c != code_now:
                return f"code changed: {was_c or '(not tracked)'} -> {code_now or '(not tracked)'}"
            return f"metadata changed: {stored_meta} -> {meta_now}"
        for k in sorted(ids_now):
            was = (stored_inputs.get(k) or {}).get("id")
            if was == ids_now[k]:
                continue
            if was is None:
                return f"new input: {k}  (not read when this was last built)"
            if self._became_known(k, was):
                # NOT a data change. An input the pipeline writes contributes its
                # STAMPED id; before anything had stamped it, it was read as a root and
                # contributed its fingerprint instead. Same bytes, different term, and
                # it happens exactly once per artifact — on the first run that declares
                # it. Worth saying, because "input moved" reads as "your data changed"
                # and sends someone looking for a change that never happened.
                return (f"input became declared: {k}  {was} -> {ids_now[k]}  "
                        f"(its producer stamped it for the first time; the data is "
                        f"unchanged, and this cannot recur)")
            return f"input moved: {k}  {was} -> {ids_now[k]}"
        return f"id moved: {entry['id']} -> {id_now}"

    def _became_known(self, rel: str, was: str) -> bool:
        """Did `rel`'s id change ONLY because it went from unstamped to stamped?

        True when the old value is what it would have fingerprinted to as a root, and it
        now has a record. That is a transition, not a movement: the same bytes described
        two ways, once before the pipeline knew who wrote them.
        """
        from .paths import fields

        try:
            if fields(rel):
                members = self.instances_of(rel)
                if not members or any(self.record_of(m) is None for m in members):
                    return False
                body = "|".join(
                    f"{m}={_fp.compute(self.iv.resolve(m), self._how_of(m))}"
                    for m in members)
                as_root = "coll:" + hashlib.sha256(
                    body.encode()).hexdigest()[:_fp.DIGEST_LEN]
                return as_root == was
            if self.record_of(rel) is None:
                return False
            with self.iv.bookkeeping():
                p = self.iv.resolve(rel)
                return p.exists() and _fp.compute(p, self._how_of(rel)) == was
        except Exception:
            return False                    # a diagnosis must never raise

    def _how_of(self, rel: str) -> str:
        """The fp strategy an artifact was stamped with, defaulting to the usual one."""
        return (self.record_of(rel) or {}).get("fp_how") or "data"

    def is_current(self, rel: str, code: str | None = None,
                   version: str | None = None, fingerprint: bool = True) -> bool:
        return self.why_stale(rel, code, version, fingerprint) is None


# ── the record file ───────────────────────────────────────────────────────────

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def record_filename(rel: str) -> str:
    """The file one artifact's stamp lives in.

    The readable half is for whoever opens the directory; the digest is what makes it a
    NAME rather than a guess — two rel paths that differ only in a character the sanitiser
    folds (`raw/box/{season}.parquet` and `raw/box_{season}.parquet`) must not share a
    file. The rel path is stored inside the record, so the filename is only an address.
    """
    safe = _UNSAFE.sub("_", rel).lstrip("._")[:96] or "artifact"
    return f"{safe}-{hashlib.sha256(rel.encode()).hexdigest()[:12]}.json"


def _read_record(p) -> dict:
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise StateError(
            f"state record {p} is unreadable: {e}. Fix or delete it — deleting costs a "
            f"rebuild of that one artifact, which is recoverable; guessing is not.") from e
    if raw.get("state_version") != STATE_VERSION:
        raise StateError(
            f"state record {p} is version {raw.get('state_version')}, this iv writes "
            f"{STATE_VERSION}. Delete the directory to rebuild from scratch.")
    if not raw.get("rel"):
        raise StateError(f"state record {p} does not say which artifact it is about.")
    return raw


def read_records(state_dir) -> dict[str, dict]:
    """Every stamp in a state directory, by rel path. For tests and outside tools."""
    d = Path(state_dir) if not hasattr(state_dir, "glob") else state_dir
    return {} if not d.exists() else \
        {r["rel"]: r for r in (_read_record(f) for f in sorted(d.glob("*.json")))}


def _part_of(meta: str, key: str) -> str:
    """One `key=value` chunk out of a metadata term."""
    for chunk in meta.split("|"):
        if chunk.startswith(key + "="):
            return chunk[len(key) + 1:]
    return ""
