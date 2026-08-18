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
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import fingerprint as _fp
from .errors import StateError
from .fingerprint import ABSENT

STATE_VERSION = 2

# The closed vocabulary. Each entry changes what the id MEANS, not merely what is done
# with it — which is why they live at the write site rather than in a list somewhere.
POLICIES = ("tracked", "manual", "settled", "exempt", "clock")


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

    @property
    def path(self):
        """Where the stamps live.

        Under the OUT root by default, because the state describes what THIS pipeline
        built — a local run writing into a scratch tree must not stamp the shared one.
        `state_path=` overrides it outright, which is what a shadow run over someone
        else's data wants.
        """
        return self.iv.state_path_override or (self.iv.out_root / self.iv.state_rel)

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

        `exempt` drops the input term: a walk-forward artifact fit only on completed
        periods has declared inputs that move nightly and cannot reach it. The metadata is
        then the whole rule, which is exactly the trade — a new period needs a version
        bump to appear.
        """
        body = [f"fp={fp_value}", f"meta={meta}"]
        if policy != "exempt":
            body += [f"{k}={inputs[k]}" for k in sorted(inputs)]
        return hashlib.sha256("|".join(body).encode()).hexdigest()[:_fp.DIGEST_LEN]

    # ── the state file ────────────────────────────────────────────────────────

    def load(self) -> dict:
        """The whole state file.

        Fails LOUD on malformed JSON. Returning `{}` there would make invalidation a no-op
        and every builder rebuild forever, which reads as "the cache does not work" rather
        than as "the state file is corrupt". A missing file is the only legal empty state.
        """
        if self._cache is not None:
            return self._cache
        p = self.path
        with self.iv.bookkeeping():
            if not p.exists():
                self._cache = {"version": STATE_VERSION, "artifacts": {}}
                return self._cache
            try:
                raw = json.loads(p.read_text())
            except (json.JSONDecodeError, OSError) as e:
                raise StateError(
                    f"state file {p} is unreadable: {e}. Fix or delete it — deleting "
                    f"costs a full rebuild, which is recoverable; guessing is not.") from e
        if raw.get("version") != STATE_VERSION:
            raise StateError(
                f"state file {p} is version {raw.get('version')}, this invalidator writes "
                f"{STATE_VERSION}. Delete it to rebuild from scratch.")
        self._cache = raw
        return self._cache

    def save(self, data: dict) -> None:
        p = self.path
        body = json.dumps(data, indent=2, sort_keys=True)
        with self.iv.bookkeeping():
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(p, Path):
                tmp = p.with_suffix(p.suffix + ".tmp")
                tmp.write_text(body)
                os.replace(tmp, p)          # atomic, so a crash cannot leave it torn
            else:
                p.write_text(body)

    def record_of(self, rel: str) -> dict | None:
        return self.load()["artifacts"].get(rel)

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

    def input_ids(self, inputs: dict[str, object],
                  assume_unchanged: dict[str, dict] | None = None) -> dict[str, str]:
        """`{rel: fp strategy}` -> `{rel: id}`. The strategy is only used for roots.

        `assume_unchanged` is the cheap mode: a ROOT's id is taken from the record rather
        than computed from the file. Fingerprinting roots is the ONLY I/O a check does —
        6.7s for a 21-file collection over a bucket — and most of the time the question is
        about code or a version, which costs nothing to answer.

        A derived input is a record lookup either way, so nothing is given up there.
        """
        out = {}
        for rel, how in inputs.items():
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
        stamped = {rel for rel in self.load()["artifacts"] if rx.match(rel)}
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
        else:
            # Concurrently: a collection is one-file-per-partition by construction, so
            # folding its members serially is the loop this whole module exists to kill.
            self.prefetch({rel: how for rel in found})
            body = "|".join(f"{rel}={self.id_of(rel, how)}" for rel in found)
            out = "coll:" + hashlib.sha256(body.encode()).hexdigest()[:_fp.DIGEST_LEN]
        self._collections[ckey] = out
        return out

    # ── stamping ──────────────────────────────────────────────────────────────

    def stamp(self, rel: str, *, spec: Spec, inputs: dict[str, object], by: str,
              fp_value: str | None = None, parts: dict[str, str] | None = None) -> str:
        """Fingerprint the artifact, fold everything into one id, write the record.

        Called only on a write that completed. A raise inside the write leaves no record,
        so a half-written artifact reads as stale rather than as fresh.

        `inputs` maps each input's rel path to the strategy to use IF it is a root. Both
        the ids and the strategies are stored: the ids are what the next check compares
        against, the strategies are what lets it recompute a root's id at all.
        """
        p = self.iv.resolve_out(rel)
        with self.iv.bookkeeping():
            if not p.exists():
                raise StateError(
                    f"{rel} was declared written but is not on disk. Nothing is stamped — "
                    f"a stamp means one thing only: THIS code produced THIS file.")
            value = fp_value if fp_value is not None else _fp.compute(p, spec.fp)

        ids = self.input_ids(inputs)
        meta = self.metadata_term(spec.policy, spec.code, spec.version)
        new_id = self.compute_id(value, meta, ids, spec.policy)

        data = self.load()
        entry = {
            "id": new_id,
            "fp": value,
            "meta": meta,
            "fp_how": spec.fp if isinstance(spec.fp, str) else "<callable>",
            "policy": spec.policy,
            "version": spec.version,
            "terminal": spec.terminal,
            "why": spec.why,
            "in": {k: {"id": ids[k],
                       "fp": inputs[k] if isinstance(inputs[k], str) else "<callable>"}
                   for k in sorted(ids)},
            "by": by,
            "at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        }
        if parts is not None:
            entry["parts"] = parts
        data["artifacts"][rel] = entry
        self.save(data)
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
                return "not on disk (no partition of it exists)"
            for inst in found:
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
        wanted = {k: v.get("fp", "data") for k, v in stored_inputs.items()}

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
        ids_now = self.input_ids(wanted, None if fingerprint else stored_inputs)
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
            if was != ids_now[k]:
                return f"input moved: {k}  {was} -> {ids_now[k]}"
        return f"id moved: {entry['id']} -> {id_now}"

    def is_current(self, rel: str, code: str | None = None,
                   version: str | None = None, fingerprint: bool = True) -> bool:
        return self.why_stale(rel, code, version, fingerprint) is None


def _part_of(meta: str, key: str) -> str:
    """One `key=value` chunk out of a metadata term."""
    for chunk in meta.split("|"):
        if chunk.startswith(key + "="):
            return chunk[len(key) + 1:]
    return ""
