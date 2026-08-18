"""Rebuild only the partitions that moved, reuse the rest.

WHY THIS EXISTS. A pipeline run in August adds one game to the current season. Every
earlier season is finished: its inputs cannot change, so every derived row in it is already
correct. But an artifact has ONE id, so it is wholly current or wholly stale, and one game
rebuilds twenty-one seasons.

    reuse   2006..2025   (20 partitions, ~95% of the rows)
    rebuild 2026         (the only one whose inputs moved)

The per-partition key falls straight out of the templates. A stage that reads
`raw/box/{season}.parquet` already declares a per-partition input, so partition 2026's key
is the id of `raw/box/2026.parquet` — no separate per-partition fingerprint to invent and
keep in sync. Inputs WITHOUT the partition key in their path affect every partition, so
they enter every partition's key.

WHAT MAKES THIS SOUND, and it is worth stating because it is the whole safety argument:
**the loop must be causally closed per partition.** Every cross-partition term has to be
backward-looking — a running total up to the current row, a first-seen marker, a career
average — so a 2026 game changes 2026 rows and nothing earlier. A builder with a term that
looks FORWARD across partitions makes this cache wrong, and silently so. That is why the
A/A test is not optional: build incrementally and from scratch, and compare UNSORTED.

WHERE THE STAMPS LIVE. In the artifact's own state record, under `parts`. Not in extra
columns on the artifact: stamping rows would change its schema, which means a version bump
and two bookkeeping columns in front of every downstream consumer. A partition is only
reusable when it has BOTH a matching stamp and actual rows, so a record that outlives the
data it describes cannot be believed.
"""
from __future__ import annotations

import hashlib

from . import static as _static
from .errors import DagioError
from .fingerprint import DIGEST_LEN
from .state import Spec
from .paths import fields as _template_fields
from .paths import render as _render_template
from .paths import render_partial as _render_partial

_MISSING = object()


class PartitionCache:
    """Per-partition reuse for an artifact with a partition column."""

    def __init__(self, iv, output: str, key: str, *, why: str,
                 fp: str = "data", policy: str = "tracked", extra: str = "",
                 part: dict[str, str] | None = None) -> None:
        self.iv = iv
        # TWO names for one artifact, and both are needed. The TEMPLATE is the literal in
        # the source, so it is what the static scan knows it by. The RENDERED path is the
        # file on disk. A pipeline that produces one artifact per dataset — a panel per
        # feed, each season-partitioned — has no other way to stay statically readable.
        self.template = output
        self.part = dict(part or {})
        self.artifact = _render_template(output, part) if part else output
        self.key = key
        self.spec = Spec(why=why, fp=fp, policy=policy)
        self.extra = extra
        self._inputs = _static.inputs_for_artifact(iv, self.template)
        if self._inputs is None:
            raise DagioError(
                f"{self.template} has no single declared producer, so its per-partition inputs "
                f"cannot be read off the code. Run `dagio check`.")
        self._per = self._applicable({t: h for t, h in self._inputs.items()
                                      if key in _template_fields(t)})
        # The same branch rule applies to the whole-artifact inputs: `load()` declares a
        # flat panel and a nested one, and a caller that named no league reads the flat one.
        self._global = self._applicable({t: h for t, h in self._inputs.items()
                                         if key not in _template_fields(t)})
        self._existing = _MISSING
        self._plan_extra: dict[str, str] = {}
        self._fp_of: dict[str, str] = {}

    def _applicable(self, per: dict[str, str]) -> dict[str, str]:
        """Which per-partition templates this partition actually reads.

        The static scan is a UNION over branches, and a producer that picks its path by a
        runtime value declares every branch. `load_raw` reads
        `raw/{dataset}/{dataset}_{season}.parquet` for the tree's own league and
        `raw/{league}/{dataset}/{dataset}_{season}.parquet` for a nested sub-league; only
        one of those is a real input of any given panel.

        Two rules, and between them they pick the branch that was taken:

        A template this partition cannot RENDER is not its input — the missing field names
        a distinction this caller never made. And where one renderable template's fields
        are a strict SUBSET of another's, the caller supplied the extra field on purpose,
        so the more specific one is the branch it took. Without the second rule a sub-league
        panel claims the parent league's flat raw feed — which moves nightly — and rebuilds
        every night for a feed it never opened.
        """
        known = set(self.part) | {self.key}
        fields = {t: set(_template_fields(t)) for t in per}
        ok = {t: h for t, h in per.items() if fields[t] <= known}
        return {t: h for t, h in ok.items()
                if not any(fields[t] < fields[u] for u in ok)}

    # ── the key ───────────────────────────────────────────────────────────────

    def _warm(self, partitions: list[str]) -> None:
        """Fingerprint every root these partitions key on, concurrently, before keying."""
        want = {self._fill(t): how for t, how in self._global.items()}
        for template, how in self._per.items():
            for p in partitions:
                want[self._fill(template, {self.key: self._fp_of.get(p, p)})] = how
        self.iv.state.prefetch(want)

    def _key(self, partition: str) -> str:
        """This partition's id: the metadata, the global inputs, and ITS inputs.

        `fp_of` redirects which partition's inputs govern a row. A walk-forward artifact
        rates cohort C from data restricted to C-1, so cohort 2026 is a function of 2025 —
        a finished period that cannot move. Keyed on its own partition instead, 2026's key
        would turn over every night and refit every cohort for nothing.
        """
        source = self._fp_of.get(partition, partition)
        # An input template may carry the caller's OTHER fields too — a per-dataset,
        # per-season raw feed is `raw/{dataset}/{dataset}_{season}.parquet`, and only
        # `season` is the partition key.
        # `exempt` drops the WHOLE-ARTIFACT inputs, and only those. A walk-forward
        # artifact rates period C from data restricted to C-1, so `box_features` — one
        # file covering every season, moving every night — cannot reach a completed
        # cohort, and folding it in refits all of them for nothing. What CAN reach the
        # row is per-partition by construction: the `{key}` templates below, and whatever
        # the caller states through `extra=` on `plan`.
        ids = {} if self.spec.policy == "exempt" else dict(self.iv.state.input_ids(
            {self._fill(t): how for t, how in self._global.items()}))
        for template, how in self._per.items():
            rel = self._fill(template, {self.key: source})
            ids[rel] = self.iv.state.id_of(rel, how)
        body = "|".join([
            f"meta={self.iv.state.metadata_term(self.spec.policy)}",
            f"extra={self.extra}",
            # A per-partition choice the ids cannot see — which source feed built it, say.
            f"local={self._plan_extra.get(partition, '')}",
            *[f"{k}={ids[k]}" for k in sorted(ids)],
        ])
        return hashlib.sha256(body.encode()).hexdigest()[:DIGEST_LEN]

    def _fill(self, template: str, extra: dict[str, str] | None = None) -> str:
        """Render whatever fields this cache knows; leave the rest as placeholders.

        PARTIAL on purpose. `_key` always renders fully — `_applicable` has already
        dropped anything this partition cannot name — while `commit` wants the raw feed
        with its dataset fixed and its season still free, which is a collection.
        """
        from .paths import fields as _fields
        vals = {**self.part, **(extra or {})}
        have = {k: v for k, v in vals.items() if k in _fields(template)}
        return _render_partial(template, have)

    def _stamps(self) -> dict[str, str]:
        return (self.iv.record_of(self.artifact) or {}).get("parts") or {}

    # ── planning ──────────────────────────────────────────────────────────────

    def existing(self):
        """The artifact as it stands, or None if it is not there.

        The `_MISSING` sentinel rather than None-means-unread: a polars frame has no
        truth value, so `self._existing or None` raises rather than doing what it looks
        like it does.
        """
        if self._existing is _MISSING:
            import polars as pl
            # resolve_OUT. This artifact is an OUTPUT, so the copy that matters is the one
            # THIS pipeline built. Reading it through `resolve()` takes the SHARED tree
            # when overlay is off, so a run reuses rows it did not produce — a full
            # rebuild followed by a reuse-everything pass then loses whatever the rebuild
            # added, which is exactly how the A/A caught it.
            p = self.iv.resolve_out(self.artifact)
            with self.iv.bookkeeping():
                self._existing = pl.read_parquet(p) if p.exists() else None
        return self._existing

    def plan(self, want: list[str], *, extra: dict[str, str] | None = None,
             fp_of: dict[str, str] | None = None,
             force: bool | None = None) -> tuple[list[str], list[str]]:
        """Split `want` into (reuse, rebuild).

        Reuse needs a matching stamp AND rows actually present. Both `extra` and `fp_of`
        are remembered, so `commit` writes the same key the plan was made against.
        """
        self._plan_extra = dict(extra or {})
        self._fp_of = dict(fp_of or {})
        want = [str(w) for w in want]
        if self.iv.force if force is None else force:
            # `--force` has to reach HERE, not only the guard above. Forcing an outer
            # guard while this cache reuses every partition is a rebuild that rebuilds
            # nothing, and it looks exactly like it worked.
            return [], want

        df = self.existing()
        if df is None or self.key not in df.columns:
            return [], want
        stamps = self._stamps()
        have = {str(v) for v in df[self.key].unique().to_list()}
        self._warm([p for p in want if p in have])
        reuse = [p for p in want if p in have and stamps.get(p) == self._key(p)]
        return reuse, [p for p in want if p not in set(reuse)]

    def reused_rows(self, reuse: list[str]):
        import polars as pl
        df = self.existing()
        if df is None or not reuse:
            return pl.DataFrame()
        return df.filter(pl.col(self.key).cast(pl.String).is_in(list(reuse)))

    # ── committing ────────────────────────────────────────────────────────────

    def commit(self, out, built: list[str], reuse: list[str]) -> str:
        """Write the artifact, stamp it, and re-stamp exactly `built + reuse`.

        The artifact's own inputs come from the CODE, not from what this process happened
        to read: a run that reused every partition read nothing, and an empty input map
        would make the artifact permanently current.

        A partition in neither list is dropped from `parts` rather than left behind, so a
        stamp can never outlive the rows it describes.
        """
        p = self.iv.resolve_out(self.artifact)
        p.parent.mkdir(parents=True, exist_ok=True)
        # `str(p)`: polars reads and writes a gs:// URI natively, but handed a CloudPath
        # object it goes through __fspath__, writes cloudpathlib's LOCAL CACHE, and never
        # uploads — exit 0, nothing in the bucket.
        out.write_parquet(str(p))
        keys = sorted(set(built) | set(reuse))
        self._warm(keys)
        parts = {part: self._key(part) for part in keys}
        # `_per | _global`, not `_inputs`: the branches this artifact does not take are
        # not its lineage, and recording them draws an edge nothing ever read.
        #
        # FILLED, and that is not cosmetic. The raw feed is
        # `raw/{dataset}/{dataset}_{season}.parquet`; recorded with `{dataset}` still
        # free, the player_box panel claims a collection over EVERY raw dataset in the
        # tree — a bucket-wide glob and a footer read per file, for feeds it never
        # opened. `season` stays free, because that one really is the growing set.
        inputs = {self._fill(t): how
                  for t, how in {**self._global, **self._per}.items()}
        new_id = self.iv.state.stamp(self.artifact, spec=self.spec, inputs=inputs,
                                     by=self.iv.node(), parts=parts)
        # The artifact on disk is what we just wrote, so the memo from before the write
        # is a lie. It only shows in a process that plans twice — which is a test, and
        # was one: every partition read as "no artifact yet" and rebuilt.
        self._existing = out
        self.iv.record("io", op="write", rel=self.artifact, why=self.spec.why,
                       partitioned_by=self.key, id=new_id,
                       built=sorted(built), reused=sorted(reuse))
        return new_id

    def report(self, reuse: list[str], rebuild: list[str]) -> None:
        print(f"  partitions [{self.artifact}] by {self.key}")
        print(f"    reuse   ({len(reuse):>2}): {_span(reuse)}")
        print(f"    rebuild ({len(rebuild):>2}): {_span(rebuild)}")


def _span(parts: list[str]) -> str:
    if not parts:
        return "—"
    s = sorted(parts)
    return s[0] if len(s) == 1 else ", ".join(s) if len(s) <= 3 else f"{s[0]}..{s[-1]}"


# ── the fan-out helper ────────────────────────────────────────────────────────

def for_each(iv, over, build_one, *, output: str, key: str, why: str,
             part: dict[str, str] | None = None,
             fp: str = "data", policy: str = "tracked",
             extra: dict[str, str] | None = None,
             fp_of: dict[str, str] | None = None, extra_key: str = "",
             force: bool | None = None, quiet: bool = False):
    """Run `build_one(partition)` only for the partitions that moved.

    `build_one` returns the frame for one partition. Everything else — planning, reusing,
    concatenating, checking coverage, writing, stamping — happens here.

    `force` defaults to whatever the enclosing `build_if_needed(force=...)` was given, so
    `--force` reaches BOTH layers without being threaded by hand. Forcing the outer guard
    while this cache reuses every partition is a rebuild that rebuilds nothing, and it
    looks exactly like it worked. Pass it explicitly to override.
    """
    import polars as pl

    force = iv.force if force is None else force

    cache = PartitionCache(iv, output, key, why=why, fp=fp,
                           policy=policy, extra=extra_key, part=part)
    want = [str(p) for p in over]
    reuse, rebuild = cache.plan(want, extra=extra, fp_of=fp_of, force=force)
    if not quiet:
        cache.report(reuse, rebuild)

    built: dict[str, object] = {}
    for part in rebuild:
        frame = build_one(part)
        if frame is None or frame.height == 0:
            raise DagioError(
                f"{output}: build_one({part!r}) produced no rows. A partition that is "
                f"genuinely empty has to say so explicitly — writing a partial artifact "
                f"reads as real downstream.")
        built[part] = frame

    # Assemble in the order the CALLER asked for, not rebuilt-then-reused. Otherwise the
    # row order of the artifact depends on which partitions happened to be stale, and an
    # incremental build is not frame-identical to a full one — which matters, because row
    # order is an input to anything that slices, takes .first(), or sums floats.
    frames = [built[p] if p in built else cache.reused_rows([p]) for p in want]
    out = pl.concat(frames, how="diagonal_relaxed") if len(frames) > 1 else frames[0]
    covered = {str(v) for v in out[key].unique().to_list()}
    if covered != set(want):
        raise DagioError(
            f"{output}: expected partitions {sorted(want)} but built {sorted(covered)}. "
            f"Refusing to write a partial artifact.")
    cache.commit(out, rebuild, reuse)
    return out
