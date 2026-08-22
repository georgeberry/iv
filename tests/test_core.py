"""The primitives, end to end, on a real directory tree.

Every assertion here is about a DECISION — did this stage run or skip — because that is the
only thing the package exists to get right.
"""
from __future__ import annotations

import polars as pl
import pytest

from iv import shards as _sh
from iv.core import Pipeline
from iv.errors import DeclError, StateError


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(root=tmp_path / "data", stage_dir=tmp_path / "stage",
                    project_root=tmp_path)


def seed(iv, dataset, part=None, n=3, extra=0, why="an upstream feed"):
    """Put a shard on disk the way a fetcher would."""
    with iv.writes(dataset, why=why, part=part) as out:
        pl.DataFrame({"a": range(n), "b": [x + extra for x in range(n)]}).write_parquet(out)


def rows(iv, dataset):
    return pl.read_parquet(iv.reads(dataset, why="check")).height


def out_step(iv, ran=None):
    """`raw/feed/` -> `processed/out/`. Literal datasets, because a step needs them."""
    @iv.step(why="passthrough")
    def build():
        if ran is not None:
            ran.append(1)
        with iv.writes("processed/out/", why="passthrough") as out:
            pl.read_parquet(iv.reads("raw/feed/", why="the upstream")).write_parquet(out)
    return build


def end_step(iv, ran=None):
    """`processed/mid/` -> `processed/end/`, the expensive one in cascade tests."""
    @iv.step(why="the expensive one")
    def build():
        if ran is not None:
            ran.append(1)
        with iv.writes("processed/end/", why="the expensive one") as out:
            pl.read_parquet(iv.reads("processed/mid/", why="mid")).write_parquet(out)
    return build


# ── the basic loop ────────────────────────────────────────────────────────────

def test_a_step_runs_then_skips(iv):
    seed(iv, "raw/feed/")
    ran = []
    build = out_step(iv, ran)
    assert build() is True and len(ran) == 1
    assert build() is False and len(ran) == 1
    assert rows(iv, "processed/out/") == 3


def test_a_moved_upstream_rebuilds(iv):
    seed(iv, "raw/feed/")
    build = out_step(iv)
    build()
    seed(iv, "raw/feed/", n=9)
    assert iv.why_stale("processed/out/").startswith("its inputs moved")
    assert build() is True and rows(iv, "processed/out/") == 9


def test_a_real_change_reaches_downstream(iv):
    seed(iv, "raw/feed/")
    ran = []
    build = out_step(iv, ran)
    build()
    seed(iv, "raw/feed/", extra=99)
    build()
    assert len(ran) == 2


# ── the skip check sees every output ──────────────────────────────────────────

def test_a_stage_with_several_outputs_is_checked_on_all_of_them(iv):
    """One fit, several outputs. Losing any one of them must bring the stage back."""
    seed(iv, "raw/feed/")
    ran = []

    @iv.step(why="one computation, three outputs")
    def build():
        ran.append(1)
        df = pl.read_parquet(iv.reads("raw/feed/", why="the upstream"))
        with iv.writes("out/ratings/", why="ratings") as o:
            df.write_parquet(o)
        with iv.writes("out/careers/", why="careers") as o:
            df.select("a").write_parquet(o)
        with iv.writes("out/summary/", why="summary") as o:
            df.head(1).write_parquet(o)

    assert build.outputs == ("out/ratings/", "out/careers/", "out/summary/")
    build()
    assert build() is False and len(ran) == 1

    # A crash between two of the writes leaves exactly this state.
    import shutil
    shutil.rmtree(iv.resolve_out("out/careers/"))
    assert build() is True, "a missing output must bring the stage back"
    assert len(ran) == 2
    assert iv.resolve_out("out/careers/").exists()


def test_a_step_that_writes_nothing_never_skips_because_not_knowing_means_running(iv):
    """Not knowing what a stage produces means running it. The safe direction."""
    ran = []

    @iv.step(why="declares no output")
    def build():
        ran.append(1)

    assert build.outputs == ()
    build(); build()
    assert len(ran) == 2


def test_part_and_code_are_ambient_for_the_body(iv):
    seed(iv, "raw/feed/")

    @iv.step(why="one partition of a sharded output", part={"season": "2026"})
    def build():
        with iv.writes("processed/by_season/", why="one season") as out:
            pl.read_parquet(iv.reads("raw/feed/", why="the upstream")).write_parquet(out)

    build()
    assert list(_sh.current_shards(iv.resolve_out("processed/by_season/"))) == ["season=2026"]
    assert build() is False


# ── code and versions stop at their own stage ─────────────────────────────────

def test_a_code_edit_alone_does_not_rerun_anything(iv):
    """Editing a step is invisible. Only a file can invalidate.

    The step used to be hashed, which read as covering a logic change and did not: the
    hash sees the decorated function, and the logic it calls lives in modules the hash
    never opens. A half-covering mechanism is worse than none, because it is the half you
    do not have that you stop checking. So the rule is uniform — an artifact moves when an
    input file moves, and a builder's own version is an input like any other.
    """
    seed(iv, "raw/feed/")

    def build(extra):
        @iv.step(why="passthrough")
        def mid():
            df = pl.read_parquet(iv.reads("raw/feed/", why="the upstream"))
            if extra:
                df = df.select(pl.all())
            with iv.writes("processed/mid/", why="mid") as o:
                df.write_parquet(o)
        return mid()

    assert build(False) is True
    assert build(True) is False, "an edit is not a file, so nothing moved"


def test_a_version_is_a_file_and_only_moves_what_reads_it(iv):
    seed(iv, "raw/feed/")
    iv.constants("config/model/", why="what the fits answer to", v="m1")
    end_ran = []

    def build_all():
        @iv.step(why="model output")
        def modelled():
            iv.reads("config/model/", why="a model change must rebuild this")
            with iv.writes("processed/modelled/", why="model output") as o:
                pl.read_parquet(iv.reads("raw/feed/", why="the upstream")).write_parquet(o)
        @iv.step(why="no model in it")
        def plain():
            with iv.writes("processed/plain/", why="no model in it") as o:
                pl.read_parquet(iv.reads("raw/feed/", why="the upstream")).write_parquet(o)

        @iv.step(why="the expensive one")
        def end():
            end_ran.append(1)
            with iv.writes("processed/end/", why="the expensive one") as o:
                pl.read_parquet(iv.reads("processed/modelled/", why="mid")).write_parquet(o)
        return modelled(), plain(), end()

    build_all()
    assert len(end_ran) == 1
    iv.constants("config/model/", why="what the fits answer to", v="m2")
    ran_modelled, ran_plain, ran_end = build_all()
    assert ran_modelled is True, "it declared it reads the model version"
    assert ran_plain is False, "it did not"
    assert ran_end is False and len(end_ran) == 1, "the numbers did not move"


def test_constants_are_idempotent_and_touch_nothing_when_unchanged(iv):
    iv.constants("config/model/", why="the model", v="m1")
    before = sorted(x.name for x in iv.resolve_out("config/model/").iterdir())
    iv.constants("config/model/", why="the model", v="m1")
    assert sorted(x.name for x in iv.resolve_out("config/model/").iterdir()) == before


def test_the_clock_is_a_file(iv, monkeypatch):
    """What a "re-run daily" policy used to be: the day is an upstream like any other."""
    import datetime as _d
    clock = lambda: iv.constants("config/today/", why="poll once a day",
                                 date=_d.date.today().isoformat())
    clock()
    fetches = []

    @iv.step(why="a polled feed")
    def fetch():
        iv.reads("config/today/", why="poll once a day")
        iv.external("some/api", why="the upstream service")
        fetches.append(1)
        with iv.writes("raw/feed/", why="a polled feed") as o:
            pl.DataFrame({"a": [1]}).write_parquet(o)

    fetch()
    assert fetch() is False, "same day, no re-fetch"

    class Tomorrow(_d.date):
        @classmethod
        def today(cls):
            return _d.date(2099, 1, 1)
    monkeypatch.setattr("iv.core._dt.date", Tomorrow)

    clock()
    assert fetch() is True and len(fetches) == 2


def test_a_stage_that_reads_no_clock_never_re_runs(iv):
    """Fetch-once history, and now it is a fact about the reads rather than a label."""
    calls = []

    @iv.step(why="fetch-once history")
    def build():
        iv.external("sports-reference", why="a page that will not change")
        calls.append(1)
        with iv.writes("raw/archive/", why="fetch-once history") as o:
            pl.DataFrame({"a": [1]}).write_parquet(o)

    build()
    assert build() is False and len(calls) == 1


def test_a_named_partition_that_disappears_is_named(iv):
    """A stage that named the seasons it wants says which one went missing."""
    for s in ("2025", "2026"):
        seed(iv, "raw/feed/", part={"season": s})

    @iv.step(why="two named seasons")
    def build():
        got = iv.reads("raw/feed/", why="exactly these two",
                       where={"season": ["2025", "2026"]})
        with iv.writes("processed/out/", why="two named seasons") as out:
            pl.read_parquet(got).write_parquet(out)

    build()
    _sh.current_shards(iv.resolve("raw/feed/"))["season=2026"].path.unlink()
    assert "no shard for season=2026" in iv.why_stale("processed/out/")


def test_a_vanished_shard_from_a_WHOLE_dataset_read_is_a_moved_input(iv):
    """Read everything and the dataset simply has different contents now."""
    for s in ("2025", "2026"):
        seed(iv, "raw/feed/", part={"season": s})
    build = out_step(iv)
    build()
    _sh.current_shards(iv.resolve("raw/feed/"))["season=2026"].path.unlink()
    assert "its inputs moved" in iv.why_stale("processed/out/")
    assert build() is True


# ── failure never records ─────────────────────────────────────────────────────

def test_a_body_that_raises_records_nothing(iv):
    seed(iv, "raw/feed/")

    @iv.step(why="doomed")
    def build():
        with iv.writes("processed/out/", why="doomed"):
            iv.reads("raw/feed/", why="the upstream")
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        build()
    assert not iv.resolve_out("processed/out/").exists()
    assert "not on disk" in iv.why_stale("processed/out/")


def test_writing_nothing_is_an_error_unless_declared(iv):
    with pytest.raises(DeclError, match="allow_missing"):
        with iv.writes("processed/out/", why="nothing"):
            pass
    with iv.writes("processed/out/", why="nothing yet", allow_missing=True):
        pass
    assert "not on disk" in iv.why_stale("processed/out/")


def test_a_read_that_selects_nothing_raises(iv):
    with pytest.raises(StateError, match="selected no shards"):
        iv.reads("raw/missing/", why="not there")
    assert iv.reads("raw/missing/", why="not there", optional=True) == []


def test_why_is_required(iv):
    with pytest.raises(DeclError, match="why="):
        iv.reads("raw/feed/", why="")
    with pytest.raises(DeclError, match="why="):
        iv.step(why="")


# ── for_each ──────────────────────────────────────────────────────────────────

def test_for_each_builds_once_then_reuses(iv):
    for s in ("2024", "2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    built = []

    def one(season, out):
        built.append(season)
        pl.read_parquet(iv.reads("raw/box/", why="raw box",
                                 where={"season": [iv.PART]})).write_parquet(out)

    run = lambda: iv.for_each(["2024", "2025", "2026"], one, dataset="processed/feat/",
                              key="season", why="per-season features", quiet=True)
    assert sorted(run()) == ["2024", "2025", "2026"]
    assert run() == []
    seed(iv, "raw/box/", part={"season": "2026"}, extra=5)
    assert run() == ["2026"], "one new season rebuilds one season"
    assert len(built) == 4


def test_for_each_rebuilds_a_shard_deleted_off_disk(iv):
    """The counterpart to the loss in test_a_dataset_asked_about_as_a_whole_...: with no
    index there is nothing that remembers a shard was ever there, so being asked about BY
    NAME is what catches a gap. `for_each` iterates an explicit list, so every partition is
    asked about whether or not it is on disk — and only the missing one is rebuilt."""
    for s in ("2024", "2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    built = []

    def one(season, out):
        built.append(season)
        pl.read_parquet(iv.reads("raw/box/", why="raw box",
                                 where={"season": [iv.PART]})).write_parquet(out)

    run = lambda: iv.for_each(["2024", "2025", "2026"], one, dataset="processed/feat/",
                              key="season", why="per-season features", quiet=True)
    assert sorted(run()) == ["2024", "2025", "2026"]
    assert run() == []

    _sh.current_shards(iv.resolve_out("processed/feat/"))["season=2025"].path.unlink()
    assert run() == ["2025"], "the gap is rebuilt, and nothing else is"
    assert sorted(_sh.current_shards(iv.resolve_out("processed/feat/"))) == \
        ["season=2024", "season=2025", "season=2026"]
    assert built == ["2024", "2025", "2026", "2025"]


def test_a_walk_forward_partition_cannot_see_the_future(iv):
    """Past-only is structural: the shard is never opened, not opened-and-filtered."""
    for s in ("2024", "2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))

    def one(season, out):
        past = iv.reads("raw/box/", why="seasons strictly before this cohort",
                        where={"season": {"lt": iv.PART}})
        assert all(f"season={season}" not in p.name for p in past)
        pl.read_parquet(past).write_parquet(out)

    run = lambda: iv.for_each(["2025", "2026"], one, dataset="processed/cohort/",
                              key="season", why="a cohort fit on prior seasons", quiet=True)
    assert sorted(run()) == ["2025", "2026"]
    assert run() == []
    seed(iv, "raw/box/", part={"season": "2026"}, extra=7)
    assert run() == [], "no cohort reads 2026, so a change to it moves nothing"
    seed(iv, "raw/box/", part={"season": "2025"}, extra=7)
    assert run() == ["2026"], "only the cohort that read 2025 moves; 2025's reads 2024"


def test_an_explicit_selection_is_a_coverage_claim(iv):
    seed(iv, "raw/box/", part={"season": "2024"})
    with pytest.raises(StateError, match="2025"):
        iv.reads("raw/box/", why="two seasons", where={"season": ["2024", "2025"]})


# ── update_file_on_disk ───────────────────────────────────────────────────────

def test_an_update_read_is_lineage_or_a_stage_is_stale_against_its_own_last_output(iv):
    """Read-modify-write: a stage must not be stale against its own last output."""
    seed(iv, "raw/feed/")

    @iv.step(why="a running history")
    def build():
        iv.reads("config/today/", why="append once a day", optional=True)
        have = iv.reads("raw/log/", why="yesterday's copy", update_file_on_disk=True,
                        optional=True)
        old = pl.read_parquet(have) if have else pl.DataFrame(schema={"a": pl.Int64})
        new = pl.read_parquet(iv.reads("raw/feed/", why="today")).select("a")
        with iv.writes("raw/log/", why="a running history") as out:
            pl.concat([old, new]).write_parquet(out)

    build()
    assert rows(iv, "raw/log/") == 3
    assert iv.why_stale("raw/log/") is None, \
        "its own last output must not be an upstream of itself"
    assert build() is False


# ── ordering ──────────────────────────────────────────────────────────────────

def test_reads_come_back_in_a_stable_semantic_order(iv):
    for s in (2011, 2006, 2026, 2019):
        seed(iv, "raw/box/", part={"season": str(s)}, extra=s)
    seen = {tuple(p.name for p in iv.reads("raw/box/", why="all")) for _ in range(6)}
    assert len(seen) == 1
    got = iv.reads("raw/box/", why="all")
    assert [p.name.split(".")[0] for p in got] == \
        ["season=2006", "season=2011", "season=2019", "season=2026"]


def test_a_computed_dataset_inside_a_step_is_refused(iv):
    """Skipping it silently would leave the skip check with an incomplete output list."""
    target = "processed/out/"
    with pytest.raises(DeclError, match="LITERAL"):
        @iv.step(why="names its output with a variable")
        def build():
            with iv.writes(target, why="unreadable without running it") as out:
                pl.DataFrame({"a": [1]}).write_parquet(out)


def test_a_read_of_the_whole_dataset_notices_a_brand_new_partition(iv):
    """A joint fit over every season must re-run when a season is added.

    The counterpart of the walk-forward test: there, a new partition must NOT move an
    earlier cohort. Here it must move the fit. The difference is whether the read named a
    selection or asked for everything.
    """
    for s in ("2024", "2025"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    ran = []

    @iv.step(why="one joint fit over every season")
    def fit():
        ran.append(1)
        every = pl.read_parquet(iv.reads("raw/box/", why="every season at once"))
        with iv.writes("processed/xpm/", why="the fit") as out:
            every.head(1).write_parquet(out)

    fit()
    assert fit() is False and len(ran) == 1
    seed(iv, "raw/box/", part={"season": "2026"}, extra=2026)
    assert "its inputs moved" in iv.why_stale("processed/xpm/")
    assert fit() is True and len(ran) == 2


def test_a_write_outside_a_step_does_not_inherit_the_last_stage_s_reads(iv):
    """Otherwise a bare `writes` records whatever was read most recently as its upstream."""
    seed(iv, "raw/box/", part={"season": "2019"})

    @iv.step(why="reads the feed")
    def build():
        iv.reads("raw/box/", why="every season")
        with iv.writes("processed/out/", why="passthrough") as out:
            pl.DataFrame({"a": [1]}).write_parquet(out)

    build()
    seed(iv, "raw/box/", part={"season": "2020"})     # a bare write, after a step ran
    fresh = _sh.current_shards(iv.resolve_out("raw/box/"))["season=2020"]
    assert fresh.key == "", "a raw shard has no upstream, so its name carries no key"


# ── the range a predicate matched ─────────────────────────────────────────────

def test_a_partition_appearing_inside_the_range_read_forces_a_rebuild(iv):
    """A predicate cannot be replayed, so the record keeps the SPAN it matched.

    Backfilling a season EARLIER than a cohort's bound would have been selected, so the fit
    is now built from less than it should be and has to run again.
    """
    for s in ("2019", "2020"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))

    @iv.step(why="a cohort fit on every season before 2021")
    def cohort():
        past = iv.reads("raw/box/", why="seasons before this cohort",
                        where={"season": {"lt": "2021"}})
        with iv.writes("processed/cohort/", why="the cohort fit") as out:
            pl.read_parquet(past).write_parquet(out)

    cohort()
    assert cohort() is False

    seed(iv, "raw/box/", part={"season": "2022"}, extra=2022)
    assert cohort() is False, "later than the bound — the walk-forward guarantee"

    seed(iv, "raw/box/", part={"season": "2018"}, extra=2018)
    assert "its inputs moved" in iv.why_stale("processed/cohort/"), \
        "2018 is below the bound, so re-running the rule now selects it"
    assert cohort() is True


def test_unreadable_source_raises_rather_than_disabling_the_skip_check(iv):
    """It used to return no outputs, which meant every stage ran every time, silently."""
    ns = {"iv": iv}
    exec(compile("def build():\n    pass\n", "<string>", "exec"), ns)
    with pytest.raises(DeclError, match="cannot read the source"):
        iv.step(why="defined with exec, so it has no source")(ns["build"])


def test_a_dataset_this_stage_writes_is_never_its_own_upstream(iv):
    """However it is read. Folding an artifact into its own key would move that key every
    time it was built, so the stage would chase its own output and never settle."""
    ran = []

    @iv.step(why="rewrites the feed in place")
    def build():
        ran.append(1)
        iv.reads("raw/feed/", why="the copy on disk", update_file_on_disk=True,
                 optional=True)
        with iv.writes("raw/feed/", why="rewritten in place") as out:
            pl.DataFrame({"a": [1]}).write_parquet(out)

    build(); build(); build()
    assert len(ran) == 1 and iv.why_stale("raw/feed/") is None


def test_a_stage_whose_only_read_is_its_own_copy_on_disk_runs_once(iv):
    """Documented, not fixed: nothing can move it, so it never re-runs.

    That is correct for fetch-once history and wrong for an accumulator, and the two are
    indistinguishable from here — which is why `iv check` reports it rather than guessing.
    """
    calls = []

    @iv.step(why="appends, but reads only its own last copy")
    def build():
        calls.append(1)
        have = iv.reads("raw/solo/", why="yesterday's copy", update_file_on_disk=True,
                        optional=True)
        old = pl.read_parquet(have) if have else pl.DataFrame(schema={"n": pl.Int64})
        with iv.writes("raw/solo/", why="a running log") as out:
            pl.DataFrame({"n": list(range(len(old) + 1))}).write_parquet(out)

    build(); build(); build()
    assert len(calls) == 1 and iv.why_stale("raw/solo/") is None


def test_writing_through_a_path_reads_handed_back_is_refused(iv):
    """A shard's name is a fingerprint of its contents. Overwriting one makes it a lie."""
    seed(iv, "raw/feed/")
    got = iv.reads("raw/feed/", why="the upstream")[0]
    with pytest.raises(DeclError, match="handed back"):
        got.write_bytes(b"clobbered")


def test_verify_catches_a_shard_whose_contents_no_longer_match_its_name(iv):
    seed(iv, "raw/feed/")
    assert iv.verify("raw/feed/") == []
    shard = _sh.current_shards(iv.resolve_out("raw/feed/"))[""]
    pl.DataFrame({"a": [9, 9, 9]}).write_parquet(shard.path)
    assert "the file was changed after it was committed" in iv.verify("raw/feed/")[0]


def test_verify_reports_shards_of_one_dataset_that_disagree_on_columns(iv):
    """A merged file guaranteed one schema; a directory of shards does not."""
    with iv.writes("raw/box/", why="an old season", part={"season": "2006"}) as out:
        pl.DataFrame({"a": [1], "reason": ["dnp"]}).write_parquet(out)
    with iv.writes("raw/box/", why="a new season", part={"season": "2026"}) as out:
        pl.DataFrame({"a": [1]}).write_parquet(out)
    problems = iv.verify("raw/box/")
    assert any("SCHEMA DRIFT" in p and "reason" in p for p in problems)


def test_verify_is_quiet_when_every_shard_agrees(iv):
    for s in ("2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    assert iv.verify("raw/box/") == []


def test_reading_shards_reproduces_a_total_sort_over_the_merged_frame(iv):
    """The property SVI minibatching depends on, checked rather than reasoned about.

    The old builder concatenated every season then sorted on (season, game_id, row) —
    load-bearing, because minibatches are contiguous slices with shuffle=False. Sharded,
    there is no concat: each season is written on its own and `reads` hands them back in
    season order. Those agree exactly when each shard is internally sorted on the rest of
    the key, and this asserts it on a shuffled build order so a lucky ordering cannot pass.
    """
    import random
    rows = [{"season": s, "game_id": g, "row": r}
            for s in ("2006", "2011", "2019", "2026")
            for g in range(4) for r in range(3)]
    merged = pl.DataFrame(rows).sort(["season", "game_id", "row"])

    order = ["2019", "2026", "2006", "2011"]        # deliberately not season order
    random.Random(0).shuffle(order)
    for s in order:
        with iv.writes("processed/possessions/", why="one season", part={"season": s}) as out:
            pl.DataFrame([r for r in rows if r["season"] == s]) \
              .sort(["game_id", "row"]).write_parquet(out)

    got = pl.read_parquet(iv.reads("processed/possessions/", why="every season"))
    assert got.equals(merged), "sharded read order must equal the merged total sort"


def test_a_rebuilt_shard_does_not_disturb_the_read_order(iv):
    """One new game rebuilds one season; the other twenty must land where they were."""
    for s in ("2024", "2025", "2026"):
        with iv.writes("processed/possessions/", why="one season", part={"season": s}) as out:
            pl.DataFrame({"season": [s] * 3, "row": [0, 1, 2]}).write_parquet(out)
    before = pl.read_parquet(iv.reads("processed/possessions/", why="every season"))

    with iv.writes("processed/possessions/", why="one season", part={"season": "2025"}) as out:
        pl.DataFrame({"season": ["2025"] * 4, "row": [0, 1, 2, 3]}).write_parquet(out)
    after = pl.read_parquet(iv.reads("processed/possessions/", why="every season"))

    assert after.filter(pl.col("season") != "2025").equals(
        before.filter(pl.col("season") != "2025"))
    assert after["season"].to_list() == ["2024"] * 3 + ["2025"] * 4 + ["2026"] * 3


def test_a_dataset_may_hold_something_other_than_a_table(iv):
    """A fitted model is a file. There is no reason it should sit outside the tree.

    It used to have a bespoke side cache, keyed by hand, invisible to the graph and
    unchecked. As a dataset it is one writer and two readers like anything else — the only
    thing that differs is how its contents are digested, which the file type says.
    """
    import pickle
    seed(iv, "processed/possessions/")
    iv.constants("config/model/", why="the knobs the fit shape depends on", half_life=4.0)
    fits = []

    @iv.step(why="the joint fit, computed once and read by two stages")
    def fit():
        fits.append(1)
        iv.reads("config/model/", why="a knob change must refit")
        poss = pl.read_parquet(iv.reads("processed/possessions/", why="the design matrix"))
        with iv.writes("processed/rapm_fit/", why="the fitted model", ext=".pkl") as out:
            out.write_bytes(pickle.dumps({"betas": poss["a"].to_list()}))

    fit()
    assert fit() is False and len(fits) == 1

    got = pickle.loads(iv.reads("processed/rapm_fit/", why="the fit")[0].read_bytes())
    assert got["betas"] == [0, 1, 2]

    iv.constants("config/model/", why="the knobs the fit shape depends on", half_life=3.5)
    assert "its inputs moved" in iv.why_stale("processed/rapm_fit/")
    assert fit() is True and len(fits) == 2


def test_an_unknown_file_type_has_no_fingerprint_and_says_so(iv):
    with pytest.raises(DeclError, match="no way to fingerprint"):
        with iv.writes("processed/thing/", why="a format nothing can digest",
                       ext=".xlsx") as out:
            out.write_bytes(b"x")


def test_a_dataset_asked_about_as_a_whole_is_current_iff_every_shard_is(iv):
    """One computation, many shards. `box_features` has career-cumulative terms, so it
    cannot be built per season — but it can be WRITTEN per season, and then the stage's
    question is about all of them."""
    seed(iv, "raw/box/")
    ran = []

    @iv.step(why="one pass over every season, written out per season")
    def build():
        ran.append(1)
        src = pl.read_parquet(iv.reads("raw/box/", why="the upstream"))
        for season in ("2024", "2025", "2026"):
            with iv.writes("processed/box_features/", part={"season": season},
                           why="the box matrix for one season") as out:
                src.with_columns(pl.lit(season).alias("season")).write_parquet(out)

    build()
    assert sorted(_sh.current_shards(iv.resolve_out("processed/box_features/"))) == \
        ["season=2024", "season=2025", "season=2026"]
    assert build() is False, "every shard is current"

    # A DELETED SHARD IS NO LONGER NOTICED, and this is the one thing the index bought
    # that nothing else does. Every shard of one pass carries its own key, so the two that
    # are left still match and the stage skips — there is nothing on disk that says a third
    # was ever expected. `for_each` is unaffected: it iterates an explicit list, so a
    # missing partition is still rebuilt.
    _sh.current_shards(iv.resolve_out("processed/box_features/"))["season=2025"].path.unlink()
    assert iv.why_stale("processed/box_features/") is None
    assert build() is False and len(ran) == 1

    seed(iv, "raw/box/", extra=99)
    assert build() is True and len(ran) == 2, "a moved upstream still rebuilds all of them"


def test_adding_a_dependency_reruns_the_stage(iv):
    """A read added since the last build must fire, or it is dead for good.

    Staleness compares the inputs a build RECORDED. A newly declared read is not in that
    record, so on its own it cannot trigger: the stage skips, never runs, never records the
    new input, and the dependency silently does nothing forever. The declared-reads digest
    is what closes that loop.
    """
    seed(iv, "raw/feed/")
    seed(iv, "raw/extra/")

    def one_input():
        @iv.step(why="passthrough")
        def mid():
            with iv.writes("processed/mid/", why="mid") as o:
                pl.read_parquet(iv.reads("raw/feed/", why="the upstream")).write_parquet(o)
        return mid()

    def two_inputs():
        @iv.step(why="passthrough")
        def mid():
            iv.reads("raw/extra/", why="a dependency added after the first build")
            with iv.writes("processed/mid/", why="mid") as o:
                pl.read_parquet(iv.reads("raw/feed/", why="the upstream")).write_parquet(o)
        return mid()

    assert one_input() is True
    assert one_input() is False
    assert two_inputs() is True, "the declared inputs changed"
    assert two_inputs() is False, "and now the new one is recorded"

    # And having recorded it, the new dependency actually works.
    seed(iv, "raw/extra/", n=9)
    assert two_inputs() is True, "raw/extra/ moved"


def test_an_undeclared_read_of_the_data_tree_raises(iv):
    """The mirror of guard_writes, and the reason bare reads accumulate without it.

    A path opened without going through iv.reads() is absent from the graph and from the
    recorded inputs, so whatever it depends on can change and the artifact never rebuilds.
    Nothing about that is visible at runtime — the read succeeds and the number is wrong.
    """
    seed(iv, "raw/feed/")
    bare = next(p for p in (iv.root / "raw/feed").iterdir() if p.suffix == ".parquet")

    with pytest.raises(DeclError, match="not handed back by iv.reads"):
        pl.read_parquet(bare)

    # Declared, so the same file is fine — and outside the tree is nobody's business.
    assert pl.read_parquet(iv.reads("raw/feed/", why="declared")).height == 3


def test_updating_a_dataset_this_stage_does_not_write_is_refused(iv):
    """The flag hides a dataset from the comparison, so on someone else's it hides a real
    dependency and this stage never rebuilds when that input moves."""
    seed(iv, "raw/rosters/")

    @iv.step(why="reads a dataset a later stage writes")
    def build():
        iv.reads("raw/rosters/", why="the previous run's copy", update_file_on_disk=True)
        with iv.writes("processed/cohorts/", why="a fit per cohort") as out:
            pl.DataFrame({"a": [1]}).write_parquet(out)

    with pytest.raises(DeclError, match="but this stage writes processed/cohorts/"):
        build()
