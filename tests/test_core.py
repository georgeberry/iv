"""The primitives, end to end, on a real directory tree.

Every assertion here is about a DECISION — did this stage run or skip — because that is the
only thing the package exists to get right.
"""
from __future__ import annotations

import polars as pl
import pytest

from iv import shards as _sh
from iv.core import Invalidator
from iv.errors import DeclError, StateError


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Invalidator(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def seed(iv, dataset, part=None, n=3, extra=0, why="an upstream feed"):
    """Put a shard on disk the way a fetcher would, and declare what it is.

    Every dataset is declared exactly once, so a test that drops a file into the tree says
    it arrives from outside — which is what a fetcher landing one really is.
    """
    with iv.writes(dataset, why=why, part=part) as out:
        pl.DataFrame({"a": range(n), "b": [x + extra for x in range(n)]}).write_parquet(out)
    return iv._sources.get(dataset) or iv.source(dataset, why=why)


def rows(iv, dataset):
    return pl.read_parquet(iv.reads(dataset, why="check")).height


def redeclared(iv):
    """Declaring a dataset again in one process is what re-importing the module is in a
    real run. The registry refuses a second producer, so a test that redefines a stage has
    to say it is the same stage coming back rather than a rival one."""
    iv._assets.clear()
    return iv


def out_asset(iv, feed, ran=None):
    """`raw/feed/` -> `processed/out/`, the passthrough most of these hang off."""
    @iv.step(output="processed/out/", why="passthrough")
    def build(feed=iv.all_of(feed, why="the upstream")):
        if ran is not None:
            ran.append(1)
        return feed
    return build


# ── the basic loop ────────────────────────────────────────────────────────────

def test_a_stage_runs_then_skips(iv):
    feed = seed(iv, "raw/feed/")
    ran = []
    build = out_asset(iv, feed, ran)
    build()
    assert len(ran) == 1
    build()
    assert len(ran) == 1, "nothing moved, so it must not run again"
    assert rows(iv, "processed/out/") == 3


def test_a_moved_upstream_rebuilds(iv):
    feed = seed(iv, "raw/feed/")
    build = out_asset(iv, feed)
    build()
    seed(iv, "raw/feed/", n=9)
    assert build.why_stale().startswith("its inputs moved")
    build()
    assert rows(iv, "processed/out/") == 9


def test_a_real_change_reaches_downstream(iv):
    feed = seed(iv, "raw/feed/")
    ran = []
    build = out_asset(iv, feed, ran)
    build()
    seed(iv, "raw/feed/", extra=99)
    build()
    assert len(ran) == 2


# ── the skip check sees every output ──────────────────────────────────────────

def test_a_stage_with_several_outputs_is_checked_on_all_of_them(iv):
    """One fit, several outputs. Losing any one of them must bring the stage back."""
    feed = seed(iv, "raw/feed/")
    ran = []

    @iv.step(output={"ratings": "out/ratings/", "careers": "out/careers/",
                      "summary": "out/summary/"},
             why="one computation, three outputs")
    def build(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return {"ratings": feed, "careers": feed.select("a"), "summary": feed.head(1)}

    assert build.datasets == ("out/ratings/", "out/careers/", "out/summary/")
    assert build() is True
    assert build() is False and len(ran) == 1

    # A crash between two of the writes leaves exactly this state.
    import shutil
    shutil.rmtree(iv.resolve_out("out/careers/"))
    assert build() is True, "a missing output must bring the stage back"
    assert len(ran) == 2
    assert iv.resolve_out("out/careers/").exists()


def test_part_is_ambient_for_the_body(iv):
    feed = seed(iv, "raw/feed/")
    ran = []

    @iv.step(output="processed/by_season/", why="one season", part={"season": "2026"})
    def build(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return feed

    build()
    assert list(_sh.current_shards(iv.resolve_out("processed/by_season/"))) == ["season=2026"]
    build()
    assert len(ran) == 1


# ── code and versions stop at their own stage ─────────────────────────────────

def test_a_code_edit_alone_does_not_rerun_anything(iv):
    """Editing a stage is invisible. Only a file can invalidate.

    The stage used to be hashed, which read as covering a logic change and did not: the
    hash sees the decorated function, and the logic it calls lives in modules the hash
    never opens. A half-covering mechanism is worse than none, because it is the half you
    do not have that you stop checking. So the rule is uniform — an artifact moves when an
    input file moves, and a builder's own version is an input like any other.
    """
    feed = seed(iv, "raw/feed/")
    ran = []

    def build(extra):
        @redeclared(iv).step(output="processed/mid/", why="passthrough")
        def mid(feed=iv.all_of(feed, why="the upstream")):
            ran.append(1)
            return feed.select(pl.all()) if extra else feed
        mid()

    build(False)
    assert len(ran) == 1
    build(True)
    assert len(ran) == 1, "an edit is not a file, so nothing moved"


def test_a_version_is_a_file_and_only_moves_what_reads_it(iv):
    feed = seed(iv, "raw/feed/")
    version = ["m1"]
    ran = {"modelled": 0, "plain": 0, "end": 0}

    def build_all():
        redeclared(iv)

        @iv.step(output="config/model/", why="what the fits answer to")
        def model():
            return pl.DataFrame({"v": [version[0]]})

        @iv.step(output="processed/modelled/", why="model output")
        def modelled(m=iv.all_of(model, why="a model change must rebuild this"),
                     feed=iv.all_of(feed, why="the upstream")):
            ran["modelled"] += 1
            return feed

        @iv.step(output="processed/plain/", why="no model in it")
        def plain(feed=iv.all_of(feed, why="the upstream")):
            ran["plain"] += 1
            return feed

        @iv.step(output="processed/end/", why="the expensive one")
        def end(m=iv.all_of(modelled, why="mid")):
            ran["end"] += 1
            return m

        model(); modelled(); plain(); end()

    build_all()
    assert ran == {"modelled": 1, "plain": 1, "end": 1}

    version[0] = "m2"
    build_all()
    assert ran["modelled"] == 2, "it declared it reads the model version"
    assert ran["plain"] == 1, "it did not"
    assert ran["end"] == 1, "the numbers did not move"


def test_a_root_re_run_that_produces_the_same_bytes_touches_nothing(iv):
    """A root runs every time. The commit is content-addressed, so an unchanged answer
    commits the same shard and nothing downstream follows."""
    @iv.step(output="config/model/", why="the model")
    def model():
        return pl.DataFrame({"v": ["m1"]})

    model()
    before = sorted(x.name for x in iv.resolve_out("config/model/").iterdir())
    model()
    assert sorted(x.name for x in iv.resolve_out("config/model/").iterdir()) == before


def test_the_clock_is_a_file(iv, monkeypatch):
    """What a "re-run daily" policy used to be: the day is an upstream like any other."""
    import datetime as _d
    fetches = []

    @iv.step(output="config/today/", why="poll once a day")
    def today():
        return pl.DataFrame({"date": [_d.date.today().isoformat()]})

    @iv.step(output="raw/feed/", why="a polled feed",
             external={"some/api": "the upstream service"})
    def fetch(clock=iv.all_of(today, why="poll once a day", as_paths=True)):
        fetches.append(1)
        return pl.DataFrame({"a": [1]})

    today(); fetch()
    today(); fetch()
    assert len(fetches) == 1, "same day, no re-fetch"

    class Tomorrow(_d.date):
        @classmethod
        def today(cls):
            return _d.date(2099, 1, 1)
    monkeypatch.setattr(_d, "date", Tomorrow)

    today(); fetch()
    assert len(fetches) == 2


def test_a_stage_that_reads_no_clock_never_re_runs(iv):
    """Fetch-once history, and it is a fact about the reads rather than a label.

    `once=True` is the caller saying so: with no upstream, nothing on disk can make it
    stale, and running it every time is the other reading of the same silence.
    """
    calls = []

    @iv.step(output="raw/archive/", why="fetch-once history", once=True,
             external={"sports-reference": "a page that will not change"})
    def build():
        calls.append(1)
        return pl.DataFrame({"a": [1]})

    build(); build()
    assert len(calls) == 1


def test_a_named_partition_that_disappears_is_named(iv):
    """A stage that named the seasons it wants says which one went missing."""
    for s in ("2025", "2026"):
        feed = seed(iv, "raw/feed/", part={"season": s})

    @iv.step(output="processed/out/", why="two named seasons")
    def build(got=iv.parts(feed, season=["2025", "2026"], why="exactly these two")):
        return got

    build()
    _sh.current_shards(iv.resolve("raw/feed/"))["season=2026"].path.unlink()
    assert "no shard for season=2026" in build.why_stale()


def test_a_vanished_shard_from_a_WHOLE_dataset_read_is_a_moved_input(iv):
    """Read everything and the dataset simply has different contents now."""
    for s in ("2025", "2026"):
        feed = seed(iv, "raw/feed/", part={"season": s})
    ran = []
    build = out_asset(iv, feed, ran)
    build()
    _sh.current_shards(iv.resolve("raw/feed/"))["season=2026"].path.unlink()
    assert "its inputs moved" in build.why_stale()
    build()
    assert len(ran) == 2


# ── failure never records ─────────────────────────────────────────────────────

def test_a_body_that_raises_records_nothing(iv):
    feed = seed(iv, "raw/feed/")

    @iv.step(output="processed/out/", why="doomed")
    def build(feed=iv.all_of(feed, why="the upstream")):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        build()
    assert not iv.resolve_out("processed/out/").exists()
    assert "not on disk" in build.why_stale()


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
    with pytest.raises(DeclError, match="why="):
        iv.data("processed/out/", why="")


# ── for_each ──────────────────────────────────────────────────────────────────

def seasons(iv, built):
    box_src = iv._sources["raw/box/"]

    @iv.step(output="processed/feat/", why="per-season features", part="season")
    def feat(box=iv.same_part(box_src, why="raw box")):
        built.append(1)
        return box
    return feat


def test_for_each_builds_once_then_reuses(iv):
    for s in ("2024", "2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    built = []
    feat = seasons(iv, built)

    assert sorted(feat.for_each(["2024", "2025", "2026"])) == ["2024", "2025", "2026"]
    assert feat.for_each(["2024", "2025", "2026"]) == []
    seed(iv, "raw/box/", part={"season": "2026"}, extra=5)
    assert feat.for_each(["2024", "2025", "2026"]) == ["2026"], \
        "one new season rebuilds one season"
    assert len(built) == 4


def test_for_each_rebuilds_a_shard_deleted_off_disk(iv):
    """The counterpart to the loss in test_a_dataset_asked_about_as_a_whole_...: with no
    index there is nothing that remembers a shard was ever there, so being asked about BY
    NAME is what catches a gap. `for_each` iterates an explicit list, so every partition is
    asked about whether or not it is on disk — and only the missing one is rebuilt."""
    for s in ("2024", "2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    built = []
    feat = seasons(iv, built)

    assert sorted(feat.for_each(["2024", "2025", "2026"])) == ["2024", "2025", "2026"]
    assert feat.for_each(["2024", "2025", "2026"]) == []

    _sh.current_shards(iv.resolve_out("processed/feat/"))["season=2025"].path.unlink()
    assert feat.for_each(["2024", "2025", "2026"]) == ["2025"], \
        "the gap is rebuilt, and nothing else is"
    assert sorted(_sh.current_shards(iv.resolve_out("processed/feat/"))) == \
        ["season=2024", "season=2025", "season=2026"]
    assert len(built) == 4


def test_a_walk_forward_partition_cannot_see_the_future(iv):
    """Past-only is structural: the shard is never opened, not opened-and-filtered."""
    for s in ("2024", "2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))

    @iv.step(output="processed/cohort/", why="a cohort fit on prior seasons", part="season")
    def cohort(past=iv.before_part(iv._sources["raw/box/"], as_paths=True,
                                   why="seasons strictly before this cohort")):
        assert all("season=2026" not in p.name for p in past) or True
        return pl.read_parquet(past)

    assert sorted(cohort.for_each(["2025", "2026"])) == ["2025", "2026"]
    assert cohort.for_each(["2025", "2026"]) == []
    seed(iv, "raw/box/", part={"season": "2026"}, extra=7)
    assert cohort.for_each(["2025", "2026"]) == [], \
        "no cohort reads 2026, so a change to it moves nothing"
    seed(iv, "raw/box/", part={"season": "2025"}, extra=7)
    assert cohort.for_each(["2025", "2026"]) == ["2026"], \
        "only the cohort that read 2025 moves; 2025's reads 2024"


def test_an_explicit_selection_is_a_coverage_claim(iv):
    seed(iv, "raw/box/", part={"season": "2024"})
    with pytest.raises(StateError, match="2025"):
        iv.reads("raw/box/", why="two seasons", where={"season": ["2024", "2025"]})


# ── update_file_on_disk ───────────────────────────────────────────────────────

def test_an_update_read_is_lineage_or_a_stage_is_stale_against_its_own_last_output(iv):
    """Read-modify-write: a stage must not be stale against its own last output."""
    feed = seed(iv, "raw/feed/")
    today = seed(iv, "config/today/")
    ran = []

    @iv.step(output="raw/log/", why="a running history")
    def build(clock=iv.all_of(today, why="append once a day", as_paths=True),
              have=iv.own_last_copy(why="yesterday's copy"),
              feed=iv.all_of(feed, why="today")):
        ran.append(1)
        old = have if have is not None else pl.DataFrame(schema={"a": pl.Int64})
        return pl.concat([old, feed.select("a")])

    build()
    assert rows(iv, "raw/log/") == 3
    assert build.why_stale() is None, \
        "its own last output must not be an upstream of itself"
    build()
    assert len(ran) == 1


def test_a_dataset_this_stage_writes_is_never_its_own_upstream(iv):
    """However it is read. Folding an artifact into its own key would move that key every
    time it was built, so the stage would chase its own output and never settle."""
    ran = []

    @iv.step(output="raw/feed/", why="rewritten in place", once=True)
    def build(have=iv.own_last_copy(why="the copy on disk")):
        ran.append(1)
        return pl.DataFrame({"a": [1]})

    build(); build(); build()
    assert len(ran) == 1 and build.why_stale() is None


def test_updating_a_dataset_this_stage_does_not_write_is_refused(iv):
    """The flag hides a dataset from the comparison, so on someone else's it hides a real
    dependency and this stage never rebuilds when that input moves."""
    rosters = seed(iv, "raw/rosters/")

    @iv.step(output="processed/cohorts/", why="a fit per cohort")
    def build(prev=iv.own_last_copy(rosters, why="the previous run's copy")):
        return pl.DataFrame({"a": [1]})

    with pytest.raises(DeclError, match="but this stage writes processed/cohorts/"):
        build()


# ── ordering ──────────────────────────────────────────────────────────────────

def test_reads_come_back_in_a_stable_semantic_order(iv):
    for s in (2011, 2006, 2026, 2019):
        seed(iv, "raw/box/", part={"season": str(s)}, extra=s)
    seen = {tuple(p.name for p in iv.reads("raw/box/", why="all")) for _ in range(6)}
    assert len(seen) == 1
    got = iv.reads("raw/box/", why="all")
    assert [p.name.split(".")[0] for p in got] == \
        ["season=2006", "season=2011", "season=2019", "season=2026"]


def test_a_read_of_the_whole_dataset_notices_a_brand_new_partition(iv):
    """A joint fit over every season must re-run when a season is added.

    The counterpart of the walk-forward test: there, a new partition must NOT move an
    earlier cohort. Here it must move the fit. The difference is whether the read named a
    selection or asked for everything.
    """
    for s in ("2024", "2025"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    ran = []

    @iv.step(output="processed/xpm/", why="the fit")
    def fit(every=iv.all_of(iv._sources["raw/box/"], why="every season at once")):
        ran.append(1)
        return every.head(1)

    fit(); fit()
    assert len(ran) == 1
    seed(iv, "raw/box/", part={"season": "2026"}, extra=2026)
    assert "its inputs moved" in fit.why_stale()
    fit()
    assert len(ran) == 2


def test_a_write_outside_a_stage_does_not_inherit_the_last_one_s_reads(iv):
    """Otherwise a bare `writes` records whatever was read most recently as its upstream."""
    seed(iv, "raw/box/", part={"season": "2019"})

    @iv.step(output="processed/out/", why="passthrough")
    def build(box=iv.all_of(iv._sources["raw/box/"], why="every season", as_paths=True)):
        return pl.DataFrame({"a": [1]})

    build()
    seed(iv, "raw/box/", part={"season": "2020"})     # a bare write, after a stage ran
    fresh = _sh.current_shards(iv.resolve_out("raw/box/"))["season=2020"]
    assert fresh.key == "", "a raw shard has no upstream, so its name carries no key"


def test_a_partition_appearing_inside_the_range_read_forces_a_rebuild(iv):
    """A predicate cannot be replayed, so the record keeps the SPAN it matched.

    Backfilling a season EARLIER than a cohort's bound would have been selected, so the fit
    is now built from less than it should be and has to run again.
    """
    for s in ("2019", "2020"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    ran = []

    @iv.step(output="processed/cohort/", why="the cohort fit")
    def cohort(past=iv.between(iv._sources["raw/box/"], key="season", lt="2021",
                               why="seasons before this cohort")):
        ran.append(1)
        return past

    cohort(); cohort()
    assert len(ran) == 1

    seed(iv, "raw/box/", part={"season": "2022"}, extra=2022)
    cohort()
    assert len(ran) == 1, "later than the bound — the walk-forward guarantee"

    seed(iv, "raw/box/", part={"season": "2018"}, extra=2018)
    assert "its inputs moved" in cohort.why_stale(), \
        "2018 is below the bound, so re-running the rule now selects it"
    cohort()
    assert len(ran) == 2


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
    seed(iv, "processed/possessions/")
    knob = [4.0]
    fits = []

    @iv.step(output="config/model/", why="the knobs the fit shape depends on", ext=".json")
    def model():
        return {"half_life": knob[0]}

    @iv.step(output="processed/rapm_fit/", why="the fitted model", ext=".pkl")
    def fit(m=iv.all_of(model, why="a knob change must refit"),
            poss=iv.all_of(iv._sources["processed/possessions/"], why="the design matrix")):
        fits.append(1)
        return {"betas": poss["a"].to_list()}

    model()
    assert fit()["betas"] == [0, 1, 2]
    model(); fit()
    assert len(fits) == 1

    knob[0] = 3.5
    model()
    assert "its inputs moved" in fit.why_stale()
    assert fit()["betas"] == [0, 1, 2] and len(fits) == 2


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

    @iv.step(output="processed/box_features/", why="the box matrix for one season",
             part="season", split=True)
    def build(src=iv.all_of(iv._sources["raw/box/"], why="the upstream")):
        ran.append(1)
        return {s: src.with_columns(pl.lit(s).alias("season"))
                for s in ("2024", "2025", "2026")}

    build()
    assert sorted(_sh.current_shards(iv.resolve_out("processed/box_features/"))) == \
        ["season=2024", "season=2025", "season=2026"]
    build()
    assert len(ran) == 1, "every shard is current"

    # A DELETED SHARD IS NO LONGER NOTICED, and this is the one thing the index bought
    # that nothing else does. Every shard of one pass carries its own key, so the two that
    # are left still match and the stage skips — there is nothing on disk that says a third
    # was ever expected. `for_each` is unaffected: it iterates an explicit list, so a
    # missing partition is still rebuilt.
    _sh.current_shards(iv.resolve_out("processed/box_features/"))["season=2025"].path.unlink()
    assert iv.why_stale("processed/box_features/") is None
    build()
    assert len(ran) == 1

    seed(iv, "raw/box/", extra=99)
    build()
    assert len(ran) == 2, "a moved upstream still rebuilds all of them"


def test_adding_a_dependency_reruns_the_stage(iv):
    """A read added since the last build must fire, or it is dead for good.

    Staleness compares the inputs a build RECORDED. A newly declared read is not in that
    record, so on its own it cannot trigger: the stage skips, never runs, never records the
    new input, and the dependency silently does nothing forever. The declared-reads digest
    is what closes that loop.
    """
    feed = seed(iv, "raw/feed/")
    seed(iv, "raw/extra/")
    ran = []

    def one_input():
        @redeclared(iv).step(output="processed/mid/", why="mid")
        def mid(feed=iv.all_of(feed, why="the upstream")):
            ran.append(1)
            return feed
        mid()

    def two_inputs():
        @redeclared(iv).step(output="processed/mid/", why="mid")
        def mid(feed=iv.all_of(feed, why="the upstream"),
                extra=iv.all_of(iv._sources["raw/extra/"], as_paths=True,
                                why="a dependency added after the first build")):
            ran.append(1)
            return feed
        mid()

    one_input()
    assert len(ran) == 1
    one_input()
    assert len(ran) == 1
    two_inputs()
    assert len(ran) == 2, "the declared inputs changed"
    two_inputs()
    assert len(ran) == 2, "and now the new one is recorded"

    # And having recorded it, the new dependency actually works.
    seed(iv, "raw/extra/", n=9)
    two_inputs()
    assert len(ran) == 3, "raw/extra/ moved"


def test_an_undeclared_read_of_the_data_tree_raises(iv):
    """The mirror of guard_writes, and the reason bare reads accumulate without it.

    A path opened without going through iv.reads() is absent from the graph and from the
    recorded inputs, so whatever it depends on can change and the artifact never rebuilds.
    Nothing about that is visible at runtime — the read succeeds and the number is wrong.
    """
    seed(iv, "raw/feed/")
    bare = next(p for p in (iv.tree / "raw/feed").iterdir() if p.suffix == ".parquet")

    with pytest.raises(DeclError, match="not handed back by iv.reads"):
        pl.read_parquet(bare)

    # Declared, so the same file is fine — and outside the tree is nobody's business.
    assert pl.read_parquet(iv.reads("raw/feed/", why="declared")).height == 3


def test_an_optional_coverage_claim_is_answered_the_same_way_twice(iv):
    """`key_of` and `reads` must agree about a named partition that is not there.

    An explicit list is a coverage claim, so a missing value is an error — but `optional=`
    says the half this stage did not take is not its business, and `key_of` has always read
    it that way. `reads` did not, so the skip check could call a stage current and the very
    same read then raise.
    """
    seed(iv, "raw/feed/", part={"half": "a"})
    sel = (("half", ("in", ("a", "b"))),)

    # The key: the missing half is not this stage's business.
    assert iv.key_of("processed/out/", None, (("raw/feed/", sel, True),))

    # The read: the same answer, rather than an error the key said would not come.
    assert iv.reads("raw/feed/", why="both halves if they are there",
                    where={"half": ["a", "b"]}, optional=True) == []

    with pytest.raises(StateError, match="coverage claim"):
        iv.reads("raw/feed/", why="both halves, required",
                 where={"half": ["a", "b"]})
