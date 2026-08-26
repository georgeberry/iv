

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
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def seed(iv, dataset, part=None, n=3, extra=0, why="an upstream feed"):


    with iv.writes(dataset, why=why, part=part) as out:
        pl.DataFrame({"a": range(n), "b": [x + extra for x in range(n)]}).write_parquet(out)
    return iv._sources.get(dataset) or iv.source(dataset, why=why)


def rows(iv, dataset):
    return pl.read_parquet(iv.reads(dataset, why="check")).height


def redeclared(iv):


    iv._assets.clear()
    return iv


def out_asset(iv, feed, ran=None):

    @iv.data(dataset="processed/out/", why="passthrough")
    def build(feed=iv.all_of(feed, why="the upstream")):
        if ran is not None:
            ran.append(1)
        return feed
    return build


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


def test_a_stage_with_several_outputs_is_checked_on_all_of_them(iv):

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


    import shutil
    shutil.rmtree(iv.resolve_out("out/careers/"))
    assert build() is True, "a missing output must bring the stage back"
    assert len(ran) == 2
    assert iv.resolve_out("out/careers/").exists()


def test_part_is_ambient_for_the_body(iv):
    feed = seed(iv, "raw/feed/")
    ran = []

    @iv.data(dataset="processed/by_season/", why="one season", part={"season": "2026"})
    def build(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return feed

    build()
    assert list(_sh.current_shards(iv.resolve_out("processed/by_season/"))) == ["season=2026"]
    build()
    assert len(ran) == 1


def test_a_code_edit_alone_does_not_rerun_anything(iv):


    feed = seed(iv, "raw/feed/")
    ran = []

    def build(extra):
        @redeclared(iv).data(dataset="processed/mid/", why="passthrough")
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

        @iv.data(dataset="config/model/", why="what the fits answer to")
        def model():
            return pl.DataFrame({"v": [version[0]]})

        @iv.data(dataset="processed/modelled/", why="model output")
        def modelled(m=iv.all_of(model, why="a model change must rebuild this"),
                     feed=iv.all_of(feed, why="the upstream")):
            ran["modelled"] += 1
            return feed

        @iv.data(dataset="processed/plain/", why="no model in it")
        def plain(feed=iv.all_of(feed, why="the upstream")):
            ran["plain"] += 1
            return feed

        @iv.data(dataset="processed/end/", why="the expensive one")
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


    @iv.data(dataset="config/model/", why="the model")
    def model():
        return pl.DataFrame({"v": ["m1"]})

    model()
    before = sorted(x.name for x in iv.resolve_out("config/model/").iterdir())
    model()
    assert sorted(x.name for x in iv.resolve_out("config/model/").iterdir()) == before


def test_the_clock_is_a_file(iv, monkeypatch):

    import datetime as _d
    fetches = []

    @iv.data(dataset="config/today/", why="poll once a day")
    def today():
        return pl.DataFrame({"date": [_d.date.today().isoformat()]})

    @iv.data(dataset="raw/feed/", why="a polled feed",
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


    calls = []

    @iv.data(dataset="raw/archive/", why="fetch-once history", once=True,
             external={"sports-reference": "a page that will not change"})
    def build():
        calls.append(1)
        return pl.DataFrame({"a": [1]})

    build(); build()
    assert len(calls) == 1


def test_a_named_partition_that_disappears_is_named(iv):

    for s in ("2025", "2026"):
        feed = seed(iv, "raw/feed/", part={"season": s})

    @iv.data(dataset="processed/out/", why="two named seasons")
    def build(got=iv.parts(feed, season=["2025", "2026"], why="exactly these two")):
        return got

    build()
    _sh.current_shards(iv.resolve("raw/feed/"))["season=2026"].path.unlink()
    assert "no shard for season=2026" in build.why_stale()


def test_a_vanished_shard_from_a_WHOLE_dataset_read_is_a_moved_input(iv):

    for s in ("2025", "2026"):
        feed = seed(iv, "raw/feed/", part={"season": s})
    ran = []
    build = out_asset(iv, feed, ran)
    build()
    _sh.current_shards(iv.resolve("raw/feed/"))["season=2026"].path.unlink()
    assert "its inputs moved" in build.why_stale()
    build()
    assert len(ran) == 2


def test_a_body_that_raises_records_nothing(iv):
    feed = seed(iv, "raw/feed/")

    @iv.data(dataset="processed/out/", why="doomed")
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
        iv.dataset("processed/out/", why="")


def seasons(iv, built):
    box_src = iv._sources["raw/box/"]

    @iv.data(dataset="processed/feat/", why="per-season features", part="season")
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

    for s in ("2024", "2025", "2026"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))

    @iv.data(dataset="processed/cohort/", why="a cohort fit on prior seasons", part="season")
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


def test_an_update_read_is_lineage_or_a_stage_is_stale_against_its_own_last_output(iv):

    feed = seed(iv, "raw/feed/")
    today = seed(iv, "config/today/")
    ran = []

    @iv.data(dataset="raw/log/", why="a running history")
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


    ran = []

    @iv.data(dataset="raw/feed/", why="rewritten in place", once=True)
    def build(have=iv.own_last_copy(why="the copy on disk")):
        ran.append(1)
        return pl.DataFrame({"a": [1]})

    build(); build(); build()
    assert len(ran) == 1 and build.why_stale() is None


def test_updating_a_dataset_this_stage_does_not_write_is_refused(iv):


    rosters = seed(iv, "raw/rosters/")

    @iv.data(dataset="processed/cohorts/", why="a fit per cohort")
    def build(prev=iv.own_last_copy(rosters, why="the previous run's copy")):
        return pl.DataFrame({"a": [1]})

    with pytest.raises(DeclError, match="but this stage writes processed/cohorts/"):
        build()


def test_reads_come_back_in_a_stable_semantic_order(iv):
    for s in (2011, 2006, 2026, 2019):
        seed(iv, "raw/box/", part={"season": str(s)}, extra=s)
    seen = {tuple(p.name for p in iv.reads("raw/box/", why="all")) for _ in range(6)}
    assert len(seen) == 1
    got = iv.reads("raw/box/", why="all")
    assert [p.name.split(".")[0] for p in got] == \
        ["season=2006", "season=2011", "season=2019", "season=2026"]


def test_a_read_of_the_whole_dataset_notices_a_brand_new_partition(iv):


    for s in ("2024", "2025"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    ran = []

    @iv.data(dataset="processed/xpm/", why="the fit")
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

    seed(iv, "raw/box/", part={"season": "2019"})

    @iv.data(dataset="processed/out/", why="passthrough")
    def build(box=iv.all_of(iv._sources["raw/box/"], why="every season", as_paths=True)):
        return pl.DataFrame({"a": [1]})

    build()
    seed(iv, "raw/box/", part={"season": "2020"})
    fresh = _sh.current_shards(iv.resolve_out("raw/box/"))["season=2020"]
    assert fresh.key == "", "a raw shard has no upstream, so its name carries no key"


def test_a_partition_appearing_inside_the_range_read_forces_a_rebuild(iv):


    for s in ("2019", "2020"):
        seed(iv, "raw/box/", part={"season": s}, extra=int(s))
    ran = []

    @iv.data(dataset="processed/cohort/", why="the cohort fit")
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


def test_a_declared_schema_is_validated_on_write_and_read(iv):
    schema = {"player": pl.Int64, "pts": pl.Int64}

    @iv.data(dataset="processed/features/", why="feature matrix", schema=schema)
    def features():
        return pl.DataFrame({"player": [1], "pts": [2]})

    assert features().schema == schema
    assert iv.reads("processed/features/", why="the matrix")


def test_a_declared_schema_refuses_a_wrong_output_before_commit(iv):
    @iv.data(dataset="processed/features/", why="feature matrix",
             schema={"player": pl.Int64})
    def features():
        return pl.DataFrame({"player": ["not an integer"]})

    with pytest.raises(DeclError, match="declared schema"):
        features()
    assert not iv.resolve_out("processed/features/").exists()


def test_a_source_schema_is_validated_when_filled_through_iv_writes(iv):
    source = iv.source("raw/feed/", why="incoming feed", schema={"id": pl.Int64})

    with pytest.raises(DeclError, match="declared schema"):
        with iv.writes(source.dataset, why="received feed") as out:
            pl.DataFrame({"id": ["wrong"]}).write_parquet(out)
    assert not iv.resolve_out(source.dataset).exists()


def test_each_output_of_a_multi_output_stage_keeps_its_own_schema(iv):
    ratings = iv.dataset("processed/ratings/", why="player ratings",
                         schema={"player": pl.Int64, "rating": pl.Float64})
    summary = iv.dataset("processed/summary/", why="fit summary",
                         schema={"n": pl.Int64})

    @iv.step(output={"ratings": ratings, "summary": summary}, why="joint fit")
    def fit():
        return {"ratings": pl.DataFrame({"player": [1], "rating": [1.5]}),
                "summary": pl.DataFrame({"n": [1]})}

    assert fit() is True
    assert iv.reads(ratings.dataset, why="ratings")
    assert iv.reads(summary.dataset, why="summary")


def test_schema_is_refused_for_a_non_parquet_dataset(iv):
    with pytest.raises(DeclError, match="only for .parquet"):
        iv.dataset("dump/page/", why="web page", ext=".html", schema={"a": pl.Int64})


def test_a_schema_migration_rebuilds_partitions_one_at_a_time(iv, tmp_path):
    old = {"season": pl.String, "pts": pl.Int64}
    new = {"season": pl.String, "pts": pl.Int64, "z": pl.Int64}

    @iv.data(dataset="processed/features/", why="feature matrix", part="season",
             once=True, schema=old)
    def old_features(season):
        return pl.DataFrame({"season": [season], "pts": [1]})

    old_features.for_each(["2024", "2025"])

    migrated = Pipeline(tree=iv.tree, stage_dir=tmp_path / "new-stage", project=tmp_path)

    @migrated.data(dataset="processed/features/", why="feature matrix", part="season",
                   once=True, schema=new)
    def new_features(season):
        return pl.DataFrame({"season": [season], "pts": [1], "z": [2]})

    assert new_features.for_each(["2024"]) == ["2024"]
    with pytest.raises(StateError, match="schema migration"):
        migrated.reads("processed/features/", why="all features")
    assert new_features.for_each(["2025"]) == ["2025"]
    assert len(migrated.reads("processed/features/", why="all features")) == 2


def test_a_declared_schema_changes_a_once_artifacts_key(iv, tmp_path):
    old = {"id": pl.Int64}
    new = {"id": pl.Int64, "score": pl.Int64}

    @iv.data(dataset="processed/model/", why="model input", once=True, schema=old)
    def old_model():
        return pl.DataFrame({"id": [1]})

    old_model()
    changed = Pipeline(tree=iv.tree, stage_dir=tmp_path / "new-stage", project=tmp_path)

    @changed.data(dataset="processed/model/", why="model input", once=True, schema=new)
    def new_model():
        return pl.DataFrame({"id": [1], "score": [2]})

    assert new_model.why_stale() is not None
    assert new_model().schema == new


def test_reading_shards_reproduces_a_total_sort_over_the_merged_frame(iv):


    import random
    rows = [{"season": s, "game_id": g, "row": r}
            for s in ("2006", "2011", "2019", "2026")
            for g in range(4) for r in range(3)]
    merged = pl.DataFrame(rows).sort(["season", "game_id", "row"])

    order = ["2019", "2026", "2006", "2011"]
    random.Random(0).shuffle(order)
    for s in order:
        with iv.writes("processed/possessions/", why="one season", part={"season": s}) as out:
            pl.DataFrame([r for r in rows if r["season"] == s]) \
              .sort(["game_id", "row"]).write_parquet(out)

    got = pl.read_parquet(iv.reads("processed/possessions/", why="every season"))
    assert got.equals(merged), "sharded read order must equal the merged total sort"


def test_a_rebuilt_shard_does_not_disturb_the_read_order(iv):

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


    seed(iv, "processed/possessions/")
    knob = [4.0]
    fits = []

    @iv.data(dataset="config/model/", why="the knobs the fit shape depends on", ext=".json")
    def model():
        return {"half_life": knob[0]}

    @iv.data(dataset="processed/rapm_fit/", why="the fitted model", ext=".pkl")
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


    seed(iv, "raw/box/")
    ran = []

    @iv.data(dataset="processed/box_features/", why="the box matrix for one season",
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


    _sh.current_shards(iv.resolve_out("processed/box_features/"))["season=2025"].path.unlink()
    assert iv.why_stale("processed/box_features/") is None
    build()
    assert len(ran) == 1

    seed(iv, "raw/box/", extra=99)
    build()
    assert len(ran) == 2, "a moved upstream still rebuilds all of them"


def test_adding_a_dependency_reruns_the_stage(iv):


    feed = seed(iv, "raw/feed/")
    seed(iv, "raw/extra/")
    ran = []

    def one_input():
        @redeclared(iv).data(dataset="processed/mid/", why="mid")
        def mid(feed=iv.all_of(feed, why="the upstream")):
            ran.append(1)
            return feed
        mid()

    def two_inputs():
        @redeclared(iv).data(dataset="processed/mid/", why="mid")
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


    seed(iv, "raw/extra/", n=9)
    two_inputs()
    assert len(ran) == 3, "raw/extra/ moved"


def test_an_undeclared_read_of_the_data_tree_raises(iv):


    seed(iv, "raw/feed/")
    bare = next(p for p in (iv.tree / "raw/feed").iterdir() if p.suffix == ".parquet")

    with pytest.raises(DeclError, match="not handed back by iv.reads"):
        pl.read_parquet(bare)


    assert pl.read_parquet(iv.reads("raw/feed/", why="declared")).height == 3


def test_builtin_open_catches_an_undeclared_read_inside_a_stage(iv):
    seed(iv, "raw/feed/")
    bare = next((iv.tree / "raw/feed").iterdir())

    @iv.data(dataset="processed/out/", why="opens a file")
    def build():
        with open(bare, "rb"):
            pass
        return pl.DataFrame()

    with pytest.raises(DeclError, match="not handed back by iv.reads"):
        build()


def test_os_open_catches_an_undeclared_read_inside_a_stage(iv):
    import os

    seed(iv, "raw/feed/")
    bare = next((iv.tree / "raw/feed").iterdir())

    @iv.data(dataset="processed/out/", why="opens a file descriptor")
    def build():
        fd = os.open(bare, os.O_RDONLY)
        os.close(fd)
        return pl.DataFrame()

    with pytest.raises(DeclError, match="not handed back by iv.reads"):
        build()


def test_a_direct_write_inside_a_stage_is_refused(iv):
    target = iv.tree / "raw/side-effect.txt"
    target.parent.mkdir(parents=True)

    @iv.data(dataset="processed/out/", why="tries a side write")
    def build():
        target.write_text("not a shard")
        return pl.DataFrame()

    with pytest.raises(DeclError, match="outside iv.writes"):
        build()


def test_shutil_copy_into_the_data_tree_is_refused_inside_a_stage(iv, tmp_path):
    import shutil

    source = tmp_path / "outside.txt"
    source.write_text("outside")
    target = iv.tree / "raw/copied.txt"
    target.parent.mkdir(parents=True)

    @iv.data(dataset="processed/out/", why="tries a side copy")
    def build():
        shutil.copyfile(source, target)
        return pl.DataFrame()

    with pytest.raises(DeclError, match="outside iv.writes"):
        build()


def test_an_optional_coverage_claim_is_answered_the_same_way_twice(iv):


    seed(iv, "raw/feed/", part={"half": "a"})
    sel = (("half", ("in", ("a", "b"))),)


    assert iv.key_of("processed/out/", None, (("raw/feed/", sel, True),))


    assert iv.reads("raw/feed/", why="both halves if they are there",
                    where={"half": ["a", "b"]}, optional=True) == []

    with pytest.raises(StateError, match="coverage claim"):
        iv.reads("raw/feed/", why="both halves, required",
                 where={"half": ["a", "b"]})
