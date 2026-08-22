"""A declared dataset, end to end.

Every assertion is about a DECISION — did this body run, or did it not — because that is
the only thing the package exists to get right. The cases that matter most are the ones
where something CHANGED and a stage had to notice.
"""
from __future__ import annotations

import polars as pl
import pytest

from iv import Invalidator
from iv import graph as _graph
from iv import shards as _sh
from iv.errors import DeclError, StateError


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Invalidator(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def frame(n=2, extra=0):
    return pl.DataFrame({"player": list(range(n)), "pts": [x + extra for x in range(n)]})


# ── the basic loop ────────────────────────────────────────────────────────────

def test_a_derived_asset_builds_then_skips(iv):
    ran = []

    @iv.data("raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.data("processed/out/", why="passthrough")
    def out(feed=iv.all_of("raw/feed/", why="the upstream")):
        ran.append(1)
        return feed

    feed()
    assert out().height == 2 and len(ran) == 1
    assert out().height == 2 and len(ran) == 1, "nothing moved, so it must not run again"


def test_a_moved_upstream_rebuilds(iv):
    n = [2]
    ran = []

    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame(n[0])

    @iv.data("processed/out/", why="passthrough")
    def out(feed=iv.all_of("raw/feed/", why="the upstream")):
        ran.append(1)
        return feed

    feed(); out()
    assert len(ran) == 1
    n[0] = 9
    feed()
    assert out.why_stale().startswith("its inputs moved")
    assert out().height == 9 and len(ran) == 2


def test_a_rebuild_that_does_not_move_the_bytes_stops_there(iv):
    """The early cutoff: an upstream re-runs, produces the same shard, and the tail sits."""
    ran = []

    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    @iv.data("processed/out/", why="passthrough")
    def out(feed=iv.all_of("raw/feed/", why="the upstream")):
        ran.append(1)
        return feed

    feed(); out()
    feed()                                   # runs again — a root always does
    out()
    assert len(ran) == 1, "the bytes did not move, so the downstream must not follow"


# ── a root is what lets anything in ───────────────────────────────────────────

def test_a_root_runs_every_time_because_nothing_else_can_notice_the_world(iv):
    """No upstream means no question `why_stale` can ask, so skipping seals the tree shut."""
    ran = []
    n = [2]

    @iv.data("raw/feed/", why="fetched from outside")
    def feed():
        ran.append(1)
        return frame(n[0])

    feed()
    feed()
    assert len(ran) == 2, "a root that skips can never see new data"
    n[0] = 7
    assert feed().height == 7, "the change reached the tree"


def test_once_is_how_a_fetch_once_archive_opts_out(iv):
    ran = []

    @iv.data("raw/archive/", why="a one-time backfill", once=True)
    def archive():
        ran.append(1)
        return frame()

    archive(); archive()
    assert len(ran) == 1


def test_a_root_that_always_runs_is_not_warned_about_as_running_once(iv):
    @iv.data("raw/feed/", why="fetched from outside")
    def feed():
        return frame()

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    errors, warns = _graph.check(_graph.build(iv))
    assert not errors
    assert not [w for w in warns if "RUNS ONCE" in w], (
        "a root asset re-runs every time, so the warning would be false")


def test_once_is_warned_about(iv):
    @iv.data("raw/archive/", why="a one-time backfill", once=True)
    def archive():
        return frame()

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(a=iv.all_of("raw/archive/", why="the backfill")):
        return a

    _, warns = _graph.check(_graph.build(iv))
    assert [w for w in warns if "RUNS ONCE" in w]


# ── partitions ────────────────────────────────────────────────────────────────

def seasons_pipeline(iv, pts=None):
    pts = pts if pts is not None else {"2024": 10, "2025": 20, "2026": 30}
    ran = []

    @iv.data("raw/box/", why="raw box for one season", part="season")
    def box(season):
        ran.append(f"box:{season}")
        return pl.DataFrame({"player": [1, 2], "pts": [pts[season], pts[season] + 1]})

    @iv.data("processed/features/", why="per-season features", part="season")
    def features(box=iv.same_part("raw/box/", why="this season's box")):
        ran.append("features")
        return box.with_columns((pl.col("pts") * 2).alias("z"))

    @iv.data("processed/cohorts/", why="a fit on prior seasons only", part="season")
    def cohorts(past=iv.before_part("processed/features/", why="every prior season")):
        ran.append("cohorts")
        return past.select(pl.col("z").sum().alias("total"))

    return box, features, cohorts, ran, pts


def test_for_each_builds_once_then_reuses(iv):
    box, features, _, ran, _ = seasons_pipeline(iv)
    box.for_each(["2024", "2025"])
    assert features.for_each(["2024", "2025"]) == ["2024", "2025"]
    ran.clear()
    assert features.for_each(["2024", "2025"]) == []
    assert ran == []


def test_only_the_partition_that_moved_is_rebuilt(iv):
    box, features, _, ran, pts = seasons_pipeline(iv)
    box.for_each(["2024", "2025"])
    features.for_each(["2024", "2025"])
    pts["2024"] = 999
    box.for_each(["2024", "2025"])
    ran.clear()
    assert features.for_each(["2024", "2025"]) == ["2024"], \
        "2025's box did not move, so its features must be reused"


def test_a_cohort_cannot_see_the_future(iv):
    box, features, cohorts, ran, _ = seasons_pipeline(iv)
    box.for_each(["2024", "2025"])
    features.for_each(["2024", "2025"])
    cohorts.for_each(["2025"])

    ran.clear()
    box.for_each(["2026"]); features.for_each(["2026"])
    assert cohorts.for_each(["2025"]) == [], \
        "a season ABOVE the bound cannot change a cohort fit on prior seasons"


def test_a_season_backfilled_below_the_bound_is_picked_up(iv):
    box, features, cohorts, ran, _ = seasons_pipeline(iv)
    box.for_each(["2025"]); features.for_each(["2025"])
    cohorts.for_each(["2026"])

    box.for_each(["2024"]); features.for_each(["2024"])
    assert cohorts.for_each(["2026"]) == ["2026"], \
        "a season BELOW the bound is inside the fit and must bring it back"


def test_a_partitioned_call_names_its_partition(iv):
    box, _, _, _, _ = seasons_pipeline(iv)
    assert box("2024").height == 2
    assert box(season="2025").height == 2
    with pytest.raises(DeclError, match="a call names one"):
        box()


def test_an_unpartitioned_asset_takes_no_partition(iv):
    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    with pytest.raises(DeclError, match="takes no partition value"):
        feed("2024")
    with pytest.raises(DeclError, match="nothing to iterate"):
        feed.for_each(["2024"])


# ── round trip: a cached call returns what the body returned ─────────────────

def test_a_dict_round_trips_through_json(iv):
    @iv.data("config/knobs/", why="the knobs", ext=".json")
    def knobs():
        return {"half_life": 4.0, "seed": 0}

    first = knobs()
    assert first == {"half_life": 4.0, "seed": 0}
    assert knobs() == first and isinstance(knobs(), dict)


def test_a_dict_is_refused_by_parquet_rather_than_silently_reshaped(iv):
    """The trap this exists to close: dict in, DataFrame out, and the same function then
    gives two types depending on whether the shard happened to be current."""
    @iv.data("config/knobs/", why="the knobs")
    def knobs():
        return {"half_life": 4.0}

    with pytest.raises(DeclError, match="would give two different types"):
        knobs()


def test_a_frame_round_trips_through_parquet(iv):
    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    assert isinstance(feed(), pl.DataFrame)
    assert feed().equals(frame())


def test_an_arbitrary_object_round_trips_through_pickle(iv):
    @iv.data("config/thing/", why="something exotic", ext=".pkl")
    def thing():
        return {"a", "b"}

    assert thing() == {"a", "b"} and isinstance(thing(), set)


def test_a_body_may_write_the_file_itself(iv):
    """The escape hatch: take `out` and nothing is inferred about the value."""
    @iv.data("dump/page/", why="a rendered page", ext=".html", terminal=True)
    def page(out):
        out.write_text("<h1>hi</h1>")

    assert page() == "<h1>hi</h1>"


def test_a_body_that_returns_nothing_and_takes_no_out_is_an_error(iv):
    @iv.data("processed/out/", why="produces nothing")
    def out():
        return None

    with pytest.raises(DeclError, match="returned None"):
        out()


# ── what is refused ───────────────────────────────────────────────────────────

def test_one_dataset_has_one_producer(iv):
    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    with pytest.raises(DeclError, match="already written by"):
        @iv.data("raw/feed/", why="the same feed again")
        def feed2():
            return frame()


def test_building_one_stage_from_inside_another_is_refused(iv):
    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    @iv.data("processed/out/", why="reaches for a stage instead of declaring it")
    def out(unused=iv.all_of("raw/feed/", why="declared, but then ignored")):
        return feed()

    feed()
    with pytest.raises(DeclError, match="called from inside another stage"):
        out()


def test_a_parameter_iv_cannot_supply_is_refused(iv):
    with pytest.raises(DeclError, match="not something iv can supply"):
        @iv.data("processed/out/", why="takes something unexplained")
        def out(mystery):
            return frame()


def test_a_partition_relative_read_needs_a_partitioned_stage(iv):
    with pytest.raises(DeclError, match="only means something where there is a partition"):
        @iv.data("processed/out/", why="not partitioned, but reads as if it were")
        def out(box=iv.same_part("raw/box/", why="this season")):
            return box


def test_an_undeclared_read_of_the_tree_is_still_caught(iv):
    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()
    feed()

    @iv.data("processed/out/", why="reads behind iv's back")
    def out():
        return pl.read_parquet(list(iv.resolve_out("raw/feed/").iterdir())[0])

    with pytest.raises(DeclError, match="was not handed back by iv.reads"):
        out()


def test_loading_a_shard_that_was_never_built_says_so(iv):
    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    with pytest.raises(StateError, match="it was not built"):
        feed.load()


# ── the update: read your own last copy ───────────────────────────────────────

def test_a_stage_may_read_the_copy_it_is_about_to_overwrite(iv):
    """Excluded from the comparison, or the stage is permanently stale against itself."""
    day = ["day1"]

    @iv.data("config/today/", why="the clock", ext=".json")
    def today():
        return {"date": day[0]}

    @iv.data("raw/log/", why="a running log, appended once a day")
    def log(today=iv.all_of("config/today/", why="append once a day"),
            prior=iv.own_last_copy("raw/log/", why="yesterday's copy")):
        old = prior if prior is not None else pl.DataFrame(schema={"date": pl.Utf8})
        return pl.concat([old, pl.DataFrame({"date": [today["date"]]})]).unique("date")

    today(); log()
    assert log.is_current(), "its own copy must not make it stale against itself"

    day[0] = "day2"
    today()
    assert log().height == 2, "a new day appended"
    assert log.is_current()


# ── one computation, several outputs ──────────────────────────────────────────

def test_a_stage_with_several_outputs_runs_once(iv):
    ran = []

    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    @iv.step(outputs={"a": "processed/a/", "b": "processed/b/", "c": "processed/c/"},
             why="one computation, three tables")
    def fit(feed=iv.all_of("raw/feed/", why="the upstream")):
        ran.append(1)
        return {"a": feed, "b": feed.head(1), "c": feed.tail(1)}

    feed()
    assert fit() is True and len(ran) == 1
    assert fit() is False and len(ran) == 1
    for ds in ("processed/a/", "processed/b/", "processed/c/"):
        assert iv.is_current(ds)


def test_losing_any_one_output_brings_the_whole_stage_back(iv):
    """A crash between two of the writes leaves exactly this state."""
    import shutil
    ran = []

    @iv.data("raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.step(outputs={"a": "processed/a/", "b": "processed/b/"}, why="two tables")
    def fit(feed=iv.all_of("raw/feed/", why="the upstream")):
        ran.append(1)
        return {"a": feed, "b": feed.head(1)}

    feed(); fit()
    shutil.rmtree(iv.resolve_out("processed/b/"))
    assert fit() is True and len(ran) == 2
    assert iv.resolve_out("processed/b/").exists()


def test_an_undeclared_output_is_refused(iv):
    @iv.step(outputs={"a": "processed/a/"}, why="returns something it did not declare")
    def fit():
        return {"a": frame(), "surprise": frame()}

    with pytest.raises(DeclError, match="does not declare as outputs"):
        fit()


def test_a_missing_output_is_refused_unless_allowed(iv):
    @iv.step(outputs={"a": "processed/a/", "b": "processed/b/"}, why="returns one of two")
    def fit():
        return {"a": frame()}

    with pytest.raises(DeclError, match="declares the output 'b'"):
        fit()


def test_allow_missing_lets_an_output_stay_absent(iv):
    @iv.step(outputs={"a": "processed/a/",
                      "b": iv.output("processed/b/", allow_missing=True)},
             why="the second table is not always producible")
    def fit():
        return {"a": frame()}

    fit()
    assert iv.resolve_out("processed/a/").exists()
    assert not iv.resolve_out("processed/b/").exists()


def test_terminal_belongs_to_the_output_not_the_stage(iv):
    @iv.data("raw/feed/", why="the feed")
    def feed():
        return frame()

    @iv.step(outputs={"read_by_someone": "processed/a/",
                      "read_by_a_person": iv.output("processed/b/", terminal=True)},
             why="one fit, one table nothing downstream reads")
    def fit(feed=iv.all_of("raw/feed/", why="the upstream")):
        return {"read_by_someone": feed, "read_by_a_person": feed}

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(a=iv.all_of("processed/a/", why="the table a stage reads")):
        return a

    errors, _ = _graph.check(_graph.build(iv))
    assert not errors, errors


# ── two stages, two shards, one dataset ───────────────────────────────────────

def test_two_stages_may_share_a_dataset_by_writing_different_partitions(iv):
    @iv.data("raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.data("derived/blocks/", why="the first block", part={"source": "a"})
    def block_a(feed=iv.all_of("raw/feed/", why="the upstream")):
        return frame(2)

    @iv.data("derived/blocks/", why="the second block", part={"source": "b"})
    def block_b(feed=iv.all_of("raw/feed/", why="the upstream")):
        return frame(3)

    feed(); block_a(); block_b()
    names = sorted(p.name for p in iv.resolve_out("derived/blocks/").iterdir())
    assert len(names) == 2 and names[0].startswith("source=a")
    assert pl.read_parquet(iv.reads("derived/blocks/", why="both")).height == 5


def test_the_same_shard_written_twice_is_refused(iv):
    @iv.data("derived/blocks/", why="the first block", part={"source": "a"})
    def block_a():
        return frame()

    with pytest.raises(DeclError, match="different partitions"):
        @iv.data("derived/blocks/", why="the same shard again", part={"source": "a"})
        def block_a2():
            return frame()


def test_a_fixed_partition_takes_no_call_argument(iv):
    @iv.data("derived/blocks/", why="one block", part={"source": "a"})
    def block():
        return frame()

    with pytest.raises(DeclError, match="names no other"):
        block("b")


# ── one computation, many shards ──────────────────────────────────────────────

def test_split_writes_a_shard_per_returned_key(iv):
    ran = []

    @iv.data("raw/feed/", why="the feed", once=True)
    def feed():
        return pl.DataFrame({"season": ["2024", "2024", "2025"], "pts": [1, 2, 3]})

    @iv.data("derived/features/", why="built in one pass, split by season",
             part="season", split=True)
    def features(feed=iv.all_of("raw/feed/", why="every season at once")):
        ran.append(1)
        return {str(s): rows for (s,), rows in feed.group_by("season", maintain_order=True)}

    feed(); features()
    assert len(ran) == 1
    assert sorted(_sh.current_shards(iv.resolve_out("derived/features/"))) == \
        ["season=2024", "season=2025"]
    assert features() is False, "every shard is current, so the pass must not repeat"


def test_a_split_stage_must_return_a_mapping(iv):
    @iv.data("derived/features/", why="split, but returns a frame",
             part="season", split=True)
    def features():
        return frame()

    with pytest.raises(DeclError, match="returns {season: value}"):
        features()


def test_split_needs_a_partition_key(iv):
    with pytest.raises(DeclError, match="needs a partition key"):
        @iv.data("derived/features/", why="split with nothing to split on", split=True)
        def features():
            return {"a": frame()}


# ── external sources ──────────────────────────────────────────────────────────

def test_an_external_source_is_declared_and_drawn(iv):
    @iv.data("raw/feed/", why="fetched from an API",
             external={"espn/feeds": "ESPN's season files"})
    def feed(clock=iv.all_of("config/today/", why="poll once a day")):
        return frame()

    @iv.data("config/today/", why="the clock", ext=".json")
    def today():
        return {"date": "2026-08-22"}

    g = _graph.build(iv)
    node = next(n for n in g.stages if n.endswith("::feed"))
    assert [s.dataset for s in g.stages[node].externals] == ["external:espn/feeds"]


# ── a stage that writes nothing ───────────────────────────────────────────────

def test_a_stage_may_write_nothing_and_still_declare_what_it_reads(iv):
    """A fetch fills a download cache; a publish uploads. Neither leaves an artifact iv
    names, so there is nothing to be stale and nothing to skip on — but the reads are
    still worth declaring, or the graph cannot draw the edge."""
    ran = []

    @iv.data("dump/site/", why="the payload", terminal=True)
    def site():
        return frame()

    @iv.step(why="copy the payload somewhere outside the tree",
             external={"gs://bucket": "the bucket the app reads"})
    def publish(payload=iv.all_of("dump/site/", load=False, why="what to upload")):
        ran.append(list(payload))

    site()
    assert publish() is True and len(ran) == 1
    assert publish() is True and len(ran) == 2, "nothing on disk can say it is done"
    assert ran[0] and str(ran[0][0]).endswith(".parquet"), "it was handed the paths"

    g = _graph.build(iv)
    node = next(n for n in g.stages if n.endswith("::publish"))
    assert [s.dataset for s in g.stages[node].triggers] == ["dump/site/"]
    assert [s.dataset for s in g.stages[node].externals] == ["external:gs://bucket"]
    assert not g.stages[node].outputs
    errors, warns = _graph.check(g)
    assert not errors, errors
    assert not [w for w in warns if "RUNS ONCE" in w], (
        "it writes nothing, so there is no artifact for the warning to be about")


def test_a_stage_that_writes_nothing_has_no_partition(iv):
    with pytest.raises(DeclError, match="no shard for part= to name"):
        @iv.step(why="writes nothing but claims a partition", part="season")
        def act(x=iv.all_of("raw/feed/", why="something")):
            pass
