

from __future__ import annotations

import polars as pl
import pytest
from typer.testing import CliRunner

from iv import Pipeline
from iv import graph as _graph
from iv.cli import app
from iv import shards as _sh
from iv.errors import DeclError, StateError


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def frame(n=2, extra=0):
    return pl.DataFrame({"player": list(range(n)), "pts": [x + extra for x in range(n)]})


def test_a_derived_asset_builds_then_skips(iv):
    ran = []

    @iv.data(dataset="raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.data(dataset="processed/out/", why="passthrough")
    def out(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return feed

    feed()
    assert out().height == 2 and len(ran) == 1
    assert out().height == 2 and len(ran) == 1, "nothing moved, so it must not run again"


def test_composite_partitions_match_every_dimension_and_for_each_dicts(tmp_path):
    iv = Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage", project=tmp_path)
    raw = iv.source("raw/games/", why="league-season games")
    for league in ("nba", "wnba"):
        for season in ("2024", "2025"):
            with iv.writes(raw.dataset, why="seed", part={"league": league, "season": season}) as out:
                frame(extra=1 if league == "nba" else 2).write_parquet(out)

    @iv.data(dataset="processed/games/", why="one league-season result",
             part=("league", "season"),
             universe=[{"league": league, "season": season}
                       for league in ("nba", "wnba") for season in ("2024", "2025")])
    def games(league, season, source=iv.same_part(raw, why="the matching games")):
        return source.with_columns(pl.lit(f"{league}-{season}").alias("slice"))

    rebuilt = games.for_each([{"league": "nba", "season": "2025"}])
    assert rebuilt == [{"league": "nba", "season": "2025"}]
    assert games(league="nba", season="2025").get_column("slice").unique().item() == "nba-2025"


def test_a_version_reruns_its_stage_but_identical_output_stops_downstream(iv):
    calls = []

    @iv.data(dataset="raw/feed/", why="stable feed", once=True)
    def feed():
        return frame()

    @iv.data(dataset="processed/mid/", why="versioned transform")
    def mid(source=iv.all_of(feed, why="the feed")):
        calls.append("mid")
        return source

    @iv.data(dataset="processed/end/", why="consumer")
    def end(source=iv.all_of(mid, why="the versioned output")):
        calls.append("end")
        return source

    feed(); mid(); end()
    iv._versions[mid.dataset] = "1"
    assert mid.why_stale()
    mid()
    assert end.why_stale() is None
    end()
    assert calls == ["mid", "end", "mid"]


def test_an_action_cannot_have_a_version(iv):
    with pytest.raises(DeclError, match="version= belongs to an output"):
        @iv.step(why="publishes a report", version="1")
        def publish():
            pass


def test_runner_targets_one_composite_partition_and_only_requires_current_upstream(tmp_path, monkeypatch):
    iv = Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage", project=tmp_path)

    universe = [{"league": league, "season": "2025"} for league in ("nba", "wnba")]

    @iv.data(dataset="raw/games/", why="one feed shard", part=("league", "season"),
             universe=universe, once=True)
    def raw(league, season):
        return frame(extra=1 if league == "nba" else 2)

    @iv.data(dataset="processed/games/", why="one derived shard", part=("league", "season"),
             universe=universe)
    def games(league, season, source=iv.same_part(raw, why="the matching feed")):
        return source

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    runner = CliRunner()
    blocked = runner.invoke(app, ["run", "--only", "games", "--part", "league=nba", "--part", "season=2025"])
    assert blocked.exit_code == 1 and "stale upstream" in blocked.output
    result = runner.invoke(app, ["run", "--up-to", "games", "--part", "league=nba", "--part", "season=2025"])
    assert result.exit_code == 0, result.output
    assert games.is_current(league="nba", season="2025")
    assert not games.is_current(league="wnba", season="2025")


def test_runner_force_allows_only_with_a_stale_upstream(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="a root that may have changed")
    def feed():
        return frame()

    @iv.data(dataset="processed/out/", why="uses the existing feed")
    def out(source=iv.all_of(feed, why="the feed")):
        return source

    feed()
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["run", "--only", "out", "--force"])
    assert result.exit_code == 0, result.output
    assert out.is_current()


def test_the_runner_builds_a_split_stage_once_not_once_per_partition(tmp_path, monkeypatch):
    iv = Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage", project=tmp_path)
    ran = []

    @iv.data(dataset="raw/feed/", why="every season in one file", once=True)
    def feed():
        return pl.DataFrame({"season": ["2024", "2024", "2025"], "pts": [1, 2, 3]})

    @iv.data(dataset="processed/features/", why="one pass, split by season",
             part="season", split=True)
    def features(source=iv.all_of(feed, why="every season at once")):
        ran.append(1)
        return {str(s): rows for (s,), rows in source.group_by("season", maintain_order=True)}

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["run", "--up-to", "features"])
    assert result.exit_code == 0, result.output
    assert ran == [1]
    assert sorted(_sh.current_shards(iv.resolve_out("processed/features/"))) == \
        ["season=2024", "season=2025"]


def test_a_stage_universe_decides_what_the_runner_enumerates(tmp_path, monkeypatch):
    iv = Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage", project=tmp_path)
    built = []

    @iv.data(dataset="raw/feed/", why="every season the feed has",
             part="season", universe=["2006", "2024", "2025"], once=True)
    def feed(season):
        return frame()

    @iv.data(dataset="processed/fit/", why="only the seasons with a prior behind them",
             part="season", universe=lambda: ["2024", "2025"])
    def fit(season, source=iv.same_part(feed, why="the matching feed")):
        built.append(season)
        return source

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["run", "--up-to", "fit"])
    assert result.exit_code == 0, result.output
    assert built == ["2024", "2025"], "2006 is outside this stage's universe"
    assert sorted(_sh.current_shards(iv.resolve_out("raw/feed/"))) == \
        ["season=2006", "season=2024", "season=2025"]


def test_runner_requires_a_stage_universe_for_dynamic_work(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="one shard per season", part="season", once=True)
    def feed(season):
        return frame()

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code == 1
    assert "iv run needs universe=" in result.output
    assert feed("2025").height == 2, "a direct, fully named shard stays valid"


def test_runner_prints_plan_progress_and_stage_output(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="a visible step", once=True)
    def feed():
        print("fetching the official feed")
        return frame()

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    runner = CliRunner()
    first = runner.invoke(app, ["run"])
    assert first.exit_code == 0, first.output
    assert "iv run · 1 stage shard(s)" in first.output
    assert "[1/1]" in first.output
    assert "rebuild — not on disk" in first.output
    assert "output" in first.output and "│ fetching the official feed" in first.output
    assert "reran (" in first.output and "1 reran, 0 current — skipped" in first.output

    second = runner.invoke(app, ["run"])
    assert second.exit_code == 0, second.output
    assert "current — skipped (" in second.output
    assert "0 reran, 1 current — skipped" in second.output


def test_determinism_runs_one_stage_twice_without_touching_production_outputs(iv, monkeypatch):
    @iv.data(dataset="processed/out/", why="a stable result")
    def out():
        return frame()

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["determinism", "--only", "out"])
    assert result.exit_code == 0, result.output
    assert "deterministic" in result.output
    assert not iv.resolve_out(out.dataset).exists(), "trials must never write production output"


def test_determinism_reports_different_output_fingerprints(iv, monkeypatch):
    calls = [0]

    @iv.data(dataset="processed/out/", why="an unstable result")
    def out():
        calls[0] += 1
        return frame(extra=calls[0])

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["determinism", "--only", "out"])
    assert result.exit_code == 1
    assert "not deterministic" in result.output
    assert "processed/out/" in result.output and "!=" in result.output


def test_determinism_can_target_one_partition(iv, monkeypatch):
    @iv.data(dataset="processed/out/", part="season", universe=["2024", "2025"],
             why="one stable seasonal result")
    def out(season):
        return frame(extra=int(season))

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(
        app, ["determinism", "--only", "out", "--part", "season=2025"])
    assert result.exit_code == 0, result.output
    assert "1 stage shard(s) matched" in result.output


def test_determinism_sample_uses_the_last_declared_partition_and_skips_actions(iv, monkeypatch):
    seen = []

    @iv.data(dataset="processed/out/", part="season", universe=["2025", "2023", "2024"],
             why="one stable seasonal result")
    def out(season):
        seen.append(season)
        return frame(extra=int(season))

    @iv.step(why="has no output")
    def publish():
        raise AssertionError("actions are skipped")

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["determinism", "--sample"])
    assert result.exit_code == 0, result.output
    assert seen == ["2025", "2025"]
    assert "ok       " in result.output and "season=2025" in result.output
    assert "skipped" in result.output and "publish" in result.output


def test_determinism_refuses_actions(iv, monkeypatch):
    @iv.step(why="has no output")
    def publish():
        raise AssertionError("must not run")

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["determinism", "--only", "publish"])
    assert result.exit_code == 1
    assert "action with no output" in result.output


def test_runner_names_an_upstream_that_changed_earlier_in_the_run(iv, monkeypatch):
    value = [1]

    @iv.data(dataset="raw/feed/", why="a moving root")
    def feed():
        return frame(extra=value[0])

    @iv.data(dataset="processed/out/", why="a consumer")
    def out(source=iv.all_of(feed, why="the feed")):
        return source

    feed(); out()
    value[0] = 2
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code == 0, result.output
    assert "rebuild — root has no declared inputs" in result.output
    assert "rebuild — upstream changed: raw/feed/(one shard)" in result.output


def test_runner_writes_one_incremental_output_log(iv, monkeypatch, tmp_path):
    @iv.data(dataset="raw/feed/", why="a logged stage", once=True)
    def feed():
        print("first line")
        print("second line", file=__import__("sys").stderr)
        return frame()

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    log = tmp_path / "logs" / "run.log"
    runner = CliRunner()

    first = runner.invoke(app, ["run", "--log", str(log)])
    assert first.exit_code == 0, first.output
    text = log.read_text()
    assert "iv run · 1 stage shard(s)" in text
    assert "[1/1]" in text and "first line" in text and "second line" in text
    assert "[iv] rebuild — not on disk" in text
    assert "[iv] reran (" in text and "1 reran, 0 current — skipped" in text

    second = runner.invoke(app, ["run", "--log", str(log)])
    assert second.exit_code == 0, second.output
    text = log.read_text()
    assert text.count("iv run ·") == 1, "each run replaces rather than appends to the log"
    assert "[iv] current — skipped (" in text
    assert "0 reran, 1 current — skipped" in text


def test_runner_preserves_output_log_when_a_stage_fails(iv, monkeypatch, tmp_path):
    @iv.data(dataset="raw/feed/", why="a failing stage", once=True)
    def feed():
        print("evidence before failure")
        raise RuntimeError("boom")

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    log = tmp_path / "failed.log"
    result = CliRunner().invoke(app, ["run", "--log", str(log)])
    assert result.exit_code == 1
    text = log.read_text()
    assert "evidence before failure" in text
    assert "[iv] failed (" in text


def test_impact_shows_a_stage_cone_and_tick_propagation(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="a stable feed", once=True)
    def feed():
        return frame()

    @iv.data(dataset="processed/out/", why="a consumer")
    def out(source=iv.all_of(feed, why="the feed")):
        return source

    feed(); out()
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["impact", "feed", "--tick"])
    assert result.exit_code == 0, result.output
    assert "upstream" in result.output and "this stage" in result.output
    assert "downstream" in result.output and "out" in result.output
    assert "if " in result.output and "will run" in result.output
    assert "may rebuild" in result.output
    assert "processed/out/(one shard)" in result.output


def test_impact_ignores_a_missing_dynamic_output_when_tracing(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="one season", part="season", once=True,
             universe=["2025"])
    def feed(season):
        return frame()

    @iv.data(dataset="processed/out/", why="one derived season", part="season",
             universe=["2025"])
    def out(season, source=iv.same_part(feed, why="the matching season")):
        return source

    feed("2025")
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["impact", "feed", "--tick"])
    assert result.exit_code == 0, result.output


def test_impact_ticks_only_the_selected_output_partition(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="one shard per season", part="season",
             universe=["2025", "2026"], once=True)
    def feed(season):
        return frame(extra=int(season))

    @iv.data(dataset="processed/same/", why="the matching season", part="season",
             universe=["2025", "2026"])
    def same(season, source=iv.same_part(feed, why="this season")):
        return source

    @iv.data(dataset="processed/cumulative/", why="history through the season",
             part="season", universe=["2025", "2026"])
    def cumulative(season, source=iv.before_part(
            feed, inclusive=True, why="history through this season")):
        return source

    feed.for_each(); same.for_each(); cumulative.for_each()
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(
        app, ["impact", "feed", "--tick", "--tick-part", "season=2026"])
    assert result.exit_code == 0, result.output
    assert "if " in result.output and "[season=2026] changes" in result.output
    assert "processed/same/season=2026" in result.output
    assert "processed/cumulative/season=2026" in result.output
    assert "processed/same/season=2025" not in result.output
    assert "processed/cumulative/season=2025" not in result.output


def test_impact_tick_part_is_an_exact_output_selector(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="an unpartitioned feed", once=True)
    def feed():
        return frame()

    feed()
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    runner = CliRunner()

    missing_tick = runner.invoke(app, ["impact", "feed", "--tick-part", "season=2026"])
    assert missing_tick.exit_code == 1
    assert "--tick-part requires --tick" in missing_tick.output

    unpartitioned = runner.invoke(
        app, ["impact", "feed", "--tick", "--tick-part", "season=2026"])
    assert unpartitioned.exit_code == 1
    assert "no output shards on disk matching season=2026" in unpartitioned.output
    assert "Available: (one shard)" in unpartitioned.output


def test_impact_tick_part_supports_composite_partitions(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="league-season shards",
             part=("league", "season"), once=True)
    def feed(league, season):
        return frame(extra=len(league) + int(season))

    for league in ("nba", "wnba"):
        feed(league=league, season="2026")
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, [
        "impact", "feed", "--tick",
        "--tick-part", "league=wnba", "--tick-part", "season=2026",
    ])
    assert result.exit_code == 0, result.output
    assert "[league=wnba, season=2026] changes" in result.output


def test_impact_propagation_respects_a_fixed_output_partition(iv, monkeypatch):
    @iv.data(dataset="raw/feed/", why="the clock", once=True)
    def feed():
        return frame()

    @iv.data(dataset="history/snapshots/", why="today's snapshot",
             part={"date": "2026-08-25"})
    def snapshot(source=iv.all_of(feed, why="what changed today")):
        return source

    feed(); snapshot()
    with iv.writes("history/snapshots/", why="an older snapshot",
                   part={"date": "2026-08-24"}) as out:
        frame(extra=1).write_parquet(out)

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["impact", "feed", "--tick"])
    assert result.exit_code == 0, result.output
    assert "history/snapshots/date=2026-08-25" in result.output
    assert "history/snapshots/date=2026-08-24" not in result.output


def test_impact_ticks_only_shards_owned_by_a_writer_of_a_shared_dataset(iv,
                                                                         monkeypatch):
    blocks = iv.dataset("derived/blocks/", why="blocks with separate writers")

    @iv.data(dataset=blocks, why="the named block", part={"source": "named"})
    def named():
        return frame()

    @iv.data(dataset=blocks, why="the other named block", part={"source": "other"})
    def other():
        return frame(extra=1)

    named(); other()
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    runner = CliRunner()

    named_result = runner.invoke(app, ["impact", "named", "--tick"])
    assert named_result.exit_code == 0, named_result.output
    assert "derived/blocks/source=named" not in named_result.output
    assert "Available" not in named_result.output

    wrong_part = runner.invoke(app, [
        "impact", "named", "--tick", "--tick-part", "source=other",
    ])
    assert wrong_part.exit_code == 1
    assert "Available: source=named" in wrong_part.output


def test_for_each_with_no_argument_builds_the_declared_universe(iv):
    @iv.data(dataset="raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.data(dataset="processed/out/", why="one shard per declared season",
             part="season", universe=["2024", "2025"])
    def out(season, source=iv.all_of(feed, why="the feed")):
        return source

    feed()
    assert out.for_each() == ["2024", "2025"]
    assert out.for_each() == [], "everything it declares is current"


def test_for_each_with_no_argument_and_no_universe_says_so(iv):
    @iv.data(dataset="raw/feed/", why="the feed", part="season")
    def feed(season):
        return frame()

    with pytest.raises(DeclError, match="declares none"):
        feed.for_each()


def test_a_universe_needs_a_partition_to_be_keyed_on(iv):
    with pytest.raises(DeclError, match="needs part= to say what they are keyed on"):
        @iv.data(dataset="raw/feed/", why="unpartitioned", universe=["2024"])
        def feed():
            return frame()


def test_a_split_stage_has_no_per_partition_universe(iv):
    with pytest.raises(DeclError, match="no per-partition universe"):
        @iv.data(dataset="raw/feed/", why="one pass", part="season", split=True,
                 universe=["2024"])
        def feed():
            return {"2024": frame()}


def test_a_partition_with_nothing_to_build_returns_none_under_allow_missing(iv):
    ran = []

    @iv.data(dataset="raw/feed/", why="the feed", part="season", allow_missing=True)
    def feed(season):
        ran.append(season)
        return None if season == "2006" else frame()

    feed("2006")
    feed("2024")
    assert ran == ["2006", "2024"]
    assert sorted(_sh.current_shards(iv.resolve_out("raw/feed/"))) == ["season=2024"]
    assert feed.why_stale("2006"), "an absent shard stays stale, so the next run retries"
    assert feed.why_stale("2024") is None


def test_returning_none_without_allow_missing_still_names_the_way_out(iv):
    @iv.data(dataset="raw/feed/", why="the feed", part="season")
    def feed(season):
        return None

    with pytest.raises(DeclError, match="allow_missing=True if this partition"):
        feed("2006")


def test_a_moved_upstream_rebuilds(iv):
    n = [2]
    ran = []

    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame(n[0])

    @iv.data(dataset="processed/out/", why="passthrough")
    def out(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return feed

    feed(); out()
    assert len(ran) == 1
    n[0] = 9
    feed()
    assert out.why_stale().startswith("its inputs moved")
    assert out().height == 9 and len(ran) == 2


def test_a_rebuild_that_does_not_move_the_bytes_stops_there(iv):

    ran = []

    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()

    @iv.data(dataset="processed/out/", why="passthrough")
    def out(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return feed

    feed(); out()
    feed()
    out()
    assert len(ran) == 1, "the bytes did not move, so the downstream must not follow"


def test_a_root_runs_every_time_because_nothing_else_can_notice_the_world(iv):

    ran = []
    n = [2]

    @iv.data(dataset="raw/feed/", why="fetched from outside")
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

    @iv.data(dataset="raw/archive/", why="a one-time backfill", once=True)
    def archive():
        ran.append(1)
        return frame()

    archive(); archive()
    assert len(ran) == 1


def test_a_root_that_always_runs_is_not_warned_about_as_running_once(iv):
    @iv.data(dataset="raw/feed/", why="fetched from outside")
    def feed():
        return frame()

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(feed=iv.all_of(feed, why="the feed")):
        return feed

    errors, warns = _graph.check(_graph.build(iv))
    assert not errors
    assert not [w for w in warns if "RUNS ONCE" in w], (
        "a root asset re-runs every time, so the warning would be false")


def test_once_is_warned_about(iv):
    @iv.data(dataset="raw/archive/", why="a one-time backfill", once=True)
    def archive():
        return frame()

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(a=iv.all_of(archive, why="the backfill")):
        return a

    _, warns = _graph.check(_graph.build(iv))
    assert [w for w in warns if "RUNS ONCE" in w]


def seasons_pipeline(iv, pts=None):
    pts = pts if pts is not None else {"2024": 10, "2025": 20, "2026": 30}
    ran = []

    @iv.data(dataset="raw/box/", why="raw box for one season", part="season")
    def box(season):
        ran.append(f"box:{season}")
        return pl.DataFrame({"player": [1, 2], "pts": [pts[season], pts[season] + 1]})

    @iv.data(dataset="processed/features/", why="per-season features", part="season")
    def features(box=iv.same_part(box, why="this season's box")):
        ran.append("features")
        return box.with_columns((pl.col("pts") * 2).alias("z"))

    @iv.data(dataset="processed/cohorts/", why="a fit on prior seasons only", part="season")
    def cohorts(past=iv.before_part(features, why="every prior season")):
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
    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()

    with pytest.raises(DeclError, match="takes no partition value"):
        feed("2024")
    with pytest.raises(DeclError, match="nothing to iterate"):
        feed.for_each(["2024"])


def test_a_dict_round_trips_through_json(iv):
    @iv.data(dataset="config/knobs/", why="the knobs", ext=".json")
    def knobs():
        return {"half_life": 4.0, "seed": 0}

    first = knobs()
    assert first == {"half_life": 4.0, "seed": 0}
    assert knobs() == first and isinstance(knobs(), dict)


def test_a_dict_is_refused_by_parquet_rather_than_silently_reshaped(iv):


    @iv.data(dataset="config/knobs/", why="the knobs")
    def knobs():
        return {"half_life": 4.0}

    with pytest.raises(DeclError, match="would give two different types"):
        knobs()


def test_a_frame_round_trips_through_parquet(iv):
    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()

    assert isinstance(feed(), pl.DataFrame)
    assert feed().equals(frame())


def test_an_arbitrary_object_round_trips_through_pickle(iv):
    @iv.data(dataset="config/thing/", why="something exotic", ext=".pkl")
    def thing():
        return {"a", "b"}

    assert thing() == {"a", "b"} and isinstance(thing(), set)


def test_a_body_may_write_the_file_itself(iv):

    @iv.data(dataset="dump/page/", why="a rendered page", ext=".html")
    def page(out):
        out.write_text("<h1>hi</h1>")

    assert page() == "<h1>hi</h1>"


def test_a_body_that_returns_nothing_and_takes_no_out_is_an_error(iv):
    @iv.data(dataset="processed/out/", why="produces nothing")
    def out():
        return None

    with pytest.raises(DeclError, match="returned None"):
        out()


def test_one_dataset_has_one_producer(iv):
    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()

    with pytest.raises(DeclError, match="already written by"):
        @iv.data(dataset="raw/feed/", why="the same feed again")
        def feed2():
            return frame()


def test_building_one_stage_from_inside_another_is_refused(iv):
    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()

    @iv.data(dataset="processed/out/", why="reaches for a stage instead of declaring it")
    def out(unused=iv.all_of(feed, why="declared, but then ignored")):
        return feed()

    feed()
    with pytest.raises(DeclError, match="called from inside another stage"):
        out()


def test_a_parameter_iv_cannot_supply_is_refused(iv):
    with pytest.raises(DeclError, match="not something iv can supply"):
        @iv.data(dataset="processed/out/", why="takes something unexplained")
        def out(mystery):
            return frame()


def test_a_partition_relative_read_needs_a_partitioned_stage(iv):
    box = iv.source("raw/box/", why="arrives from outside")
    with pytest.raises(DeclError, match="only means something where there is a partition"):
        @iv.data(dataset="processed/out/", why="not partitioned, reads as if it were")
        def out(b=iv.same_part(box, why="this season")):
            return b


def test_an_undeclared_read_of_the_tree_is_still_caught(iv):
    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()
    feed()

    @iv.data(dataset="processed/out/", why="reads behind iv's back")
    def out():
        return pl.read_parquet(list(iv.resolve_out("raw/feed/").iterdir())[0])

    with pytest.raises(DeclError, match="asks the data tree a question"):
        out()


def test_loading_a_shard_that_was_never_built_says_so(iv):
    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()

    with pytest.raises(StateError, match="it was not built"):
        feed.load()


def test_a_stage_may_read_the_copy_it_is_about_to_overwrite(iv):

    day = ["day1"]

    @iv.data(dataset="config/today/", why="the clock", ext=".json")
    def today():
        return {"date": day[0]}

    @iv.data(dataset="raw/log/", why="a running log, appended once a day")
    def log(today=iv.all_of(today, why="append once a day"),
            prior=iv.own_last_copy(why="yesterday's copy")):
        old = prior if prior is not None else pl.DataFrame(schema={"date": pl.Utf8})
        return pl.concat([old, pl.DataFrame({"date": [today["date"]]})]).unique("date")

    today(); log()
    assert log.is_current(), "its own copy must not make it stale against itself"

    day[0] = "day2"
    today()
    assert log().height == 2, "a new day appended"
    assert log.is_current()


def test_a_stage_with_several_outputs_runs_once(iv):
    ran = []

    @iv.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()

    @iv.step(output={"a": "processed/a/", "b": "processed/b/", "c": "processed/c/"},
             why="one computation, three tables")
    def fit(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return {"a": feed, "b": feed.head(1), "c": feed.tail(1)}

    feed()
    assert fit() is True and len(ran) == 1
    assert fit() is False and len(ran) == 1
    for ds in ("processed/a/", "processed/b/", "processed/c/"):
        assert iv.is_current(ds)


def test_losing_any_one_output_brings_the_whole_stage_back(iv):

    import shutil
    ran = []

    @iv.data(dataset="raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.step(output={"a": "processed/a/", "b": "processed/b/"}, why="two tables")
    def fit(feed=iv.all_of(feed, why="the upstream")):
        ran.append(1)
        return {"a": feed, "b": feed.head(1)}

    feed(); fit()
    shutil.rmtree(iv.resolve_out("processed/b/"))
    assert fit() is True and len(ran) == 2
    assert iv.resolve_out("processed/b/").exists()


def test_an_undeclared_output_is_refused(iv):
    @iv.step(output={"a": "processed/a/"}, why="returns something it did not declare")
    def fit():
        return {"a": frame(), "surprise": frame()}

    with pytest.raises(DeclError, match="does not declare as outputs"):
        fit()


def test_a_missing_output_is_refused_unless_allowed(iv):
    @iv.step(output={"a": "processed/a/", "b": "processed/b/"}, why="returns one of two")
    def fit():
        return {"a": frame()}

    with pytest.raises(DeclError, match="declares the output 'b'"):
        fit()


def test_allow_missing_lets_an_output_stay_absent(iv):
    @iv.step(output={"a": "processed/a/",
                      "b": iv.dataset("processed/b/", why="the optional table",
                                   allow_missing=True)},
             why="the second table is not always producible")
    def fit():
        return {"a": frame()}

    fit()
    assert iv.resolve_out("processed/a/").exists()
    assert not iv.resolve_out("processed/b/").exists()


def test_a_dataset_nothing_reads_is_terminal_without_being_told(iv):


    feed = iv.source("raw/feed/", why="arrives from outside")

    @iv.step(output={"read_by_someone": "processed/a/", "read_by_a_person": "processed/b/"},
             why="one fit, one table nothing downstream reads")
    def fit(f=iv.all_of(feed, why="the upstream")):
        return {"read_by_someone": f, "read_by_a_person": f}

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(a=iv.all_of(fit["read_by_someone"], why="the table a stage reads")):
        return a

    g = _graph.build(iv)
    assert g.is_terminal("processed/b/"), "nothing reads it, so it is a leaf"
    assert not g.is_terminal("processed/a/"), "site reads it"
    errors, _ = _graph.check(g)
    assert errors == [], errors

def test_two_stages_may_share_a_dataset_by_writing_different_partitions(iv):
    @iv.data(dataset="raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.data(dataset="derived/blocks/", why="the first block", part={"source": "a"})
    def block_a(feed=iv.all_of(feed, why="the upstream")):
        return frame(2)

    @iv.data(dataset="derived/blocks/", why="the second block", part={"source": "b"})
    def block_b(feed=iv.all_of(feed, why="the upstream")):
        return frame(3)

    feed(); block_a(); block_b()
    names = sorted(p.name for p in iv.resolve_out("derived/blocks/").iterdir())
    assert len(names) == 2 and names[0].startswith("source=a")
    assert pl.read_parquet(iv.reads("derived/blocks/", why="both")).height == 5


def test_the_same_shard_written_twice_is_refused(iv):
    @iv.data(dataset="derived/blocks/", why="the first block", part={"source": "a"})
    def block_a():
        return frame()

    with pytest.raises(DeclError, match="different partitions"):
        @iv.data(dataset="derived/blocks/", why="the same shard again", part={"source": "a"})
        def block_a2():
            return frame()


def test_a_fixed_partition_takes_no_call_argument(iv):
    @iv.data(dataset="derived/blocks/", why="one block", part={"source": "a"})
    def block():
        return frame()

    with pytest.raises(DeclError, match="names no other"):
        block("b")


def test_split_writes_a_shard_per_returned_key(iv):
    ran = []

    @iv.data(dataset="raw/feed/", why="the feed", once=True)
    def feed():
        return pl.DataFrame({"season": ["2024", "2024", "2025"], "pts": [1, 2, 3]})

    @iv.data(dataset="derived/features/", why="built in one pass, split by season",
             part="season", split=True)
    def features(feed=iv.all_of(feed, why="every season at once")):
        ran.append(1)
        return {str(s): rows for (s,), rows in feed.group_by("season", maintain_order=True)}

    feed(); features()
    assert len(ran) == 1
    assert sorted(_sh.current_shards(iv.resolve_out("derived/features/"))) == \
        ["season=2024", "season=2025"]
    assert features() is False, "every shard is current, so the pass must not repeat"


def test_a_split_stage_must_return_a_mapping(iv):
    @iv.data(dataset="derived/features/", why="split, but returns a frame",
             part="season", split=True)
    def features():
        return frame()

    with pytest.raises(DeclError, match="returns {season: value}"):
        features()


def test_split_needs_a_partition_key(iv):
    with pytest.raises(DeclError, match="needs a partition key"):
        @iv.data(dataset="derived/features/", why="split with nothing to split on", split=True)
        def features():
            return {"a": frame()}


def test_split_cannot_write_through_out(iv):
    with pytest.raises(DeclError, match="split=True returns many partition shards"):
        @iv.data(dataset="derived/features/", why="wrong split protocol",
                 part="season", split=True)
        def features(out):
            frame().write_parquet(out)


def test_stage_part_and_output_shard_cannot_both_name_ownership(iv):
    shared = iv.dataset("derived/features/", why="a shared table")
    with pytest.raises(DeclError, match="partition ownership is declared both"):
        @iv.data(dataset=shared.shard(source="intl"), why="wrong ownership source",
                 part="season")
        def features(season):
            return frame()


def test_one_stage_cannot_claim_the_same_output_shard_twice(iv):
    with pytest.raises(DeclError, match="both write"):
        @iv.step(output={"first": "derived/features/", "second": "derived/features/"},
                 why="two names for one shard")
        def features():
            return {"first": frame(), "second": frame()}


def test_an_external_source_is_declared_and_drawn(iv):
    @iv.data(dataset="config/today/", why="the clock", ext=".json")
    def today():
        return {"date": "2026-08-22"}

    @iv.data(dataset="raw/feed/", why="fetched from an API",
             external={"espn/feeds": "ESPN's season files"})
    def feed(clock=iv.all_of(today, as_paths=True, why="poll once a day")):
        return frame()

    g = _graph.build(iv)
    node = next(n for n in g.stages if n.endswith("::feed"))
    assert [s.dataset for s in g.stages[node].externals] == ["external:espn/feeds"]


def test_a_stage_may_write_nothing_and_still_declare_what_it_reads(iv):


    ran = []

    @iv.data(dataset="dump/site/", why="the payload")
    def site():
        return frame()

    @iv.step(why="copy the payload somewhere outside the tree",
             external={"gs://bucket": "the bucket the app reads"})
    def publish(payload=iv.all_of(site, as_paths=True, why="what to upload")):
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
    feed = iv.source("raw/feed/", why="arrives from outside")
    with pytest.raises(DeclError, match="no shard for part= to name"):
        @iv.step(why="writes nothing but claims a partition", part="season")
        def act(x=iv.all_of(feed, why="something")):
            pass


def test_as_paths_hands_back_paths_and_the_default_hands_back_contents(iv):
    @iv.data(dataset="raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    got = {}

    @iv.data(dataset="processed/contents/", why="wants the frame")
    def contents(f=iv.all_of(feed, why="the upstream")):
        got["contents"] = f
        return f

    @iv.data(dataset="processed/paths/", why="wants the paths")
    def paths(f=iv.all_of(feed, as_paths=True, why="the upstream")):
        got["paths"] = f
        return frame()

    feed(); contents(); paths()
    assert isinstance(got["contents"], pl.DataFrame)
    assert isinstance(got["paths"], list)
    assert str(got["paths"][0]).endswith(".parquet")


def test_an_optional_read_that_selects_nothing_is_empty_either_way(iv):


    got = {}
    absent = iv.source("raw/absent/", why="arrives from outside")

    @iv.data(dataset="processed/out/", why="reads what may not be there")
    def out(p=iv.all_of(absent, optional=True, as_paths=True, why="maybe"),
            c=iv.all_of(absent, optional=True, why="maybe")):
        got.update(paths=p, contents=c)
        return frame()

    out()
    assert got["paths"] == [] and got["contents"] is None


def test_naming_the_stage_builds_the_same_graph_as_naming_the_path(iv):
    @iv.data(dataset="raw/feed/", why="the feed", once=True)
    def feed():
        return frame()

    @iv.data(dataset="processed/out/", why="passthrough")
    def out(f=iv.all_of(feed, why="the upstream")):
        return f

    assert [r.dataset for r in out.reads] == ["raw/feed/"]
    errors, _ = _graph.check(_graph.build(iv))
    assert errors == [], errors


def test_naming_a_stage_that_writes_several_does_not_say_which(iv):
    @iv.step(output={"a": "out/a/", "b": "out/b/"}, why="two tables")
    def two():
        return {"a": frame(), "b": frame()}

    with pytest.raises(DeclError, match="does not say which"):
        iv.all_of(two, why="ambiguous")


def test_a_bare_own_last_copy_means_this_stage_s_own_output(iv):

    day = ["day1"]
    ran = []

    @iv.data(dataset="config/today/", why="the clock", ext=".json")
    def today():
        return {"date": day[0]}

    @iv.data(dataset="raw/log/", why="a running log, appended once a day")
    def log(clock=iv.all_of(today, as_paths=True, why="append once a day"),
            prior=iv.own_last_copy(why="yesterday's copy")):
        ran.append(1)
        old = prior if prior is not None else pl.DataFrame(schema={"date": pl.Utf8})
        return pl.concat([old, pl.DataFrame({"date": [day[0]]})]).unique("date")

    assert [(r.dataset, r.is_own) for r in log.reads] == \
        [("config/today/", False), ("raw/log/", True)]

    today(); log()
    assert log.is_current(), "its own copy must not make it stale against itself"
    day[0] = "day2"
    today()
    assert log().height == 2 and len(ran) == 2


def test_a_bare_own_last_copy_on_a_multi_output_stage_is_refused(iv):
    with pytest.raises(DeclError, match="writes several"):
        @iv.step(output={"a": "out/a/", "b": "out/b/"}, why="two tables")
        def two(prior=iv.own_last_copy(why="which one?")):
            return {"a": frame(), "b": frame()}


def test_a_declared_dataset_is_produced_by_naming_it_in_output(iv):


    feed = iv.source("raw/feed/", why="arrives from outside")
    ratings = iv.dataset("processed/ratings/", why="one row per player")
    summary = iv.dataset("processed/summary/", why="one row per season")

    @iv.step(output={"r": ratings, "s": summary}, why="the joint fit")
    def fit(f=iv.all_of(feed, why="the design matrix")):
        return {"r": frame(), "s": frame()}

    assert set(fit.datasets) == {"processed/ratings/", "processed/summary/"}
    assert iv._datasets["processed/ratings/"].why == "one row per player"


def test_a_read_names_the_declaration_rather_than_the_key_it_arrives_under(iv):


    feed = iv.source("raw/feed/", why="arrives from outside")
    ratings = iv.dataset("processed/ratings/", why="one row per player")

    @iv.step(output={"r": ratings}, why="the fit")
    def fit(f=iv.all_of(feed, why="the design matrix")):
        return {"r": frame()}

    assert iv.all_of(ratings, why="x") == iv.all_of(fit["r"], why="x")


def test_one_output_needs_no_declaration_of_its_own(iv):

    feed = iv.source("raw/feed/", why="arrives from outside")

    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(f=iv.all_of(feed, why="the feed")):
        return frame()

    assert mid.dataset == "processed/mid/"
    assert mid.single and not mid.acts_only


def test_a_dataset_is_declared_once_as_one_thing_or_the_other(iv):
    iv.dataset("processed/mid/", why="this pipeline writes it")
    with pytest.raises(DeclError, match="one or the other"):
        iv.source("processed/mid/", why="no it does not")


def test_a_source_cannot_be_named_as_something_a_stage_writes(iv):


    feed = iv.source("raw/feed/", why="arrives from outside")
    with pytest.raises(DeclError, match="declared a source"):
        @iv.data(dataset=feed, why="writing what arrives from outside")
        def clobber():
            return frame()


def test_the_advice_in_the_own_last_copy_error_is_advice_that_works(iv):


    box = iv.dataset("raw/box/", why="the box scores being patched")
    pbp = iv.dataset("raw/pbp/", why="the play-by-play being patched")

    @iv.step(output={"box": box, "pbp": pbp}, why="patch both from the per-game endpoint")
    def patch(was=iv.own_last_copy(box, why="the copy this amends")):
        return {"box": frame(), "pbp": frame()}

    assert patch.reads[0].dataset == "raw/box/"
    assert patch.reads[0].is_own


def test_a_multi_output_stage_cannot_read_its_own_copy_without_a_declaration(iv):


    with pytest.raises(DeclError, match="declared above and named here"):
        @iv.step(output={"box": "raw/box/", "pbp": "raw/pbp/"}, why="patch both")
        def patch(was=iv.own_last_copy(why="the copy this amends")):
            return {"box": frame(), "pbp": frame()}


def test_declaring_the_same_dataset_twice_is_refused(iv):
    iv.dataset("processed/x/", why="a rating per player")
    with pytest.raises(DeclError, match="already declared"):
        iv.dataset("processed/x/", why="something else entirely")


def test_the_second_declaration_does_not_silently_win(iv):


    iv.dataset("processed/x/", why="the original reason")
    with pytest.raises(DeclError):
        iv.dataset("processed/x/", why="a contradicting reason")
    assert iv._datasets["processed/x/"].why == "the original reason"


def test_a_stage_may_not_write_the_path_of_a_declared_dataset_out_again(iv):


    iv.dataset("processed/x/", why="declared up here")
    with pytest.raises(DeclError, match="already declared on its own line"):
        @iv.data(dataset="processed/x/", why="and written out again down here")
        def x():
            return frame()


def test_naming_the_declaration_is_one_declaration_used_twice(iv):


    college = iv.dataset("derived/college/", why="one row per amateur source")

    @iv.data(dataset=college, why="the NCAA block", part={"source": "ncaa"})
    def ncaa():
        return frame()

    @iv.data(dataset=college, why="the G-League block", part={"source": "gleague"})
    def gleague():
        return frame()

    assert ncaa.dataset == gleague.dataset == "derived/college/"
    assert _graph.check(_graph.build(iv))[0] == []


def test_declaring_a_dataset_a_stage_already_wrote_inline_is_refused(iv):

    @iv.data(dataset="processed/x/", why="declared inline, where it is written")
    def x():
        return frame()

    with pytest.raises(DeclError, match="already declared by 'x'"):
        iv.dataset("processed/x/", why="and again up here")


def test_declaring_the_same_source_twice_is_refused(iv):
    iv.source("raw/bios/", why="dropped in by hand once a year")
    with pytest.raises(DeclError, match="already declared a source"):
        iv.source("raw/bios/", why="dropped in by hand once a year")


def test_reading_a_declared_dataset_nothing_writes_is_an_error(iv):

    ghost = iv.dataset("processed/ghost/", why="a table that lost its writer")

    @iv.data(dataset="processed/y/", why="reads it")
    def y(g=iv.all_of(ghost, why="the thing nobody makes")):
        return g

    errors, _ = _graph.check(_graph.build(iv))
    assert any("READ, NOBODY WRITES  processed/ghost/" in e for e in errors)
    assert any("::y" in e for e in errors), "it names WHO reads it"


def test_a_dataset_only_optional_readers_want_is_a_warning_not_an_error(iv):
    ONE_LEAGUE_ONLY = iv.dataset("derived/tracking/", why="one league has this feed")

    @iv.data(dataset="processed/features/", why="the box matrix")
    def features(tracking=iv.all_of(ONE_LEAGUE_ONLY, optional=True,
                                    why="present on one league, absent on the other")):
        return frame()

    errors, warns = _graph.check(_graph.build(iv))
    assert not [e for e in errors if "derived/tracking/" in e]
    assert any("OPTIONAL, NOBODY WRITES" in w and "derived/tracking/" in w for w in warns)


def test_a_dataset_a_required_read_wants_and_nothing_writes_is_still_an_error(iv):
    MISSING = iv.dataset("derived/tracking/", why="nothing writes this")

    @iv.data(dataset="processed/features/", why="the box matrix")
    def features(tracking=iv.all_of(MISSING, why="required, so it must be written")):
        return frame()

    errors, _ = _graph.check(_graph.build(iv))
    assert any("READ, NOBODY WRITES" in e and "derived/tracking/" in e for e in errors)


def test_a_declared_dataset_nothing_reads_or_writes_is_only_a_warning(iv):

    iv.dataset("processed/ghost/", why="a table that lost its writer")

    @iv.data(dataset="processed/y/", why="minds its own business")
    def y():
        return frame()

    errors, warns = _graph.check(_graph.build(iv))
    assert errors == []
    assert any("DECLARED, NOBODY WRITES  processed/ghost/" in w for w in warns)


@pytest.mark.parametrize("write", [
    pytest.param(lambda iv, s: iv.data(dataset=s, why="w"), id="@iv.data names the source"),
    pytest.param(lambda iv, s: iv.data(dataset="raw/bios/", why="w"), id="@iv.data names its path"),
    pytest.param(lambda iv, s: iv.step(output={"a": "processed/a/", "b": s}, why="w"),
                 id="@iv.step, among several"),
    pytest.param(lambda iv, s: iv.step(output={"a": "processed/a/", "b": "raw/bios/"}, why="w"),
                 id="@iv.step by path, among several"),
])
def test_no_stage_may_write_a_source(iv, write):


    bios = iv.source("raw/bios/", why="dropped in by hand once a year")
    with pytest.raises(DeclError, match="declared a source"):
        write(iv, bios)(lambda: frame())


def test_a_source_is_filled_through_iv_writes_which_is_still_allowed(iv):


    bios = iv.source("raw/bios/", why="dropped in by hand once a year")
    with iv.writes("raw/bios/", why="the annual drop") as out:
        frame().write_parquet(out)

    @iv.data(dataset="processed/y/", why="reads the bios")
    def y(b=iv.all_of(bios, why="the bios")):
        return b

    assert y().height == 2


def test_a_stage_writing_several_names_the_shard_on_the_output(iv):


    college = iv.dataset("derived/college/", why="one row per amateur source")
    intl_raw = iv.dataset("derived/intl/", why="the scraped international stats")

    @iv.step(output={"raw": intl_raw, "features": college.shard(source="intl")},
             why="one scrape, two artifacts, and only one is a block of a shared table")
    def intl():
        return {"raw": frame(), "features": frame()}

    assert intl.part_for("derived/college/") == (("source", "intl"),)
    assert not intl.part_for("derived/intl/"), "the whole dataset, not a shard of one"


def test_naming_a_shard_is_not_a_second_declaration(iv):


    college = iv.dataset("derived/college/", why="one row per amateur source")

    @iv.data(dataset=college, why="the NCAA block", part={"source": "ncaa"})
    def ncaa():
        return frame()

    @iv.step(output={"raw": "derived/intl/", "features": college.shard(source="intl")},
             why="the international block, plus the scrape it came from")
    def intl():
        return {"raw": frame(), "features": frame()}

    assert _graph.check(_graph.build(iv))[0] == []


def test_shard_with_no_partition_is_refused(iv):
    college = iv.dataset("derived/college/", why="one row per amateur source")
    with pytest.raises(DeclError, match="names the literal partition"):
        college.shard()


def test_gc_drops_a_partition_the_stage_no_longer_keys_on(iv, monkeypatch):
    @iv.data(dataset="processed/thing/", why="was one shard, now one per season",
             part="season", universe=["2025"])
    def thing(season):
        return frame()

    thing(season="2025")
    d = iv.resolve_out("processed/thing/")
    tmp = _sh.stage("orphan", iv.stage_dir)
    frame().write_parquet(tmp)
    _sh.commit(tmp, d, part=None)
    assert "" in _sh.list_shards(d)

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)
    result = CliRunner().invoke(app, ["gc", "processed/thing/"])
    assert result.exit_code == 0, result.output
    assert "orphaned partition" in result.output
    assert sorted(_sh.list_shards(d)) == ["season=2025"]
    assert thing.is_current(season="2025")
