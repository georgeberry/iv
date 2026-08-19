"""Per-partition reuse, and the A/A that has to pass before it can be trusted."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import polars as pl
import pytest

SEASONS = ["2024", "2025", "2026"]

PIPELINE = '''
    from iv import Invalidator

    iv = Invalidator(
        data_root="data",
        data_version="v1",
        source_dirs=["stages"],
        stages=["stages/fetch.py", "stages/totals.py", "stages/summary.py"],
    )
'''

TOTALS = '''
    """Per-season team totals."""
    import polars as pl
    from pipeline import iv

    SEASONS = ["2024", "2025", "2026"]

    def build_one(season):
        box = pl.read_parquet(iv.reads(
            "raw/box/{season}.parquet", why="raw box scores for one season",
            part={"season": season}))
        rates = pl.read_parquet(iv.reads(
            "raw/league_rates.parquet", why="league-wide pace; affects every season"))
        return (box.group_by("team", maintain_order=True)
                   .agg(pl.col("pts").sum())
                   .with_columns(pl.lit(season).alias("season"),
                                 pl.lit(rates["pace"][0]).alias("pace")))

    iv.for_each(SEASONS, build_one,
                output="processed/team_totals.parquet", key="season",
                why="per-(season, team) totals")
'''

SUMMARY = '''
    import polars as pl
    from pipeline import iv

    @iv.step("processed/summary.parquet", why="one row per season", terminal=True)
    def build(out):
        t = pl.read_parquet(iv.reads("processed/team_totals.parquet",
                                     why="per-(season, team) totals"))
        t.group_by("season").agg(pl.col("pts").sum()).write_parquet(out)

    build()
'''

FETCH = '''
    """Pull the raw box scores, one file per season."""
    import os
    import polars as pl
    from pipeline import iv

    SEASONS = ["2024", "2025", "2026"]
    BASE = {"2024": 90, "2025": 95, "2026": 100}

    iv.external("vendor/boxscores", why="the upstream feed, one file per season")
    for s in SEASONS:
        with iv.writes("raw/box/{season}.parquet",
                       why="raw box scores for one season", part={"season": s}) as p:
            n = BASE[s] + int(os.environ.get("BUMP_" + s, 0))
            pl.DataFrame({"team": ["A", "B"], "pts": [n, n - 5]}).write_parquet(p)
'''


def box(season: str, extra: int = 0) -> pl.DataFrame:
    n = {"2024": 90, "2025": 95, "2026": 100}[season] + extra
    return pl.DataFrame({"team": ["A", "B"], "pts": [n, n - 5]})


@pytest.fixture
def parts(project):
    (project / "pipeline.py").write_text(textwrap.dedent(PIPELINE).lstrip())
    for rel, body in (("stages/totals.py", TOTALS), ("stages/summary.py", SUMMARY)):
        (project / rel).write_text(textwrap.dedent(body).lstrip())
    raw = project / "data" / "raw" / "box"
    raw.mkdir(parents=True, exist_ok=True)
    for s in SEASONS:
        box(s).write_parquet(raw / f"{s}.parquet")
    pl.DataFrame({"pace": [98.0]}).write_parquet(
        project / "data" / "raw" / "league_rates.parquet")
    return project


@pytest.fixture
def fetched(parts):
    """The same project, but the per-season files are WRITTEN by a stage rather than
    dropped on disk — so each one is a stamped artifact rather than a root."""
    (parts / "stages" / "fetch.py").write_text(textwrap.dedent(FETCH).lstrip())
    return parts


def run(project, stage: str, *, force: bool = False, check: bool = True, **env_extra):
    env = {**os.environ, "PYTHONPATH": str(project), "NO_COLOR": "1",
           "PYTHONDONTWRITEBYTECODE": "1", **env_extra}
    env.pop("VIRTUAL_ENV", None)
    if force:
        env["IV_FORCE"] = "1"
    else:
        env.pop("IV_FORCE", None)
    r = subprocess.run([sys.executable, stage], cwd=project, env=env,
                       capture_output=True, text=True)
    if check:
        assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout + r.stderr


def totals(project) -> pl.DataFrame:
    return pl.read_parquet(project / "data" / "processed" / "team_totals.parquet")


# ── planning ──────────────────────────────────────────────────────────────────

def test_cold_build_covers_every_partition(parts):
    out = run(parts, "stages/totals.py")
    assert "rebuild ( 3)" in out and "reuse   ( 0)" in out
    assert set(totals(parts)["season"].unique()) == set(SEASONS)


def test_one_moved_partition_rebuilds_only_that_one(parts):
    run(parts, "stages/totals.py")
    box("2026", extra=7).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")

    out = run(parts, "stages/totals.py")
    assert "reuse   ( 2)" in out, out
    assert "rebuild ( 1): 2026" in out, out
    assert totals(parts).filter(pl.col("season") == "2026")["pts"].max() == 107


def test_a_global_input_moves_every_partition(parts):
    """An input without the partition key in its path affects all of them, so it is in
    every partition's key."""
    run(parts, "stages/totals.py")
    pl.DataFrame({"pace": [101.0]}).write_parquet(
        parts / "data" / "raw" / "league_rates.parquet")
    assert "rebuild ( 3)" in run(parts, "stages/totals.py")


def test_nothing_moved_reuses_everything(parts):
    run(parts, "stages/totals.py")
    out = run(parts, "stages/totals.py")
    assert "reuse   ( 3)" in out and "rebuild ( 0)" in out


def test_the_artifacts_own_id_still_works_downstream(parts):
    """A run that reused every partition read nothing. The artifact's inputs come from the
    CODE, not from what the process happened to read — otherwise its input map would be
    empty and it would be permanently current."""
    run(parts, "stages/totals.py")
    run(parts, "stages/summary.py")

    from iv.state import read_records
    st = read_records(parts / "data" / ".iv" / "state")
    entry = st["processed/team_totals.parquet"]
    assert set(entry["in"]) == {"raw/box/{season}.parquet", "raw/league_rates.parquet"}

    box("2026", extra=3).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")
    out = run(parts, "stages/totals.py") + run(parts, "stages/summary.py")
    assert "rebuild ( 1): 2026" in out
    assert "input moved: processed/team_totals.parquet" in out


# ── the gate ──────────────────────────────────────────────────────────────────

def test_aa_incremental_equals_a_full_rebuild(parts):
    """The gate, and it has already caught one real bug: an incremental build that
    concatenated rebuilt-then-reused produced a different ROW ORDER from a full one.

    Compare UNSORTED — sorting both sides before comparing is the wrong test for a file
    whose order is itself an input, and it passes while the bug is live.
    """
    run(parts, "stages/totals.py")
    box("2026", extra=11).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")
    run(parts, "stages/totals.py")
    incremental = totals(parts)

    run(parts, "stages/totals.py", force=True)
    full = totals(parts)

    assert incremental.equals(full), f"incremental != full\n{incremental}\n{full}"


def test_force_bypasses_both_layers(parts):
    run(parts, "stages/totals.py")
    out = run(parts, "stages/totals.py", force=True)
    assert "rebuild ( 3)" in out, "forcing the guard while the cache reuses everything " \
                                  "is a rebuild that rebuilds nothing"


def test_a_partition_producing_no_rows_is_an_error(parts):
    src = parts / "stages" / "totals.py"
    src.write_text(src.read_text().replace(
        "    return (box.group_by",
        "    if season == '2025': return box.head(0)\n    return (box.group_by"))
    out = run(parts, "stages/totals.py", check=False)
    assert "produced no rows" in out


def test_fp_of_redirects_which_partition_governs(parts):
    """A walk-forward artifact: partition C is built from C-1, so C-1's inputs key it.
    Keyed on its own partition, the live one would turn over nightly for nothing."""
    src = parts / "stages" / "totals.py"
    src.write_text(src.read_text().replace(
        'why="per-(season, team) totals")',
        'why="per-(season, team) totals",\n'
        '            fp_of={"2025": "2024", "2026": "2025"})'))
    run(parts, "stages/totals.py")

    # 2026's own raw file moves. Nothing is keyed on it, so nothing rebuilds.
    box("2026", extra=50).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")
    assert "reuse   ( 3)" in run(parts, "stages/totals.py")

    # 2025's does. That keys 2026 — and 2025 keys off 2024, so 2025 stays put.
    box("2025", extra=50).write_parquet(parts / "data" / "raw" / "box" / "2025.parquet")
    out = run(parts, "stages/totals.py")
    assert "rebuild ( 1): 2026" in out, out


# ── a partitioned feed the pipeline WRITES ────────────────────────────────────

def test_why_stale_on_a_templated_artifact_checks_every_instance(fetched):
    """The template is ONE node in the graph and MANY files on disk. Asking whether it is
    current has to mean asking about every instance, not resolving a path with a
    `{placeholder}` still in it."""
    run(fetched, "stages/fetch.py")
    (fetched / "data" / "raw" / "box" / "2025.parquet").unlink()

    sys.path.insert(0, str(fetched))
    try:
        import importlib
        pipeline = importlib.import_module("pipeline")
        importlib.reload(pipeline)
        reason = pipeline.iv.why_stale("raw/box/{season}.parquet")
    finally:
        sys.path.remove(str(fetched))
        sys.modules.pop("pipeline", None)
    assert reason.startswith("raw/box/2025.parquet: not on disk"), reason


def test_the_collection_and_its_members_agree(fetched):
    """A collection folds each instance's ID, not its fingerprint.

    Fingerprinting the collection while `id_of` reads the stamp for a single file makes
    the two disagree — which surfaced as an outer guard saying "stale" over a partition
    cache saying "nothing to do".
    """
    run(fetched, "stages/fetch.py")
    run(fetched, "stages/totals.py")

    run(fetched, "stages/fetch.py", BUMP_2026="13")
    out = run(fetched, "stages/totals.py")

    assert "rebuild ( 1): 2026" in out, out
    assert "reuse   ( 2)" in out, out


def test_editing_a_written_artifact_behind_our_back_moves_nothing(fetched):
    """A file the pipeline writes is identified by its STAMP. Editing it without going
    through iv leaves the stamp — and therefore the id — untouched, and the
    collection agrees rather than reporting a phantom change."""
    run(fetched, "stages/fetch.py")
    run(fetched, "stages/totals.py")

    box("2026", extra=99).write_parquet(fetched / "data" / "raw" / "box" / "2026.parquet")
    out = run(fetched, "stages/totals.py")
    assert "reuse   ( 3)" in out, out


# ── one artifact per dataset, each season-partitioned ─────────────────────────

MULTI = '''
    import polars as pl
    from pipeline import iv

    DATASETS = ["player_box", "schedule"]
    SEASONS = ["2024", "2025"]

    def build_one(season, dataset):
        box = pl.read_parquet(iv.reads(
            "raw/{dataset}/{dataset}_{season}.parquet",
            why="one season of one raw feed", fp="rows",
            part={"dataset": dataset, "season": season}))
        return box.with_columns(pl.lit(season).alias("season"))

    for ds in DATASETS:
        iv.for_each(SEASONS, lambda s, ds=ds: build_one(s, ds),
                    output="processed/panel/{dataset}.parquet", key="season",
                    part={"dataset": ds},
                    why="a parsed panel — one file per dataset, every season")
'''


@pytest.fixture
def multi(project):
    (project / "pipeline.py").write_text(textwrap.dedent(PIPELINE).lstrip())
    (project / "stages" / "panels.py").write_text(textwrap.dedent(MULTI).lstrip())
    for ds in ("player_box", "schedule"):
        d = project / "data" / "raw" / ds
        d.mkdir(parents=True, exist_ok=True)
        for s in ("2024", "2025"):
            pl.DataFrame({"a": [int(s)]}).write_parquet(d / f"{ds}_{s}.parquet")
    return project


def test_a_templated_output_is_one_artifact_per_dataset(multi):
    """The path differs by a runtime value AND the artifact is season-partitioned. The
    template stays the literal the scan reads; the rendered path is the file on disk."""
    out = run(multi, "stages/panels.py")
    assert out.count("rebuild ( 2)") == 2, out
    for ds in ("player_box", "schedule"):
        assert (multi / "data" / "processed" / "panel" / f"{ds}.parquet").exists()

    assert "reuse   ( 2)" in run(multi, "stages/panels.py")


def test_only_the_dataset_whose_raw_moved_rebuilds(multi):
    run(multi, "stages/panels.py")
    # A ROW, not a value: that feed is read with fp="rows", so a same-count edit is
    # invisible by design — the coarse-strategy hazard, demonstrated on our own test.
    pl.DataFrame({"a": [2025, 999]}).write_parquet(
        multi / "data" / "raw" / "player_box" / "player_box_2025.parquet")

    out = run(multi, "stages/panels.py")
    lines = [l for l in out.splitlines() if "rebuild" in l or "partitions" in l]
    assert any("rebuild ( 1): 2025" in l for l in lines), lines
    assert any("rebuild ( 0)" in l for l in lines), \
        "the other dataset's panel must not move"


# ── branches ──────────────────────────────────────────────────────────────────

BRANCHED = '''
    """One builder, two path shapes — the tree's own league is flat, a sub-league nests."""
    import polars as pl
    from pipeline import iv

    SEASONS = ["2024", "2025"]

    def build_one(season, league, dataset):
        if league == "own":
            p = iv.reads("raw/{dataset}/{dataset}_{season}.parquet",
                         why="one season of the tree's own raw feed",
                         part={"dataset": dataset, "season": season})
        else:
            p = iv.reads("raw/{league}/{dataset}/{dataset}_{season}.parquet",
                         why="one season of a nested sub-league feed",
                         part={"league": league, "dataset": dataset, "season": season})
        return pl.read_parquet(p).with_columns(pl.lit(season).alias("season"))

    iv.for_each(SEASONS, lambda s: build_one(s, "own", "box"),
                output="processed/panel/{dataset}.parquet", key="season",
                part={"dataset": "box"}, why="the tree's own panel")

    iv.for_each(SEASONS, lambda s: build_one(s, "sub", "box"),
                output="processed/panel/{league}/{dataset}.parquet", key="season",
                part={"league": "sub", "dataset": "box"}, why="a sub-league panel")
'''


@pytest.fixture
def branched(project):
    (project / "pipeline.py").write_text(textwrap.dedent(PIPELINE).lstrip())
    (project / "stages" / "totals.py").write_text(textwrap.dedent(BRANCHED).lstrip())
    for rel in ("raw/box", "raw/sub/box"):
        (project / "data" / rel).mkdir(parents=True)
    for s in ("2024", "2025"):
        for rel in ("raw/box", "raw/sub/box"):
            pl.DataFrame({"pts": [1]}).write_parquet(
                project / "data" / rel / f"box_{s}.parquet")
    return project


def test_a_branch_this_partition_cannot_render_is_not_its_input(branched):
    """The static scan is a union over branches. The flat panel never opens the nested
    feed, and cannot even name it — `{league}` is a distinction it never made — so
    demanding a value for it would make the panel unbuildable."""
    out = run(branched, "stages/totals.py")
    assert "rebuild ( 2)" in out, out

    from iv.state import read_records
    st = read_records(branched / "data" / ".iv" / "state")
    # FILLED where this cache knows the value, free where it does not: the dataset is
    # fixed, the season is the growing set.
    assert set(st["processed/panel/box.parquet"]["in"]) == {
        "raw/box/box_{season}.parquet"}


def test_the_sub_league_panel_does_not_key_on_the_parents_feed(branched):
    """Both templates RENDER for the sub-league — it knows a dataset and a season, which is
    all the flat one needs. Keyed on both, the parent's nightly feed would rebuild a panel
    it never read."""
    run(branched, "stages/totals.py")
    pl.DataFrame({"pts": [99]}).write_parquet(
        branched / "data" / "raw" / "box" / "box_2025.parquet")

    out = run(branched, "stages/totals.py")
    assert out.count("rebuild ( 1): 2025") == 1, out
    assert out.count("reuse   ( 2)") == 1, out


# ── walk-forward ──────────────────────────────────────────────────────────────

WALK = '''
    import os
    import polars as pl
    from pipeline import iv

    COHORTS = ["2024", "2025", "2026"]
    PREV = {"2024": "2023", "2025": "2024", "2026": "2025"}

    # One file, many seasons — the shape `box_features` has. Declared so the cache can
    # scope it; read inside `building()` so a cohort cannot see past its bound.
    iv.reads("processed/features.parquet", why="per-season features, all seasons")

    # Cohort C is fit on data through C-1, so what can move it is stated per cohort.
    EXTRA = {c: os.environ.get("SRC_" + c, "v0") for c in COHORTS}

    cache = iv.partitions("processed/cohorts.parquet", "season",
                          why="one row per cohort")
    reuse, rebuild = cache.plan(COHORTS, extra=EXTRA, upto=PREV)
    cache.report(reuse, rebuild)

    rows = []
    for c in rebuild:
        with cache.building(c) as bound:
            seen = iv.frame("processed/features.parquet",
                            why="per-season features, all seasons")
            rows.append({"season": c, "v": EXTRA[c], "n": seen.height, "bound": bound})
    fresh = pl.DataFrame(rows) if rows else None
    old = cache.reused_rows(reuse) if reuse else None
    out = pl.concat([f for f in (old, fresh) if f is not None], how="vertical_relaxed") \\
        .sort("season")
    cache.commit(out, rebuild, reuse)
'''


@pytest.fixture
def walk(project):
    (project / "pipeline.py").write_text(textwrap.dedent(PIPELINE).lstrip())
    (project / "stages" / "totals.py").write_text(textwrap.dedent(WALK).lstrip())
    raw = project / "data" / "raw" / "box"
    raw.mkdir(parents=True, exist_ok=True)
    for s in SEASONS:
        box(s).write_parquet(raw / f"{s}.parquet")
    pl.DataFrame({"pace": [98.0]}).write_parquet(
        project / "data" / "raw" / "league_rates.parquet")
    return project


def _features(project, rows):
    """A period-partitioned input: one file, many seasons, like `box_features`."""
    p = project / "data" / "processed" / "features.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(p)


def test_the_live_period_cannot_move_work_fit_on_completed_ones(walk):
    """The whole point. A scoped input is period-partitioned, so cohort C depends on its
    rows through C-1 — not on a file that happens to also contain the live season.

    Keyed on the whole file, every completed cohort refits the moment tonight's games
    land: 5m24s of frozen fits in the pipeline this was built for."""
    _features(walk, {"season": ["2023", "2024", "2025"], "x": [1, 2, 3]})
    run(walk, "stages/totals.py")

    # The LIVE period arrives. No cohort is bounded at or past it, so nothing turns over.
    _features(walk, {"season": ["2023", "2024", "2025", "2026"], "x": [1, 2, 3, 99]})
    out = run(walk, "stages/totals.py")
    assert "reuse   ( 3)" in out and "rebuild ( 0)" in out, out


def test_a_correction_to_an_old_period_DOES_propagate(walk):
    """The half `exempt` threw away. Dropping the input term entirely cannot tell the live
    period moving from an old one being corrected, so it traded a false rebuild for a
    missed one."""
    _features(walk, {"season": ["2023", "2024", "2025"], "x": [1, 2, 3]})
    run(walk, "stages/totals.py")

    # 2024 is corrected. Cohorts bounded at 2024 or later must rebuild; 2024 (bound 2023)
    # must not.
    _features(walk, {"season": ["2023", "2024", "2025"], "x": [1, 77, 3]})
    out = run(walk, "stages/totals.py")
    assert "rebuild ( 2)" in out, out
    assert "reuse   ( 1): 2024" in out, out


def test_the_source_a_cohort_was_fit_on_is_what_rebuilds_it(walk):
    _features(walk, {"season": ["2023", "2024", "2025"], "x": [1, 2, 3]})
    run(walk, "stages/totals.py")
    out = run(walk, "stages/totals.py", SRC_2025="v1")
    assert "rebuild ( 1): 2025" in out, out
    assert "reuse   ( 2)" in out, out

    got = pl.read_parquet(walk / "data" / "processed" / "cohorts.parquet")
    assert dict(zip(got["season"], got["v"])) == {"2024": "v0", "2025": "v1", "2026": "v0"}


def test_a_glob_is_a_prefilter_not_the_answer(project):
    """`*` per field is greedy and blind to a repeated field.

    `raw/{league}/{dataset}/{dataset}_{season}.parquet` globs as `raw/*/*/*_*`, which
    swept up `raw/rosters/history/bak_roster_2026_2026-08-02.parquet` — a backup file in
    a tree the template does not describe — and reported it forever as an unstamped
    partition of a feed it has nothing to do with. The pattern decides membership.
    """
    import polars as pl
    from iv import Invalidator

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)
    for rel in ("raw/nba/player_box/player_box_2002.parquet",
                "raw/rosters/history/bak_roster_2026_2026-08-02.parquet"):
        p = project / "data" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"x": [1]}).write_parquet(p)

    parts = iv.state.instances_of("raw/{league}/{dataset}/{dataset}_{season}.parquet")
    assert parts == ["raw/nba/player_box/player_box_2002.parquet"], parts


def test_an_unstamped_partition_is_a_root_not_a_stale_output(project):
    """A partition on disk that nothing here has ever stamped is a ROOT.

    wvorp's refresh fetches ONE season, so 90 of 93 partitions of
    `raw/{dataset}/{dataset}_{season}` were written once in a backfill and no future run
    will ever stamp them. Their identity is their bytes — which is what dependants
    already fold, as `coll:…` — so nothing downstream is stale and the nightly report was
    something nobody could act on.

    A partition that HAS a record is still checked, which is what keeps "did the fetcher
    actually run?" answerable for the live season.
    """
    import polars as pl
    from iv import Invalidator

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)
    (project / "data" / "raw").mkdir(parents=True, exist_ok=True)
    for s in ("2024", "2025"):
        pl.DataFrame({"x": [1]}).write_parquet(project / "data" / "raw" / f"box_{s}.parquet")

    # Nothing stamped: every partition is a root, so the feed is not stale.
    assert iv.why_stale("raw/box_{season}.parquet") is None

    # Stamp the live one. It IS an output now, and stays subject to the check — here
    # via a version bump, the same way a real refresh would notice one.
    with iv.writes("raw/box_{season}.parquet", why="one season", part={"season": "2025"},
                   fp="bytes") as p:
        pl.DataFrame({"x": [1]}).write_parquet(p)
    assert iv.why_stale("raw/box_{season}.parquet") is None

    bumped = Invalidator(data_root=project / "data", data_version="v2",
                         source_dirs=["stages"], project_root=project)
    reason = bumped.why_stale("raw/box_{season}.parquet")
    assert reason is not None and "box_2025" in reason, reason
    assert "box_2024" not in reason, "the unstamped archive partition stays quiet"


def test_a_stage_building_a_period_cannot_see_the_future(walk):
    """The invariant, enforced rather than intended.

    Inside `building(C)` a period-scoped input is CUT to the bound, so a stage physically
    cannot read forward. Filtered and not asserted, for the reason `slices` filters: a
    rule the caller has to remember is a rule that gets forgotten.

    Not theoretical. wvorp's `box_features` chose its clustering palette by counting nulls
    over ALL seasons, so one null in a live-season row changed the centroids — and
    therefore `cluster_id` and every `z_cl_*` — for rows six years earlier, inside the
    block its own docs certified as safe. Under this scope the live rows are not there to
    be counted.
    """
    _features(walk, {"season": ["2023", "2024", "2025", "2026"], "x": [1, 2, 3, 4]})
    run(walk, "stages/totals.py")

    got = pl.read_parquet(walk / "data" / "processed" / "cohorts.parquet").sort("season")
    # Cohort 2024 is bounded at 2023 and must see ONE row; 2025 -> two; 2026 -> three.
    # Four rows exist. No cohort sees the fourth.
    assert dict(zip(got["season"], got["n"])) == {"2024": 1, "2025": 2, "2026": 3}, got
    assert dict(zip(got["season"], got["bound"])) == {
        "2024": "2023", "2025": "2024", "2026": "2025"}, got


def test_building_a_partition_outside_the_scope_is_refused(walk):
    """No opt-in. A partition built outside `building()` had unbounded reads, so its output
    is not trustworthy and `commit` refuses it rather than stamping a lie."""
    _features(walk, {"season": ["2023", "2024", "2025"], "x": [1, 2, 3]})
    (walk / "stages" / "totals.py").write_text(
        (walk / "stages" / "totals.py").read_text()
        .replace("with cache.building(c) as bound:", "if (bound := None) is None:"))
    out = run(walk, "stages/totals.py", check=False)
    assert "were built without" in out, out


def test_the_artifact_record_agrees_with_the_partition_keys(walk):
    """Two notions of "current" for one artifact is one too many.

    `_key` scopes a periodised input to its partition's bound. Stamping the WHOLE-FILE id
    at the artifact level made `iv status` report "input moved" about a change no partition
    would rebuild for — so the stage said nothing to do, the report said stale, and neither
    could resolve the other. wvorp's `eval_prospective_*` sat in exactly that state against
    `rookie_prior`.
    """
    _features(walk, {"season": ["2023", "2024", "2025"], "x": [1, 2, 3]})
    run(walk, "stages/totals.py")

    # A period NO partition is bounded at or past. Cohort bounds are 2023/2024/2025, so a
    # 2026 row reaches none of them — and the artifact record must say so too.
    _features(walk, {"season": ["2023", "2024", "2025", "2026"], "x": [1, 2, 3, 9]})
    out = run(walk, "stages/totals.py")
    assert "reuse   ( 3)" in out, out

    from iv import Invalidator
    iv = Invalidator(data_root=walk / "data", data_version="v1",
                     source_dirs=["stages"], project_root=walk)
    assert iv.why_stale("processed/cohorts.parquet") is None, \
        iv.why_stale("processed/cohorts.parquet")
