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
    from invalidator import Invalidator

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
        env["INVALIDATOR_FORCE"] = "1"
    else:
        env.pop("INVALIDATOR_FORCE", None)
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

    import json
    st = json.loads((parts / "data" / ".invalidator" / "state.json").read_text())
    entry = st["artifacts"]["processed/team_totals.parquet"]
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
    through invalidator leaves the stamp — and therefore the id — untouched, and the
    collection agrees rather than reporting a phantom change."""
    run(fetched, "stages/fetch.py")
    run(fetched, "stages/totals.py")

    box("2026", extra=99).write_parquet(fetched / "data" / "raw" / "box" / "2026.parquet")
    out = run(fetched, "stages/totals.py")
    assert "reuse   ( 3)" in out, out
