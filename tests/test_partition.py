"""Per-partition reuse, and the A/A that has to pass before it can be trusted."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import polars as pl
import pytest

from dagio import config as _config
from dagio import static as _static

SEASONS = ["2024", "2025", "2026"]

BUILD = '''
    """Per-season team totals."""
    import argparse
    import polars as pl
    import dagio as dg

    SEASONS = ["2024", "2025", "2026"]

    def build_one(season):
        box = pl.read_parquet(dg.reads(
            "raw/box/{season}.parquet", why="raw box scores for one season",
            part={"season": season}))
        rates = pl.read_parquet(dg.reads(
            "raw/league_rates.parquet", why="league-wide pace; affects every season"))
        return (box.group_by("team", maintain_order=True)
                   .agg(pl.col("pts").sum())
                   .with_columns(pl.lit(season).alias("season"),
                                 pl.lit(rates["pace"][0]).alias("pace")))

    def build():
        dg.for_each(SEASONS, build_one,
                    artifact="processed/team_totals.parquet", key="season",
                    why="per-(season, team) totals")

    ap = argparse.ArgumentParser()
    dg.add_guard_args(ap)
    a = ap.parse_args()
    dg.build_if_needed("processed/team_totals.parquet", build,
                       if_needed=a.if_needed, force=a.force)
'''

CONSUME = '''
    import argparse
    import polars as pl
    import dagio as dg

    def build():
        t = pl.read_parquet(dg.reads("processed/team_totals.parquet",
                                     why="per-(season, team) totals"))
        with dg.writes("processed/summary.parquet",
                       why="one row per season", terminal=True) as p:
            t.group_by("season").agg(pl.col("pts").sum()).write_parquet(p)

    ap = argparse.ArgumentParser()
    dg.add_guard_args(ap)
    a = ap.parse_args()
    dg.build_if_needed("processed/summary.parquet", build,
                       if_needed=a.if_needed, force=a.force)
'''


def box(season: str, extra: int = 0) -> pl.DataFrame:
    n = {"2024": 90, "2025": 95, "2026": 100}[season] + extra
    return pl.DataFrame({"team": ["A", "B"], "pts": [n, n - 5]})


@pytest.fixture
def parts(project):
    (project / "stages" / "totals.py").write_text(textwrap.dedent(BUILD).lstrip())
    (project / "stages" / "summary.py").write_text(textwrap.dedent(CONSUME).lstrip())
    (project / "pyproject.toml").write_text(
        (project / "pyproject.toml").read_text().replace(
            'data_root = "data"',
            'data_root = "data"\nstages = ["stages/totals.py", "stages/summary.py"]'))
    raw = project / "data" / "raw" / "box"
    raw.mkdir(parents=True, exist_ok=True)
    for s in SEASONS:
        box(s).write_parquet(raw / f"{s}.parquet")
    pl.DataFrame({"pace": [98.0]}).write_parquet(
        project / "data" / "raw" / "league_rates.parquet")
    _config.reset()
    _static.reset()
    return project


def run(project, stage: str, *args) -> str:
    env = {**os.environ, "DAGIO_PROJECT": str(project), "NO_COLOR": "1"}
    env.pop("VIRTUAL_ENV", None)
    r = subprocess.run([sys.executable, stage, *args], cwd=project, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def totals(project) -> pl.DataFrame:
    return pl.read_parquet(project / "data" / "processed" / "team_totals.parquet")


def test_cold_build_covers_every_partition(parts):
    out = run(parts, "stages/totals.py", "--if-needed")
    assert "rebuild ( 3)" in out and "reuse   ( 0)" in out
    assert set(totals(parts)["season"].unique()) == set(SEASONS)


def test_one_moved_partition_rebuilds_only_that_one(parts):
    run(parts, "stages/totals.py", "--if-needed")
    box("2026", extra=7).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")

    out = run(parts, "stages/totals.py")
    assert "reuse   ( 2)" in out, out
    assert "rebuild ( 1): 2026" in out, out
    assert totals(parts).filter(pl.col("season") == "2026")["pts"].max() == 107


def test_a_global_input_moves_every_partition(parts):
    """An input without the partition key in its path affects all of them, so it is in
    every partition's key."""
    run(parts, "stages/totals.py", "--if-needed")
    pl.DataFrame({"pace": [101.0]}).write_parquet(
        parts / "data" / "raw" / "league_rates.parquet")

    out = run(parts, "stages/totals.py")
    assert "rebuild ( 3)" in out, out


def test_nothing_moved_reuses_everything(parts):
    run(parts, "stages/totals.py", "--if-needed")
    out = run(parts, "stages/totals.py")
    assert "reuse   ( 3)" in out and "rebuild ( 0)" in out


def test_the_artifacts_own_id_still_works_downstream(parts):
    """A run that reused every partition read nothing. The artifact's inputs come from the
    CODE, not from what the process happened to read — otherwise its input map would be
    empty and it would be permanently current."""
    run(parts, "stages/totals.py", "--if-needed")
    run(parts, "stages/summary.py", "--if-needed")

    import dagio as dg
    from dagio import state as _state
    _state.reset()
    _config.reset()
    os.environ["DAGIO_PROJECT"] = str(parts)
    entry = dg.record_of("processed/team_totals.parquet")
    assert set(entry["in"]) == {"raw/box/{season}.parquet", "raw/league_rates.parquet"}

    box("2026", extra=3).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")
    out = run(parts, "stages/totals.py", "--if-needed") \
        + run(parts, "stages/summary.py", "--if-needed")
    assert "rebuild ( 1): 2026" in out
    assert "input moved: processed/team_totals.parquet" in out


def test_aa_incremental_equals_a_full_rebuild(parts):
    """The gate. Compare UNSORTED — sorting both sides before comparing is the wrong test
    and it passes while the bug is live."""
    run(parts, "stages/totals.py", "--if-needed")
    box("2026", extra=11).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")
    run(parts, "stages/totals.py", "--if-needed")
    incremental = totals(parts)

    run(parts, "stages/totals.py", "--force")
    full = totals(parts)

    assert incremental.equals(full), (
        f"incremental != full rebuild\n{incremental}\n{full}")


def test_force_bypasses_both_layers(parts):
    run(parts, "stages/totals.py", "--if-needed")
    out = run(parts, "stages/totals.py", "--force")
    assert "rebuild ( 3)" in out, "forcing the guard while the cache reuses everything " \
                                  "is a rebuild that rebuilds nothing"


def test_a_partition_producing_no_rows_is_an_error(parts):
    src = parts / "stages" / "totals.py"
    src.write_text(src.read_text().replace(
        "    return (box.group_by",
        "    if season == '2025': return box.head(0)\n    return (box.group_by"))
    env = {**os.environ, "DAGIO_PROJECT": str(parts), "NO_COLOR": "1"}
    env.pop("VIRTUAL_ENV", None)
    r = subprocess.run([sys.executable, "stages/totals.py"], cwd=parts, env=env,
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "produced no rows" in r.stderr


def test_fp_of_redirects_which_partition_governs(parts):
    """A walk-forward artifact: partition C is built from C-1, so C-1's inputs key it.
    Keyed on its own partition, the live one would turn over nightly for nothing."""
    src = parts / "stages" / "totals.py"
    src.write_text(src.read_text().replace(
        'why="per-(season, team) totals")',
        'why="per-(season, team) totals",\n'
        '                fp_of={"2025": "2024", "2026": "2025"})'))
    run(parts, "stages/totals.py", "--if-needed")

    # 2026's own raw file moves. Nothing is keyed on it, so nothing rebuilds.
    box("2026", extra=50).write_parquet(parts / "data" / "raw" / "box" / "2026.parquet")
    assert "reuse   ( 3)" in run(parts, "stages/totals.py")

    # 2025's does. That keys 2026, so 2026 rebuilds — and 2025 keys off 2024, so it does not.
    box("2025", extra=50).write_parquet(parts / "data" / "raw" / "box" / "2025.parquet")
    out = run(parts, "stages/totals.py")
    assert "rebuild ( 1): 2026" in out, out


# ── a partitioned artifact that the pipeline WRITES ───────────────────────────

FETCH = '''
    """Pull the raw box scores, one file per season."""
    import polars as pl
    import dagio as dg

    SEASONS = ["2024", "2025", "2026"]
    BASE = {"2024": 90, "2025": 95, "2026": 100}

    dg.external("vendor/boxscores", why="the upstream feed, one file per season")
    for s in SEASONS:
        with dg.writes("raw/box/{season}.parquet",
                       why="raw box scores for one season", part={"season": s}) as p:
            n = BASE[s] + int(__import__("os").environ.get("BUMP_" + s, 0))
            pl.DataFrame({"team": ["A", "B"], "pts": [n, n - 5]}).write_parquet(p)
'''


@pytest.fixture
def fetched(parts):
    (parts / "stages" / "fetch.py").write_text(textwrap.dedent(FETCH).lstrip())
    (parts / "pyproject.toml").write_text(
        (parts / "pyproject.toml").read_text().replace(
            'stages = ["stages/totals.py"',
            'stages = ["stages/fetch.py", "stages/totals.py"'))
    _config.reset()
    _static.reset()
    return parts


def test_why_stale_on_a_templated_artifact_checks_every_instance(fetched):
    """The template is ONE node in the graph and MANY files on disk. Asking whether it is
    current has to mean asking about every instance, not resolving a path with a
    `{placeholder}` still in it."""
    import dagio as dg
    from dagio import state as _state

    run(fetched, "stages/fetch.py")
    _state.reset(); _config.reset()
    os.environ["DAGIO_PROJECT"] = str(fetched)
    assert dg.why_stale("raw/box/{season}.parquet") is None

    (fetched / "data" / "raw" / "box" / "2025.parquet").unlink()
    _state.reset()
    reason = dg.why_stale("raw/box/{season}.parquet")
    assert reason.startswith("raw/box/2025.parquet: not on disk"), reason


def test_the_collection_and_its_members_agree(fetched):
    """A collection folds each instance's ID, not its fingerprint.

    Fingerprinting the collection while `id_of` reads the stamp for a single file makes
    the two disagree — which surfaced as an outer guard saying "stale" over a partition
    cache saying "nothing to do".
    """
    run(fetched, "stages/fetch.py")
    run(fetched, "stages/totals.py", "--if-needed")

    os.environ["BUMP_2026"] = "13"
    try:
        run(fetched, "stages/fetch.py")
        out = run(fetched, "stages/totals.py", "--if-needed")
    finally:
        del os.environ["BUMP_2026"]

    assert "input moved: raw/box/{season}.parquet" in out
    assert "rebuild ( 1): 2026" in out, out
    assert "reuse   ( 2)" in out, out


def test_editing_a_written_artifact_behind_dagios_back_moves_nothing(fetched):
    """A file the pipeline writes is identified by its STAMP. Editing it without going
    through dagio leaves the stamp — and therefore the id — untouched, and the collection
    agrees rather than reporting a phantom change."""
    import dagio as dg
    from dagio import state as _state

    run(fetched, "stages/fetch.py")
    run(fetched, "stages/totals.py", "--if-needed")

    box("2026", extra=99).write_parquet(fetched / "data" / "raw" / "box" / "2026.parquet")
    _state.reset(); _config.reset()
    os.environ["DAGIO_PROJECT"] = str(fetched)
    assert dg.why_stale("processed/team_totals.parquet") is None
    out = run(fetched, "stages/totals.py", "--if-needed")
    assert "is current — skipping" in out, out
    assert "reuse   ( 3)" in run(fetched, "stages/totals.py"), \
        "and with the guard off, the cache must agree — not disagree with it"
