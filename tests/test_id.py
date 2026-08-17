"""The id, end to end on a three-node pipeline.

    raw/games.parquet ──► processed/team_stats.parquet ──► processed/ratings.parquet

Every test here is a property of the rule, not of an implementation detail. If one of
these fails the package is not doing its job, whatever else passes.
"""
from __future__ import annotations

import polars as pl
import pytest

import dagio as dg
from conftest import fresh_process

GAMES = pl.DataFrame({
    "game_id": [1, 2, 3],
    "team": ["A", "B", "A"],
    "pts": [90, 85, 100],
})


# ── the toy pipeline ──────────────────────────────────────────────────────────

def write_root(project, frame=GAMES, **kw):
    p = project / "data" / "raw" / "games.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(p, **kw)


def build_team_stats():
    games = pl.read_parquet(dg.reads(
        "raw/games.parquet", why="one row per team per game; the only source of points"))
    out = games.group_by("team", maintain_order=True).agg(pl.col("pts").sum())
    with dg.writes("processed/team_stats.parquet",
                   why="season points by team; the rating denominator") as p:
        out.write_parquet(p)


def build_ratings():
    stats = pl.read_parquet(dg.reads(
        "processed/team_stats.parquet", why="season points by team"))
    out = stats.with_columns((pl.col("pts") / 100).alias("rating"))
    with dg.writes("processed/ratings.parquet",
                   why="team ratings the app renders", terminal=True) as p:
        out.write_parquet(p)


def run_pipeline():
    """One forward pass, one process per stage, each guarded on its own outputs."""
    ran = []
    fresh_process()
    ran.append(dg.build_if_needed("processed/team_stats.parquet", build_team_stats,
                                  if_needed=True))
    fresh_process()
    ran.append(dg.build_if_needed("processed/ratings.parquet", build_ratings,
                                  if_needed=True))
    return ran


# ── the tests ─────────────────────────────────────────────────────────────────

def test_cold_build_then_a_no_op_run(project):
    write_root(project)
    assert run_pipeline() == [True, True]
    assert run_pipeline() == [False, False], "a second run with no change must do nothing"


def test_moved_root_propagates_the_whole_way_down(project):
    write_root(project)
    run_pipeline()

    write_root(project, GAMES.vstack(pl.DataFrame(
        {"game_id": [4], "team": ["B"], "pts": [77]})))

    fresh_process()
    assert dg.why_stale("processed/team_stats.parquet").startswith("input moved: raw/games")
    assert run_pipeline() == [True, True], "both stages must rebuild"


def test_identical_rows_different_bytes_move_nothing(project):
    """The property the whole design rests on: a fetcher that rewrites its output every
    run must not invalidate the world."""
    write_root(project, compression="zstd")
    run_pipeline()

    write_root(project, compression="snappy")          # same rows, different bytes
    assert run_pipeline() == [False, False]


def test_touching_mtime_moves_nothing(project):
    write_root(project)
    run_pipeline()
    (project / "data" / "raw" / "games.parquet").touch()
    assert run_pipeline() == [False, False]


def test_reordered_root_rows_move_nothing_by_default(project):
    write_root(project)
    run_pipeline()
    write_root(project, GAMES.reverse())
    assert run_pipeline() == [False, False]


def test_version_bump_moves_every_id_below_it(project, bump):
    write_root(project)
    run_pipeline()

    bump("data", "0.2.0")
    fresh_process()
    assert "version bumped" in dg.why_stale("processed/team_stats.parquet")
    assert run_pipeline() == [True, True]


def test_a_bump_on_an_unselected_axis_moves_nothing(project, bump):
    """Both stages select only the `data` axis, so `model` cannot reach them."""
    write_root(project)
    run_pipeline()

    bump("model", "9.9.9")
    assert run_pipeline() == [False, False]


def test_a_current_check_reads_no_derived_artifact_off_disk(project, monkeypatch):
    """A derived input's id is a dict lookup. Only the roots are fingerprinted."""
    write_root(project)
    run_pipeline()

    from dagio import fingerprint
    seen = []
    real = fingerprint.compute
    monkeypatch.setattr(fingerprint, "compute",
                        lambda p, how="data": (seen.append(str(p)), real(p, how))[1])

    fresh_process()
    assert dg.current("processed/ratings.parquet")
    assert not any("processed/" in s for s in seen), seen


def test_a_raise_inside_writes_stamps_nothing(project):
    write_root(project)

    def broken():
        pl.read_parquet(dg.reads("raw/games.parquet", why="the source"))
        with dg.writes("processed/team_stats.parquet", why="never finished") as p:
            p.write_bytes(b"partial")
            raise RuntimeError("the fit diverged")

    fresh_process()
    with pytest.raises(RuntimeError, match="diverged"):
        broken()

    fresh_process()
    assert dg.why_stale("processed/team_stats.parquet") == "never stamped"


def test_a_rebuilt_upstream_with_identical_output_still_moves_downstream(project):
    """Stated so the trade is on the record rather than discovered later.

    Everything the artifact was built from is inside its id, so an upstream rebuild moves
    it even when the bytes it produced are the same. There is no early cutoff below a
    rebuild; there IS one at the roots, which is where refetches actually happen.
    """
    write_root(project)
    run_pipeline()
    was = dg.record_of("processed/team_stats.parquet")
    before_id, before_fp = was["id"], was["fp"]

    # A version bump that cannot change team_stats' contents at all.
    from dagio import config as _config, state as _state
    _config.configure(versions={**dg.get_config().versions, "data": "0.3.0"})
    _state.reset()

    fresh_process()
    dg.build_if_needed("processed/team_stats.parquet", build_team_stats, if_needed=True)
    after = dg.record_of("processed/team_stats.parquet")
    assert after["fp"] == before_fp, "it produced byte-identical data"
    assert after["id"] != before_id, "yet its id moved, because the metadata is in it"

    fresh_process()
    assert dg.why_stale("processed/ratings.parquet") is not None
