"""The id, end to end on a three-node pipeline, through the `@iv.step` decorator.

    raw/games.parquet ──► processed/team_stats.parquet ──► processed/ratings.parquet

Every test here is a property of the rule, not of an implementation detail. If one of
these fails the package is not doing its job, whatever else passes.
"""
from __future__ import annotations

import polars as pl
import pytest

from conftest import write_stage
from invalidator import Invalidator

GAMES = pl.DataFrame({
    "game_id": [1, 2, 3],
    "team": ["A", "B", "A"],
    "pts": [90, 85, 100],
})

STATS = '''
    import polars as pl
    from conftest_pipeline import iv

    @iv.step("processed/team_stats.parquet",
             why="season points by team; the rating denominator")
    def build(out):
        games = pl.read_parquet(iv.reads(
            "raw/games.parquet", why="one row per team per game"))
        games.group_by("team", maintain_order=True).agg(
            pl.col("pts").sum()).write_parquet(out)
'''

RATINGS = '''
    import polars as pl
    from conftest_pipeline import iv

    @iv.step("processed/ratings.parquet",
             why="team ratings the app renders", terminal=True)
    def build(out):
        stats = pl.read_parquet(iv.reads(
            "processed/team_stats.parquet", why="season points by team"))
        stats.with_columns((pl.col("pts") / 100).alias("rating")).write_parquet(out)
'''


@pytest.fixture
def pipe(project, iv):
    """The two stage files on disk, plus the in-process functions that mirror them.

    The files exist because the static scan needs real source to read; the callables are
    what the tests invoke, so a test is one process rather than four.
    """
    write_stage(project, "stages/build_stats.py", STATS)
    write_stage(project, "stages/build_ratings.py", RATINGS)
    write_root(project)

    @iv.step("processed/team_stats.parquet",
             why="season points by team; the rating denominator")
    def stats(out):
        games = pl.read_parquet(iv.reads(
            "raw/games.parquet", why="one row per team per game"))
        games.group_by("team", maintain_order=True).agg(
            pl.col("pts").sum()).write_parquet(out)

    @iv.step("processed/ratings.parquet",
             why="team ratings the app renders", terminal=True)
    def ratings(out):
        stats_df = pl.read_parquet(iv.reads(
            "processed/team_stats.parquet", why="season points by team"))
        stats_df.with_columns((pl.col("pts") / 100).alias("rating")).write_parquet(out)

    def run():
        iv.state.reset()
        return [stats(), ratings()]

    return run


def write_root(project, frame=GAMES, **kw):
    p = project / "data" / "raw" / "games.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(p, **kw)


# ── the tests ─────────────────────────────────────────────────────────────────

def test_cold_build_then_a_no_op_run(pipe):
    assert pipe() == [True, True]
    assert pipe() == [False, False], "a second run with no change must do nothing"


def test_moved_root_propagates_the_whole_way_down(project, iv, pipe):
    pipe()
    write_root(project, GAMES.vstack(pl.DataFrame(
        {"game_id": [4], "team": ["B"], "pts": [77]})))

    iv.state.reset()
    assert iv.why_stale("processed/team_stats.parquet").startswith(
        "input moved: raw/games.parquet")
    assert pipe() == [True, True], "both stages must rebuild"


def test_identical_rows_different_bytes_move_nothing(project, pipe):
    """The property the whole design rests on: a fetcher that rewrites its output every
    run must not invalidate the world."""
    write_root(project, compression="zstd")
    pipe()
    write_root(project, compression="snappy")          # same rows, different bytes
    assert pipe() == [False, False]


def test_touching_mtime_moves_nothing(project, pipe):
    pipe()
    (project / "data" / "raw" / "games.parquet").touch()
    assert pipe() == [False, False]


def test_reordered_root_rows_move_nothing_by_default(project, pipe):
    pipe()
    write_root(project, GAMES.reverse())
    assert pipe() == [False, False]


def test_data_version_bump_rebuilds_everything(project, pipe, iv):
    pipe()
    bumped = Invalidator(data_root=iv.data_root, data_version="v2",
                         source_dirs=["stages"], project_root=iv.project_root)
    assert bumped.why_stale("processed/team_stats.parquet") == \
        "data_version bumped: v1 -> v2"
    assert bumped.why_stale("processed/ratings.parquet") == \
        "data_version bumped: v1 -> v2"


def test_a_current_check_reads_no_derived_artifact_off_disk(pipe, monkeypatch, iv):
    """A derived input's id is a dict lookup. Only the roots are fingerprinted."""
    pipe()
    from invalidator import fingerprint
    seen = []
    real = fingerprint.compute
    monkeypatch.setattr(fingerprint, "compute",
                        lambda p, how="data": (seen.append(str(p)), real(p, how))[1])
    iv.state.reset()
    assert iv.is_current("processed/ratings.parquet")
    assert not any("processed/" in s for s in seen), seen


def test_a_raise_inside_a_step_stamps_nothing(project, iv):
    write_root(project)
    write_stage(project, "stages/broken.py", '''
        import polars as pl
        from conftest_pipeline import iv

        @iv.step("processed/team_stats.parquet", why="never finished")
        def build(out):
            pl.read_parquet(iv.reads("raw/games.parquet", why="the source"))
            out.write_bytes(b"partial")
            raise RuntimeError("the fit diverged")
    ''')

    @iv.step("processed/team_stats.parquet", why="never finished")
    def build(out):
        pl.read_parquet(iv.reads("raw/games.parquet", why="the source"))
        out.write_bytes(b"partial")
        raise RuntimeError("the fit diverged")

    with pytest.raises(RuntimeError, match="diverged"):
        build()
    iv.state.reset()
    assert iv.why_stale("processed/team_stats.parquet") == "never stamped"
    assert (project / "data" / "processed" / "team_stats.parquet").exists(), \
        "the partial file is on disk — which is exactly why the stamp must not be"


def test_a_step_returns_whether_it_ran(pipe):
    assert pipe() == [True, True]
    assert pipe() == [False, False]


def test_force_overrides_the_guard(project, iv, pipe):
    pipe()
    iv.force = True
    assert pipe() == [True, True]


def test_two_invalidators_are_independent(project, iv):
    """No globals: a second pipeline over the same tree has its own state and version."""
    other = Invalidator(data_root=project / "data2", data_version="other-1",
                        source_dirs=["stages"], project_root=project)
    assert other.data_root != iv.data_root
    assert other.state.path != iv.state.path
    assert iv.data_version == "v1" and other.data_version == "other-1"


def test_data_version_is_required():
    from invalidator import ConfigError
    with pytest.raises(ConfigError, match="data_version is required"):
        Invalidator(data_root="/tmp/x", data_version="")
