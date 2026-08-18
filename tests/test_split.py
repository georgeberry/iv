"""The read/write split, and a state file that lives outside the data tree.

Both exist so a local run can rebuild three stages without clobbering the tree everything
else reads — the reason wvorp has DATA_BASE/DATA_OUT_BASE — and so a shadow run over
someone else's data can stamp somewhere harmless.
"""
from __future__ import annotations

import polars as pl
import pytest

from conftest import write_stage
from invalidator import Invalidator
from invalidator.state import read_records

FRAME = pl.DataFrame({"a": [1, 2, 3]})


@pytest.fixture
def split(project):
    shared, scratch = project / "shared", project / "scratch"
    (shared / "raw").mkdir(parents=True)
    FRAME.write_parquet(shared / "raw" / "src.parquet")
    write_stage(project, "stages/s.py", '''
        from mypipe import iv
        @iv.step("processed/out.parquet", why="the output", terminal=True)
        def build(out):
            iv.reads("raw/src.parquet", why="the source")
    ''')
    return Invalidator(data_root=shared, out_root=scratch, data_version="v1",
                       source_dirs=["stages"], project_root=project)


def test_writes_land_in_out_root_and_reads_fall_back(split, project):
    assert split.resolve("raw/src.parquet") == project / "shared" / "raw" / "src.parquet"
    assert split.resolve_out("processed/out.parquet") == \
        project / "scratch" / "processed" / "out.parquet"

    with split.writes("processed/out.parquet", why="the output", terminal=True) as p:
        FRAME.write_parquet(p)
    assert (project / "scratch" / "processed" / "out.parquet").exists()
    assert not (project / "shared" / "processed").exists(), "the shared tree is untouched"


def test_the_out_root_shadows_the_shared_one_for_reads(split, project):
    """An overlay: a partial local rebuild reads its own outputs and falls back to the
    shared tree for everything it did not touch."""
    assert split.resolve("raw/src.parquet").parent.parent.name == "shared"

    local = project / "scratch" / "raw" / "src.parquet"
    local.parent.mkdir(parents=True)
    FRAME.with_columns(pl.col("a") * 10).write_parquet(local)
    assert split.resolve("raw/src.parquet") == local


def test_the_state_lives_with_the_writes_not_the_reads(split, project):
    with split.writes("processed/out.parquet", why="the output", terminal=True) as p:
        FRAME.write_parquet(p)
    assert read_records(project / "scratch" / ".invalidator" / "state")
    assert not (project / "shared" / ".invalidator").exists()


def test_state_path_puts_the_stamps_anywhere(project, tmp_path):
    shared = project / "shared"
    (shared / "raw").mkdir(parents=True)
    FRAME.write_parquet(shared / "raw" / "src.parquet")
    elsewhere = tmp_path / "somewhere" / "shadow"

    iv = Invalidator(data_root=shared, out_root=project / "scratch",
                     state_path=elsewhere, data_version="v1",
                     source_dirs=["stages"], project_root=project)
    with iv.writes("processed/out.parquet", why="the output", terminal=True) as p:
        FRAME.write_parquet(p)
    assert read_records(elsewhere)
    assert not (project / "scratch" / ".invalidator").exists()


def test_a_collection_globs_both_roots(split, project):
    """Overlay semantics have to hold for a partitioned feed too, or a locally rebuilt
    partition would be invisible next to the shared ones."""
    for root, seasons in ((project / "shared", ["2024", "2025"]),
                          (project / "scratch", ["2026"])):
        d = root / "raw" / "box"
        d.mkdir(parents=True, exist_ok=True)
        for s in seasons:
            FRAME.write_parquet(d / f"{s}.parquet")
    found = split.state.instances_of("raw/box/{season}.parquet")
    assert found == ["raw/box/2024.parquet", "raw/box/2025.parquet",
                     "raw/box/2026.parquet"]


def test_no_split_by_default(project):
    iv = Invalidator(data_root=project / "data", data_version="v1",
                     project_root=project)
    assert iv.out_root is iv.data_root
    assert "out_root" not in repr(iv)


def test_overlay_off_reads_only_the_shared_tree(project):
    """Some projects define their inputs as the shared tree and treat local output as a
    side effect — wvorp reads DATA_BASE and writes DATA_OUT_BASE, never falling back."""
    shared, scratch = project / "shared", project / "scratch"
    (shared / "raw").mkdir(parents=True)
    FRAME.write_parquet(shared / "raw" / "src.parquet")
    local = scratch / "raw" / "src.parquet"
    local.parent.mkdir(parents=True)
    FRAME.with_columns(pl.col("a") * 10).write_parquet(local)

    on = Invalidator(data_root=shared, out_root=scratch, data_version="v1",
                     project_root=project)
    off = Invalidator(data_root=shared, out_root=scratch, data_version="v1",
                      overlay=False, project_root=project)
    assert on.resolve("raw/src.parquet") == local
    assert off.resolve("raw/src.parquet") == shared / "raw" / "src.parquet"
    assert off.resolve_out("processed/x.parquet") == scratch / "processed" / "x.parquet"


def test_writing_through_a_read_resolved_path_is_refused(project):
    """The gap that let a migration bug reach prod.

    `out_root` redirects every DECLARED write. A path handed back by `reads()` points at
    the shared tree, and a stray write through it is not declared at all — so nothing
    redirected it. A variable-name collision in a migration script did exactly this and
    overwrote a live parquet with a JSON dump.
    """
    from invalidator import DeclError

    shared, scratch = project / "shared", project / "scratch"
    (shared / "raw").mkdir(parents=True)
    FRAME.write_parquet(shared / "raw" / "src.parquet")
    iv = Invalidator(data_root=shared, out_root=scratch, data_version="v1",
                     overlay=False, project_root=project)
    iv.guard_writes()

    src = iv.reads("raw/src.parquet", why="an input")
    assert src == shared / "raw" / "src.parquet"
    with pytest.raises(DeclError, match="that path was handed out by iv.reads"):
        src.write_text("clobbered")
    assert FRAME.equals(pl.read_parquet(src)), "the shared file is untouched"

    # A declared output is unaffected: it resolves under out_root and was never an input.
    with iv.writes("processed/out.parquet", why="the output", terminal=True) as p:
        FRAME.write_parquet(p)
    assert p.exists()


def test_the_write_guard_still_allows_reading(project):
    """`open` is only a write when the MODE says so. Blocking it outright blocked
    fingerprinting — which opens an input to hash it — and so blocked the check itself."""
    from invalidator import DeclError
    shared, scratch = project / "shared", project / "scratch"
    (shared / "raw").mkdir(parents=True)
    FRAME.write_parquet(shared / "raw" / "src.parquet")
    iv = Invalidator(data_root=shared, out_root=scratch, data_version="v1",
                     overlay=False, project_root=project)
    iv.guard_writes()

    src = iv.reads("raw/src.parquet", why="an input")
    assert len(src.open("rb").read(4)) == 4
    assert FRAME.equals(pl.read_parquet(src))
    with pytest.raises(DeclError):
        src.open("w")


def test_a_partitioned_artifact_reuses_what_THIS_run_built(project):
    """The A/A caught this: a full rebuild followed by a reuse-everything pass LOST rows.

    The partition cache read its own artifact through `resolve()`, which with overlay off
    is the SHARED tree — so it reused rows it had not produced, and threw away whatever
    the rebuild had just added. An output has to be read from where outputs go.
    """
    shared, scratch = project / "shared", project / "scratch"
    (shared / "processed").mkdir(parents=True)
    # The shared tree holds an OLD, shorter version of the same artifact.
    pl.DataFrame({"season": ["2024"], "v": [1]}).write_parquet(
        shared / "processed" / "t.parquet")

    write_stage(project, "stages/p.py", '''
        import polars as pl
        from mypipe import iv

        def build_one(season):
            iv.reads("raw/src.parquet", why="the source")
            return pl.DataFrame({"season": [season], "v": [int(season)]})

        iv.for_each(["2024", "2025"], build_one, output="processed/t.parquet",
                    key="season", why="two partitions")
    ''')
    (shared / "raw").mkdir(parents=True, exist_ok=True)
    FRAME.write_parquet(shared / "raw" / "src.parquet")

    iv = Invalidator(data_root=shared, out_root=scratch, data_version="v1",
                     overlay=False, source_dirs=["stages"], project_root=project)
    from invalidator.partition import for_each

    def build_one(season):
        iv.reads("raw/src.parquet", why="the source")
        return pl.DataFrame({"season": [season], "v": [int(season)]})

    full = for_each(iv, ["2024", "2025"], build_one, output="processed/t.parquet",
                    key="season", why="two partitions")
    assert full.height == 2

    iv.state.reset()
    again = for_each(iv, ["2024", "2025"], build_one, output="processed/t.parquet",
                     key="season", why="two partitions")
    assert again.equals(full), "a reuse pass must not lose what the rebuild produced"
    assert (scratch / "processed" / "t.parquet").exists()
    assert pl.read_parquet(shared / "processed" / "t.parquet").height == 1, \
        "the shared tree is untouched"
