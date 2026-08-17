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
    assert (project / "scratch" / ".invalidator" / "state.json").exists()
    assert not (project / "shared" / ".invalidator").exists()


def test_state_path_puts_the_stamps_anywhere(project, tmp_path):
    shared = project / "shared"
    (shared / "raw").mkdir(parents=True)
    FRAME.write_parquet(shared / "raw" / "src.parquet")
    elsewhere = tmp_path / "somewhere" / "shadow.json"

    iv = Invalidator(data_root=shared, out_root=project / "scratch",
                     state_path=elsewhere, data_version="v1",
                     source_dirs=["stages"], project_root=project)
    with iv.writes("processed/out.parquet", why="the output", terminal=True) as p:
        FRAME.write_parquet(p)
    assert elsewhere.exists()
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
