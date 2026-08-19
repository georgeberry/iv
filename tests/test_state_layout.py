"""Where the stamps live: one file per artifact, and the single file that came before.

The layout is a correctness matter rather than a filing preference — see
`tests/test_parallel.py` for what a shared file does to a parallel run. What is here is
the addressing (one artifact, one file, no collisions) and the one-way move off the old
`state.json`, which is worth doing rather than telling people to delete it: the records
are unchanged in shape, and the alternative is a full rebuild for a bookkeeping change.
"""
from __future__ import annotations

import json
import shutil

import polars as pl
import pytest

from iv import Invalidator
from iv.errors import StateError
from iv.state import STATE_VERSION, read_records, record_filename

FRAME = pl.DataFrame({"a": [1, 2, 3]})


def build_one(iv, project):
    """Stamp one artifact with one root input, the ordinary way."""
    (project / "data" / "raw").mkdir(parents=True, exist_ok=True)
    FRAME.write_parquet(project / "data" / "raw" / "src.parquet")
    iv.reads("raw/src.parquet", why="the only source")
    with iv.writes("processed/out.parquet", why="the output", terminal=True) as p:
        FRAME.write_parquet(p)
    return "processed/out.parquet"


def as_legacy(records: dict) -> str:
    """The single file this package used to write, from the records it writes now."""
    return json.dumps({"version": 2, "artifacts": {
        rel: {k: v for k, v in r.items() if k not in ("state_version", "rel")}
        for rel, r in records.items()}})


# ── addressing ────────────────────────────────────────────────────────────────

def test_two_artifacts_that_sanitise_alike_get_different_files():
    """The digest is not decoration: `/` and `_` both fold to `_`, and two artifacts
    sharing a record file is a lost stamp by another route."""
    assert record_filename("raw/box/{season}.parquet") != record_filename("raw/box_{season}.parquet")
    assert record_filename("raw/box/2026.parquet") == record_filename("raw/box/2026.parquet")


def test_a_record_names_the_artifact_it_is_about(iv, project):
    rel = build_one(iv, project)
    raw = json.loads((iv.state.dir / record_filename(rel)).read_text())
    assert raw["rel"] == rel and raw["state_version"] == STATE_VERSION
    assert raw["in"] == {"raw/src.parquet": raw["in"]["raw/src.parquet"]}


def test_an_artifacts_own_version_is_not_shadowed_by_the_file_version(project):
    """`version=` on a step and the layout version are two different things, and one
    envelope key named `version` wrote every record's own version as empty."""
    iv = Invalidator(data_root=project / "data", data_version="v1",
                     versions={"model": "7"}, source_dirs=["stages"],
                     project_root=project)
    (project / "data" / "raw").mkdir(parents=True, exist_ok=True)
    with iv.writes("processed/out.parquet", why="the output", terminal=True,
                   version="model") as p:
        FRAME.write_parquet(p)
    raw = json.loads((iv.state.dir / record_filename("processed/out.parquet")).read_text())
    assert raw["version"] == "model:7"
    assert raw["state_version"] == STATE_VERSION


def test_a_corrupt_record_is_fatal_not_empty(iv, project):
    """An empty state makes invalidation a no-op and every builder rebuild forever, which
    reads as "the cache does not work" rather than as "the state is corrupt"."""
    rel = build_one(iv, project)
    (iv.state.dir / record_filename(rel)).write_text("{not json")
    iv.state.reset()
    with pytest.raises(StateError, match="unreadable"):
        iv.record_of(rel)


# ── the move off the single file ──────────────────────────────────────────────

def test_a_legacy_state_json_migrates_and_the_stamps_still_count(iv, project):
    rel = build_one(iv, project)
    assert iv.is_current(rel)

    records = read_records(iv.state.dir)
    iv.state.legacy_path.write_text(as_legacy(records))
    shutil.rmtree(iv.state.dir)

    fresh = Invalidator(data_root=project / "data", data_version="v1",
                        source_dirs=["stages"], project_root=project)
    assert fresh.is_current(rel), "a migrated stamp has to mean what it meant before"
    assert set(read_records(fresh.state.dir)) == set(records)
    assert not fresh.state.legacy_path.exists(), "the migrated file is moved aside"
    assert (project / "data" / ".iv" / "state.json.migrated").exists()


def test_migration_does_not_run_once_the_directory_is_there(iv, project):
    """A stale single file beside a live directory is not a second source of truth."""
    rel = build_one(iv, project)
    iv.state.legacy_path.write_text(as_legacy({"processed/ghost.parquet": {
        "state_version": STATE_VERSION, "rel": "processed/ghost.parquet", "id": "x"}}))
    iv.state.reset()
    assert set(iv.state.records()) == {rel}


def test_an_unreadable_legacy_file_is_fatal(iv, project):
    iv.state.legacy_path.parent.mkdir(parents=True, exist_ok=True)
    iv.state.legacy_path.write_text('{"version": 99, "artifacts": {}}')
    with pytest.raises(StateError, match="version 99"):
        iv.state.records()


def test_a_state_path_naming_the_old_file_still_works(project, tmp_path):
    """Config written for the single file keeps working: it names the directory beside
    it, which is the same path the migration reads."""
    elsewhere = tmp_path / "somewhere" / "shadow.json"
    iv = Invalidator(data_root=project / "data", data_version="v1",
                     state_path=elsewhere, source_dirs=["stages"], project_root=project)
    rel = build_one(iv, project)
    assert iv.state.dir == tmp_path / "somewhere" / "shadow"
    assert iv.state.legacy_path == elsewhere
    assert set(read_records(iv.state.dir)) == {rel}


def test_the_rename_moves_the_stamps_rather_than_rebuilding(project):
    """`invalidator` -> `iv` renamed the state directory too. Copy, do not rebuild.

    For wvorp that directory is the difference between a no-op refresh and re-running the
    287-second xPM fit and everything below it — for a package name.
    """
    import json

    import polars as pl
    from iv import Invalidator

    old = project / "data" / ".invalidator" / "state"
    old.mkdir(parents=True)
    (old / "processed_thing.parquet.json").write_text(json.dumps(
        {"state_version": 3, "rel": "processed/thing.parquet", "id": "abc", "in": {}}))

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)
    rec = iv.record_of("processed/thing.parquet")
    assert rec is not None and rec["id"] == "abc", "the stamp survived the rename"
    assert (project / "data" / ".iv" / "state").exists()
    # A MOVE, not a copy. Two directories both called `state`, both holding stamps, one
    # of them dead, is the exact hazard this package exists to prevent.
    assert not (project / "data" / ".invalidator").exists()


def test_a_half_finished_move_completes(project):
    """It may find the destination already populated — finish the move anyway."""
    import json

    from iv import Invalidator

    old = project / "data" / ".invalidator" / "state"
    new = project / "data" / ".iv" / "state"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    rec = {"state_version": 3, "rel": "processed/thing.parquet", "id": "new", "in": {}}
    (new / "processed_thing.parquet.json").write_text(json.dumps(rec))
    (old / "processed_thing.parquet.json").write_text(json.dumps({**rec, "id": "old"}))
    (old / "processed_other.parquet.json").write_text(json.dumps(
        {"state_version": 3, "rel": "processed/other.parquet", "id": "o", "in": {}}))

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)
    assert iv.record_of("processed/thing.parquet")["id"] == "new", "destination wins"
    assert iv.record_of("processed/other.parquet")["id"] == "o", "the rest came across"
    assert not (project / "data" / ".invalidator").exists()
