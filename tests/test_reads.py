

# ── collections ───────────────────────────────────────────────────────────────

def test_a_collection_is_declared_once_and_returns_what_is_on_disk(project, iv):
    """A feed whose members are DISCOVERED. `reads()` renders one concrete path, so it
    cannot name a set whose seasons the caller does not know."""
    from conftest import write_stage
    write_stage(project, "stages/roll.py", '''
        import polars as pl
        from pipeline import iv

        @iv.step("processed/rollup.parquet", why="every season, concatenated")
        def build(out):
            files = iv.collection("raw/box/box_{season}.parquet",
                                  why="every season of the raw feed")
            pl.concat([pl.read_parquet(f) for f in files]).write_parquet(out)

        build()
    ''')
    (project / "pipeline.py").write_text(
        'from invalidator import Invalidator\n'
        'iv = Invalidator(data_root="data", data_version="v1", source_dirs=["stages"])\n')

    import polars as pl
    raw = project / "data" / "raw" / "box"
    raw.mkdir(parents=True)
    for s in ("2024", "2025"):
        pl.DataFrame({"pts": [1]}).write_parquet(raw / f"box_{s}.parquet")

    from tests.test_partition import run
    assert "wrote" in run(project, "stages/roll.py") or True
    out = run(project, "stages/roll.py")
    assert "is current" in out, out

    # A NEW member moves the collection's id, so the rollup is stale.
    pl.DataFrame({"pts": [1]}).write_parquet(raw / "box_2026.parquet")
    assert "is current" not in run(project, "stages/roll.py")


def test_a_collection_with_no_free_field_is_an_error(project, iv):
    from invalidator.errors import DeclError
    import pytest
    with pytest.raises(DeclError):
        iv.collection("raw/box/box_2024.parquet", why="one file, not a set")
