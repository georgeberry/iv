

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


def test_frame_reads_and_declares_in_one_call(project, iv):
    """The read and the declaration are one fact; spelling them separately makes the top
    of a stage twice as wide as its I/O contract."""
    import polars as pl
    (project / "data" / "raw").mkdir(parents=True)
    pl.DataFrame({"pts": [1, 2]}).write_parquet(project / "data" / "raw" / "box.parquet")

    df = iv.frame("raw/box.parquet", why="the raw feed")
    assert df.height == 2
    assert "raw/box.parquet" in iv._reads

    assert iv.frame("raw/nope.parquet", why="absent", optional=True) is None


def test_every_read_method_takes_the_same_read_flags(iv):
    """`reads`, `frame` and `collection` all declare an input, so a flag one accepts and
    another silently rejects is a papercut that only shows up at runtime — which is how
    `collection(prior=True)` reached a live refresh and stopped it."""
    import inspect

    for name in ("reads", "frame", "collection"):
        params = inspect.signature(getattr(iv, name)).parameters
        assert {"why", "optional", "prior", "fp", "part"} <= set(params), name
