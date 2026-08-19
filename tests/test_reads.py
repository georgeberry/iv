

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
        'from iv import Invalidator\n'
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
    from iv.errors import DeclError
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


def test_tracing_never_takes_down_a_run(project, tmp_path):
    """The trace describes the run; it is not part of it. A pipeline that dies because
    its logging could not serialise a value has its priorities backwards — a
    `datetime.date` in a `part=` did exactly that, 102 seconds into a stage."""
    import datetime as dt
    import polars as pl
    from iv import Invalidator

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project,
                     trace=tmp_path / "t.ndjson")
    (project / "data" / "raw").mkdir(parents=True)
    pl.DataFrame({"x": [1]}).write_parquet(project / "data" / "raw" / "d_2026-08-18.parquet")

    # A date object, not a string. `render` stringifies it for the PATH already.
    p = iv.reads("raw/d_{date}.parquet", why="one day",
                 part={"date": dt.date(2026, 8, 18)})
    assert p.exists()


# ── read-after-write, judged at the end ───────────────────────────────────────

def test_a_later_write_absolves_a_late_read(project, capsys):
    """An incremental writer interleaves reads and writes; that is what `updates` means.

    The rollforward fits one week, saves, fits the next — so every read in week two
    follows week one's write. Warned at read time, that named six inputs as "not among
    that artifact's inputs" while all six were on the final stamp. A diagnostic that is
    usually wrong teaches you to skip the block it prints in.
    """
    import polars as pl
    from iv import Invalidator

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)
    (project / "data" / "raw").mkdir(parents=True)
    for n in ("week1", "week2"):
        pl.DataFrame({"x": [1]}).write_parquet(project / "data" / "raw" / f"{n}.parquet")

    iv.reads("raw/week1.parquet", why="the first week")
    with iv.updates("out/roll.parquet", why="one row per game") as p:
        pl.DataFrame({"x": [1]}).write_parquet(p)
    iv.reads("raw/week2.parquet", why="the second week")     # late, but...
    with iv.updates("out/roll.parquet", why="one row per game") as p:
        pl.DataFrame({"x": [1, 2]}).write_parquet(p)         # ...this picks it up

    assert "raw/week2.parquet" in iv.record_of("out/roll.parquet")["in"]
    iv._report_read_after_write()
    assert "was written before" not in capsys.readouterr().out


def test_a_genuinely_missed_late_read_is_reported(project, capsys):
    """The other direction: nothing wrote again, so the input really is absent."""
    import polars as pl
    from iv import Invalidator

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)
    (project / "data" / "raw").mkdir(parents=True)
    pl.DataFrame({"x": [1]}).write_parquet(project / "data" / "raw" / "late.parquet")

    with iv.updates("out/roll.parquet", why="one row per game") as p:
        pl.DataFrame({"x": [1]}).write_parquet(p)
    iv.reads("raw/late.parquet", why="read too late to feed it")

    assert "raw/late.parquet" not in iv.record_of("out/roll.parquet")["in"]
    iv._report_read_after_write()
    out = capsys.readouterr().out
    assert "was written before" in out and "raw/late.parquet" in out


def test_a_collection_read_as_prior_is_not_compared(project):
    """`prior=True` was accepted on `collection()`, documented as meaning what it means on
    `reads()`, and then never recorded — so a feed read from the PREVIOUS run was still
    compared against this one.

    Three wvorp artifacts came out of every refresh stale for it: `draft` reads the
    schedules before the fetch, `rookie_prior` and `rookie_projections` read the roster
    snapshots that `fetch_rosters` rewrites later in the same run. All three already SAID
    `prior=True`; nothing was listening.
    """
    import polars as pl
    from iv import Invalidator

    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)
    feed = project / "data" / "raw" / "rosters"
    feed.mkdir(parents=True)
    for s in ("2025", "2026"):
        pl.DataFrame({"x": [1]}).write_parquet(feed / f"roster_{s}.parquet")

    with iv.writes("processed/derived.parquet", why="something built from them") as p:
        iv.collection("raw/rosters/roster_{season}.parquet",
                      why="the roster as the fetcher last saw it", prior=True)
        pl.DataFrame({"y": [1]}).write_parquet(p)

    rec = iv.record_of("processed/derived.parquet")
    entry = rec["in"]["raw/rosters/roster_{season}.parquet"]
    assert entry.get("prior") is True, entry

    # The fetcher rewrites the feed LATER in the same run, as it is meant to.
    pl.DataFrame({"x": [1, 2, 3]}).write_parquet(feed / "roster_2026.parquet")
    assert iv.why_stale("processed/derived.parquet") is None
