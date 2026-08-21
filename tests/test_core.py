"""The five primitives, end to end, on a real directory tree.

Every assertion here is about a DECISION — did this stage run or skip — because that is
the only thing the package exists to get right.
"""
from __future__ import annotations

import polars as pl
import pytest

from iv import shards as _sh
from iv.core import Pipeline
from iv.errors import DeclError, StateError


@pytest.fixture
def pipe(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(root=tmp_path / "data", stage_dir=tmp_path / "stage",
                    project_root=tmp_path)


def seed(pipe, dataset, part=None, n=3, extra=0, why="a root feed"):
    """Put a shard on disk the way a fetcher would."""
    with pipe.writes(dataset, why=why, part=part) as out:
        pl.DataFrame({"a": range(n), "b": [x + extra for x in range(n)]}).write_parquet(out)


def rows(pipe, dataset):
    return pl.read_parquet(pipe.reads(dataset, why="check")).height


# ── the basic loop ────────────────────────────────────────────────────────────

def test_a_step_runs_then_skips(pipe):
    seed(pipe, "raw/feed/")
    calls = []

    @pipe.step("processed/out/", why="doubled")
    def build(out):
        calls.append(1)
        pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)

    assert build() is True and len(calls) == 1
    assert build() is False and len(calls) == 1
    assert rows(pipe, "processed/out/") == 3


def test_a_moved_input_rebuilds(pipe):
    seed(pipe, "raw/feed/")

    @pipe.step("processed/out/", why="doubled")
    def build(out):
        pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)

    build()
    seed(pipe, "raw/feed/", n=9)                 # different rows
    assert pipe.why_stale("processed/out/").startswith("input moved")
    assert build() is True
    assert rows(pipe, "processed/out/") == 9


def test_a_real_change_does_reach_downstream(pipe):
    seed(pipe, "raw/feed/")
    runs = []

    def build():
        @pipe.step("processed/end/", why="the expensive one")
        def end(out):
            runs.append(1)
            pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)
        return end()

    build()
    seed(pipe, "raw/feed/", extra=99)        # same shape, different numbers
    build()
    assert len(runs) == 2


# ── the index degrades to a rebuild ───────────────────────────────────────────

def test_losing_the_index_rebuilds_and_never_falsely_skips(pipe):
    seed(pipe, "raw/feed/")

    @pipe.step("processed/out/", why="doubled")
    def build(out):
        pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)

    build()
    assert build() is False
    (pipe.resolve_out("processed/out/") / _sh.INDEX_NAME).unlink()
    assert "no record" in pipe.why_stale("processed/out/")
    assert build() is True


def test_a_vanished_input_shard_is_named(pipe):
    seed(pipe, "raw/feed/", part={"season": "2025"})
    seed(pipe, "raw/feed/", part={"season": "2026"})

    @pipe.step("processed/out/", why="all of it")
    def build(out):
        pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)

    build()
    _sh.current_shards(pipe.resolve("raw/feed/"))["season=2026"].path.unlink()
    assert "vanished" in pipe.why_stale("processed/out/")


# ── failure never stamps ──────────────────────────────────────────────────────

def test_a_body_that_raises_stamps_nothing(pipe):
    seed(pipe, "raw/feed/")

    @pipe.step("processed/out/", why="doubled")
    def build(out):
        pipe.reads("raw/feed/", why="the feed")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        build()
    assert not pipe.resolve_out("processed/out/").exists()
    assert "not on disk" in pipe.why_stale("processed/out/")


def test_writing_nothing_is_an_error_unless_declared(pipe):
    with pytest.raises(DeclError, match="allow_missing"):
        with pipe.writes("processed/out/", why="nothing"):
            pass
    with pipe.writes("processed/out/", why="nothing yet", allow_missing=True):
        pass
    assert "not on disk" in pipe.why_stale("processed/out/")


def test_a_read_that_selects_nothing_raises(pipe):
    with pytest.raises(StateError, match="selected no shards"):
        pipe.reads("raw/missing/", why="not there")
    assert pipe.reads("raw/missing/", why="not there", optional=True) == []


def test_why_is_required(pipe):
    with pytest.raises(DeclError, match="why="):
        pipe.reads("raw/feed/", why="")


# ── for_each ──────────────────────────────────────────────────────────────────

def test_for_each_builds_once_then_reuses(pipe):
    for s in ("2024", "2025", "2026"):
        seed(pipe, "raw/box/", part={"season": s}, extra=int(s))
    built = []

    def one(season, out):
        built.append(season)
        pl.read_parquet(pipe.reads("raw/box/", why="raw box",
                                   where={"season": [season]})).write_parquet(out)

    run = lambda: pipe.for_each(["2024", "2025", "2026"], one,
                                dataset="processed/feat/", key="season",
                                why="per-season features", quiet=True)
    assert sorted(run()) == ["2024", "2025", "2026"]
    assert run() == []
    seed(pipe, "raw/box/", part={"season": "2026"}, extra=5)
    assert run() == ["2026"], "one new season rebuilds one season"
    assert len(built) == 4


def test_a_walk_forward_partition_cannot_see_the_future(pipe):
    """Past-only is structural: the shard is never opened, not opened-and-filtered."""
    for s in ("2024", "2025", "2026"):
        seed(pipe, "raw/box/", part={"season": s}, extra=int(s))

    def one(season, out):
        past = pipe.reads("raw/box/", why="seasons strictly before this cohort",
                          where={"season": lambda v: v < season})
        assert all(f"season={season}" not in p.name for p in past)
        pl.read_parquet(past).write_parquet(out)

    run = lambda: pipe.for_each(["2025", "2026"], one, dataset="processed/cohort/",
                                key="season", why="cohort fit on prior seasons", quiet=True)
    assert sorted(run()) == ["2025", "2026"]
    assert run() == []
    seed(pipe, "raw/box/", part={"season": "2026"}, extra=7)
    assert run() == [], "no cohort reads 2026, so a change to it moves nothing"
    seed(pipe, "raw/box/", part={"season": "2025"}, extra=7)
    assert run() == ["2026"], "only the cohort that read 2025 moves; 2025's reads 2024"


def test_an_explicit_selection_is_a_coverage_claim(pipe):
    seed(pipe, "raw/box/", part={"season": "2024"})
    with pytest.raises(StateError, match="2025"):
        pipe.reads("raw/box/", why="two seasons", where={"season": ["2024", "2025"]})


def test_a_prior_read_is_lineage_not_a_trigger(pipe):
    """Read-modify-write: the stage must not be stale against its own last output."""
    seed(pipe, "raw/feed/")

    def append_once():
        @pipe.step("raw/log/", why="a running history")
        def build(out):
            have = pipe.reads("raw/log/", why="yesterday's copy", prior=True, optional=True)
            old = pl.read_parquet(have) if have else pl.DataFrame(schema={"a": pl.Int64})
            new = pl.read_parquet(pipe.reads("raw/feed/", why="today")).select("a")
            pl.concat([old, new]).write_parquet(out)
        return build()

    append_once()
    assert rows(pipe, "raw/log/") == 3
    entry = _sh.read_index(pipe.resolve_out("raw/log/"))["shards"][""]
    assert "raw/log/" not in entry["inputs"] and entry["prior"] == ["raw/log/"]


# ── ordering ──────────────────────────────────────────────────────────────────

def test_reads_come_back_in_a_stable_semantic_order(pipe):
    for s in (2011, 2006, 2026, 2019):
        seed(pipe, "raw/box/", part={"season": str(s)}, extra=s)
    seen = {tuple(p.name for p in pipe.reads("raw/box/", why="all")) for _ in range(6)}
    assert len(seen) == 1
    got = pipe.reads("raw/box/", why="all")
    assert [p.name.split(".")[0] for p in got] == \
        ["season=2006", "season=2011", "season=2019", "season=2026"]


def test_a_code_edit_reruns_only_its_own_stage(pipe):
    """`code=` hashes the function's SOURCE, so this needs two real bodies.

    Both produce identical rows. The edited stage must re-run — that is what `code=` is
    for — and the expensive stage below it must not, because nothing it reads moved.
    """
    seed(pipe, "raw/feed/")
    end_runs = []

    def v1():
        @pipe.step("processed/mid/", why="passthrough")
        def mid(out):
            pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)
        return mid()

    def v2():
        @pipe.step("processed/mid/", why="passthrough")
        def mid(out):
            df = pl.read_parquet(pipe.reads("raw/feed/", why="the feed"))
            df = df.select(pl.all())              # a real edit, identical output
            df.write_parquet(out)
        return mid()

    def end():
        @pipe.step("processed/end/", why="the expensive one")
        def e(out):
            end_runs.append(1)
            pl.read_parquet(pipe.reads("processed/mid/", why="mid")).write_parquet(out)
        return e()

    v1(); end()
    assert len(end_runs) == 1
    before = sorted(x.name for x in pipe.resolve_out("processed/mid/").iterdir())

    assert v2() is True, "the source changed, so the stage that owns it re-runs"
    assert end() is False, "the numbers did not move, so the expensive stage does not"
    assert len(end_runs) == 1
    assert sorted(x.name for x in pipe.resolve_out("processed/mid/").iterdir()) == before


def test_reformatting_is_free(pipe):
    """The hash is over the parsed tree, so whitespace and comments are not edits."""
    seed(pipe, "raw/feed/")

    def tight():
        @pipe.step("processed/mid/", why="passthrough")
        def mid(out):
            pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)
        return mid()

    def spaced():
        @pipe.step("processed/mid/", why="passthrough")
        def mid(out):
            # a comment, and some breathing room
            pl.read_parquet(
                pipe.reads("raw/feed/", why="the feed")
            ).write_parquet(out)
        return mid()

    tight()
    assert spaced() is False


# ── everything is a file ──────────────────────────────────────────────────────

def test_a_version_is_a_file_and_only_moves_what_reads_it(pipe):
    """What `version="model"` used to be. The edge is now visible in the graph."""
    seed(pipe, "raw/feed/")
    pipe.constants("config/model/", why="what the fits answer to", v="m1")
    end_runs = []

    def build():
        @pipe.step("processed/modelled/", why="model output")
        def modelled(out):
            pipe.reads("config/model/", why="a model change must rebuild this")
            pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)

        @pipe.step("processed/plain/", why="no model in it")
        def plain(out):
            pl.read_parquet(pipe.reads("raw/feed/", why="the feed")).write_parquet(out)

        @pipe.step("processed/end/", why="the expensive one")
        def end(out):
            end_runs.append(1)
            pl.read_parquet(pipe.reads("processed/modelled/", why="mid")).write_parquet(out)
        return modelled(), plain(), end()

    build()
    assert len(end_runs) == 1
    pipe.constants("config/model/", why="what the fits answer to", v="m2")
    ran_modelled, ran_plain, ran_end = build()
    assert ran_modelled is True, "it declared it reads the model version"
    assert ran_plain is False, "it did not"
    assert ran_end is False and len(end_runs) == 1, "the numbers did not move"


def test_constants_are_idempotent_and_touch_nothing_when_unchanged(pipe):
    pipe.constants("config/model/", why="the model", v="m1")
    before = sorted(x.name for x in pipe.resolve_out("config/model/").iterdir())
    pipe.constants("config/model/", why="the model", v="m1")
    assert sorted(x.name for x in pipe.resolve_out("config/model/").iterdir()) == before


def test_the_clock_is_a_file(pipe, monkeypatch):
    """What `policy="clock"` used to be: a fetcher that must re-run daily reads the day."""
    import datetime as _d
    clock = lambda: pipe.constants("config/today/", why="poll once a day",
                                   date=_d.date.today().isoformat())
    clock()
    fetches = []

    def fetch():
        @pipe.step("raw/feed/", why="a polled feed")
        def build(out):
            pipe.reads("config/today/", why="poll once a day")
            pipe.external("some/api", why="the upstream")
            fetches.append(1)
            pl.DataFrame({"a": [1]}).write_parquet(out)
        return build()

    fetch()
    assert fetch() is False, "same day, no re-fetch"

    import datetime as dt

    class Tomorrow(dt.date):
        @classmethod
        def today(cls):
            return dt.date(2099, 1, 1)
    monkeypatch.setattr("iv.core._dt.date", Tomorrow)

    clock()
    assert fetch() is True and len(fetches) == 2


def test_a_stage_that_reads_no_clock_never_re_runs(pipe):
    """What `policy="settled"` used to be, and now it is just a fact about the reads."""
    calls = []

    @pipe.step("raw/archive/", why="fetch-once history")
    def build(out):
        pipe.external("sports-reference", why="a page that will not change")
        calls.append(1)
        pl.DataFrame({"a": [1]}).write_parquet(out)

    build()
    assert build() is False and len(calls) == 1
