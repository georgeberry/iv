from __future__ import annotations

import polars as pl
import pytest

from iv import shards as _sh
from iv.core import Pipeline


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                    project=tmp_path)


def _seed(iv, n):
    with iv.writes("raw/seed/", why="an upstream feed") as out:
        pl.DataFrame({"n": [n]}).write_parquet(out)


def _cohorts(iv, seed, drafted, ran):
    @iv.data(dataset="derived/cohort/", part="year", why="one row per drafted cohort",
             universe=["2026", "2027"], allow_missing=True, version=1)
    def cohort(year, s=iv.all_of(seed, why="the draft, which decides who exists")):
        ran.append(year)
        return pl.DataFrame({"year": [year]}) if year in drafted else None

    return cohort


def test_a_partition_with_nothing_records_its_absence(iv):
    _seed(iv, 1)
    seed = iv.source("raw/seed/", why="an upstream feed")
    ran = []
    cohort = _cohorts(iv, seed, {"2026"}, ran)

    cohort("2026")
    cohort("2027")

    live = _sh.current_shards(iv.resolve_out("derived/cohort/"))
    assert sorted(live) == ["year=2026", "year=2027"]
    assert _sh.is_empty(live["year=2027"])
    assert not _sh.is_empty(live["year=2026"])


def test_a_recorded_absence_is_current_and_does_not_rebuild(iv):
    _seed(iv, 1)
    seed = iv.source("raw/seed/", why="an upstream feed")
    ran = []
    cohort = _cohorts(iv, seed, {"2026"}, ran)

    cohort("2027")
    assert ran == ["2027"]
    assert cohort.why_stale("2027") is None

    cohort("2027")
    assert ran == ["2027"], "an absence that is current must not run again"


def test_an_absence_rebuilds_into_data_when_its_inputs_move(iv):
    _seed(iv, 1)
    seed = iv.source("raw/seed/", why="an upstream feed")
    ran = []
    drafted = {"2026"}
    cohort = _cohorts(iv, seed, drafted, ran)

    cohort("2027")
    assert _sh.is_empty(_sh.current_shards(iv.resolve_out("derived/cohort/"))["year=2027"])

    drafted.add("2027")
    _seed(iv, 2)
    assert cohort.why_stale("2027") is not None
    cohort("2027")

    live = _sh.current_shards(iv.resolve_out("derived/cohort/"))
    assert not _sh.is_empty(live["year=2027"])
    assert len([p for p in iv.resolve_out("derived/cohort/").glob("year=2027*")]) == 1


def test_a_recorded_absence_is_not_handed_to_a_consumer(iv):
    _seed(iv, 1)
    seed = iv.source("raw/seed/", why="an upstream feed")
    cohort = _cohorts(iv, seed, {"2026"}, [])
    cohort("2026")
    cohort("2027")

    seen = []

    @iv.data(dataset="derived/rollup/", why="every cohort that exists", version=1)
    def rollup(rows=iv.all_of(cohort, as_paths=True, why="the cohorts")):
        seen.extend(str(r) for r in rows)
        return pl.DataFrame({"n": [len(rows)]})

    rollup()
    assert len(seen) == 1 and "year=2026" in seen[0]
    assert iv.verify("derived/cohort/") == []


def test_load_of_a_recorded_absence_is_none(iv):
    _seed(iv, 1)
    seed = iv.source("raw/seed/", why="an upstream feed")
    cohort = _cohorts(iv, seed, {"2026"}, [])
    cohort("2027")
    assert cohort.load({"year": "2027"}) is None
