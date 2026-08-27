from __future__ import annotations

import polars as pl
import pytest

from iv.core import Pipeline


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                    project=tmp_path)


def _seed(iv):
    with iv.writes("raw/feed/", why="an upstream feed") as out:
        pl.DataFrame({"a": [1, 2, 3]}).write_parquet(out)
    return iv.source("raw/feed/", why="an upstream feed")


def _build(iv, version, payload, ran, feed):
    iv._assets.clear()

    @iv.data(dataset="processed/fit/", why="the expensive one", version=version)
    def fit(f=iv.all_of(feed, why="the upstream")):
        ran["fit"] += 1
        return pl.DataFrame({"v": [payload]})

    @iv.data(dataset="processed/downstream/", why="reads the fit")
    def downstream(m=iv.all_of(fit, why="the fit")):
        ran["downstream"] += 1
        return m

    fit()
    downstream()


def test_bumping_version_rebuilds_the_stage(iv):
    feed = _seed(iv)
    ran = {"fit": 0, "downstream": 0}

    _build(iv, 1, "a", ran, feed)
    assert ran["fit"] == 1

    _build(iv, 1, "a", ran, feed)
    assert ran["fit"] == 1, "nothing moved, so nothing should rebuild"

    _build(iv, 2, "a", ran, feed)
    assert ran["fit"] == 2, "a version bump must rebuild the stage that carries it"


def test_a_version_bump_that_changes_nothing_leaves_downstream_alone(iv):
    feed = _seed(iv)
    ran = {"fit": 0, "downstream": 0}

    _build(iv, 1, "a", ran, feed)
    assert ran == {"fit": 1, "downstream": 1}

    _build(iv, 2, "a", ran, feed)
    assert ran["fit"] == 2
    assert ran["downstream"] == 1, (
        "the rebuilt output is identical, so its fingerprint is unchanged and "
        "downstream has nothing to answer to")


def test_a_version_bump_that_changes_the_output_carries_downstream(iv):
    feed = _seed(iv)
    ran = {"fit": 0, "downstream": 0}

    _build(iv, 1, "a", ran, feed)
    assert ran == {"fit": 1, "downstream": 1}

    _build(iv, 2, "b", ran, feed)
    assert ran == {"fit": 2, "downstream": 2}


def test_not_bumping_version_hides_a_changed_body(iv):
    feed = _seed(iv)
    ran = {"fit": 0, "downstream": 0}

    _build(iv, 1, "a", ran, feed)
    _build(iv, 1, "b", ran, feed)
    assert ran["fit"] == 1, (
        "iv keys on declared inputs, schema and version — never on the body. A "
        "changed body with the same version does NOT rebuild, which is exactly "
        "why version= has to be bumped by hand when a knob moves.")
