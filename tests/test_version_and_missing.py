"""Two narrower rules than `data_version`, both learned from a real pipeline.

`version=` — an artifact can answer to something beyond the data. 17 of wvorp's 31 tracked
artifacts depend on a MODEL_VERSION that must NOT rebuild the feature pipeline: that is
24s for possessions, 27-142s for box features, 287s for the fit, thrown away every time a
prior changes.

`allow_missing=` — a builder can legitimately produce nothing. A walk-forward projection
with no season to project yet, a roster that does not exist until July. wvorp's guard calls
this "the normal in-season state, not a failure".
"""
from __future__ import annotations

import polars as pl
import pytest

from conftest import write_stage
from invalidator import ConfigError, DeclError, Invalidator

FRAME = pl.DataFrame({"a": [1, 2, 3]})

STAGES = '''
    from mypipe import iv

    @iv.step("processed/features.parquet", why="the feature pipeline", code=False)
    def features(out):
        iv.reads("raw/src.parquet", why="the source")

    @iv.step("processed/fit.parquet", why="the model fit", version="model", code=False)
    def fit(out):
        iv.reads("processed/features.parquet", why="the features")
'''


def make(project, model="m1", data="v1"):
    return Invalidator(data_root=project / "data", data_version=data,
                       versions={"model": model}, source_dirs=["stages"],
                       project_root=project)


@pytest.fixture
def two(project):
    write_stage(project, "stages/s.py", STAGES)
    raw = project / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    FRAME.write_parquet(raw / "src.parquet")

    def build(iv):
        @iv.step("processed/features.parquet", why="the feature pipeline", code=False)
        def features(out):
            iv.reads("raw/src.parquet", why="the source")
            FRAME.write_parquet(out)

        @iv.step("processed/fit.parquet", why="the model fit", version="model", code=False)
        def fit(out):
            iv.reads("processed/features.parquet", why="the features")
            FRAME.write_parquet(out)

        iv.state.reset()
        return [features(), fit()]

    return build


# ── version= ──────────────────────────────────────────────────────────────────

def test_a_version_bump_moves_only_the_artifacts_that_name_it(project, two):
    assert two(make(project)) == [True, True]
    assert two(make(project)) == [False, False]

    bumped = make(project, model="m2")
    assert bumped.why_stale("processed/features.parquet") is None, \
        "the feature pipeline does not name the model version, so it cannot see it"
    assert bumped.why_stale("processed/fit.parquet") == \
        "version bumped: model:m1 -> model:m2"
    assert two(bumped) == [False, True]


def test_the_global_data_version_still_moves_everything(project, two):
    two(make(project))
    bumped = make(project, data="v2")
    assert bumped.why_stale("processed/features.parquet") == "data_version bumped: v1 -> v2"
    assert bumped.why_stale("processed/fit.parquet") == "data_version bumped: v1 -> v2"


def test_an_unknown_version_name_raises_rather_than_defaulting(project, two):
    """A default here would key the artifact on a constant, and an artifact keyed on a
    constant is permanently, silently current."""
    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project)   # no versions=

    @iv.step("processed/fit.parquet", why="the model fit", version="model", code=False)
    def fit(out):
        FRAME.write_parquet(out)

    with pytest.raises(ConfigError, match="not one of this Invalidator's versions"):
        fit()


def test_version_must_be_a_literal_name_not_the_value(project):
    """Passing the value would be a name the scan cannot resolve, and then the CLI could
    never see a bump that a run would see."""
    with pytest.raises(DeclError, match='version= must be a string literal'):
        write_stage(project, "stages/bad.py", '''
            from mypipe import iv
            MODEL_VERSION = "m1"

            @iv.step("processed/x.parquet", why="a computed version",
                     version=MODEL_VERSION, code=False)
            def build(out):
                iv.reads("raw/src.parquet", why="the source")
        ''')
        make(project).graph()


def test_the_stored_record_carries_the_resolved_version(project, two):
    iv = make(project)
    two(iv)
    assert iv.record_of("processed/fit.parquet")["version"] == "model:m1"
    assert iv.record_of("processed/features.parquet")["version"] == ""


# ── allow_missing= ────────────────────────────────────────────────────────────

def test_an_output_that_is_not_written_is_not_stamped_and_does_not_raise(project, capsys):
    write_stage(project, "stages/s.py", '''
        from mypipe import iv

        @iv.step("processed/maybe.parquet", why="a projection with nothing to project yet",
                 allow_missing=True, terminal=True, code=False)
        def build(out):
            iv.reads("raw/src.parquet", why="the source")
    ''')
    raw = project / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    FRAME.write_parquet(raw / "src.parquet")
    iv = make(project)

    produced = []

    @iv.step("processed/maybe.parquet", why="a projection with nothing to project yet",
             allow_missing=True, terminal=True, code=False)
    def build(out):
        iv.reads("raw/src.parquet", why="the source")
        if produced:
            FRAME.write_parquet(out)

    assert build() is True                       # it ran
    assert "nothing produced — not stamped" in capsys.readouterr().out
    assert iv.record_of("processed/maybe.parquet") is None

    iv.state.reset()
    assert iv.why_stale("processed/maybe.parquet") == "not on disk"

    produced.append(1)                           # now it has something to write
    assert build() is True
    iv.state.reset()
    assert iv.why_stale("processed/maybe.parquet") is None


def test_without_allow_missing_not_writing_is_an_error(project):
    from invalidator import StateError
    iv = make(project)

    @iv.step("processed/gone.parquet", why="should have written", terminal=True, code=False)
    def build(out):
        pass

    with pytest.raises(StateError, match="declared written but is not on disk"):
        build()
