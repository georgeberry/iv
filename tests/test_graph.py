"""The scan and the checks: what the code DECLARES, read without running it."""
from __future__ import annotations

import pytest

from iv import graph as _graph
from iv import static as _static
from iv.core import Pipeline
from iv.errors import DeclError

from tests.conftest import write_stage


@pytest.fixture
def project(tmp_path):
    (tmp_path / "stages").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    return tmp_path


def iv_for(project, **kw):
    return Pipeline(root=project / "data", source_dirs=["stages"],
                    project_root=project, **kw)


def g_for(project, **kw):
    return _graph.build(iv_for(project, **kw))


def stage(project, name, body):
    write_stage(project, f"stages/{name}.py", "from p import iv\n\n" + body)


MID = '''
@iv.step(why="the middle")
def build():
    iv.reads("raw/feed/", why="the feed")
    with iv.writes("processed/mid/", why="the middle"):
        pass
'''

END = '''
@iv.step(why="the app reads it")
def build():
    iv.reads("processed/mid/", why="the middle")
    with iv.writes("dump/site/", why="the app reads it", terminal=True):
        pass
'''


# ── the scan ──────────────────────────────────────────────────────────────────

def test_a_step_marks_a_stage_and_its_writes_are_the_outputs(project):
    stage(project, "mid", MID)
    st = _static.scan(iv_for(project))["stages/mid.py"]
    assert st.steps == ("build",)
    assert [s.dataset for s in st.outputs] == ["processed/mid/"]
    assert [s.dataset for s in st.inputs] == ["raw/feed/"]
    assert st.outputs[0].why == "the middle"


def test_a_step_needs_a_why(project):
    stage(project, "bad", "@iv.step()\ndef build():\n    pass\n")
    with pytest.raises(DeclError, match="why="):
        _static.scan(iv_for(project))


def test_a_computed_dataset_name_is_refused(project):
    stage(project, "bad", 'DS = "processed/x/"\niv.reads(DS, why="unreadable")\n')
    with pytest.raises(DeclError, match="string LITERAL"):
        _static.scan(iv_for(project))


def test_a_call_without_why_is_not_ours(project):
    """The `why=` is what makes a match deliberate; a bare `.reads(...)` is someone else's."""
    stage(project, "other", 'conn.reads("not/ours/")\niv.reads("raw/feed/", why="ours")\n')
    st = _static.scan(iv_for(project))["stages/other.py"]
    assert [s.dataset for s in st.inputs] == ["raw/feed/"]


def test_a_trailing_slash_is_not_a_second_spelling(project):
    stage(project, "a", '@iv.step(why="no slash")\ndef build():\n'
                        '    iv.reads("raw/feed/", why="the feed")\n'
                        '    with iv.writes("processed/mid", why="no slash"):\n        pass\n')
    stage(project, "b", MID.replace('iv.reads("raw/feed/", why="the feed")',
                                    'iv.reads("processed/mid/", why="with a slash")')
                           .replace('"processed/mid/", why="the middle"',
                                    '"processed/end/", why="downstream", terminal=True'))
    g = g_for(project)
    assert g.producers_of("processed/mid/") == ["stages/a.py"]
    assert g.consumers_of("processed/mid/") == ["stages/b.py"]


def test_the_scan_finds_a_write_inside_a_helper(project):
    stage(project, "mid", '''
def _extra():
    with iv.writes("processed/extra/", why="from a helper", terminal=True):
        pass

@iv.step(why="the stage")
def build():
    iv.reads("raw/feed/", why="the feed")
    with iv.writes("processed/mid/", why="its own", terminal=True):
        pass
    _extra()
''')
    st = _static.scan(iv_for(project))["stages/mid.py"]
    assert {s.dataset for s in st.outputs_of("build")} == \
        {"processed/mid/", "processed/extra/"}


# ── the checks ────────────────────────────────────────────────────────────────

def test_a_clean_pipeline_passes(project):
    stage(project, "mid", MID)
    stage(project, "end", END)
    assert _graph.check(g_for(project)) == ([], [])


def test_a_read_with_no_producer_outside_the_roots_is_an_error(project):
    stage(project, "mid", '''
@iv.step(why="the middle")
def build():
    iv.reads("processed/nobody_writes/", why="nothing produces this")
    with iv.writes("processed/mid/", why="the middle", terminal=True):
        pass
''')
    errors, _ = _graph.check(g_for(project))
    assert any("READ WITH NO PRODUCER" in e for e in errors)


def test_a_root_prefix_is_how_out_of_band_data_is_declared(project):
    stage(project, "mid", MID.replace('why="the middle"):', 'why="the middle", terminal=True):'))
    assert _graph.check(g_for(project))[0] == [], "raw/ is a root"


def test_a_write_nobody_reads_needs_terminal(project):
    stage(project, "mid", MID)
    errors, _ = _graph.check(g_for(project))
    assert any("WRITE WITH NO CONSUMER" in e for e in errors)


def test_two_writers_of_one_dataset_is_an_error(project):
    for n in ("a", "b"):
        stage(project, n, MID.replace("def build()", f"def build_{n}()")
                             .replace('why="the middle"):', 'why="the middle", terminal=True):'))
    errors, _ = _graph.check(g_for(project))
    assert any("TWO WRITERS" in e for e in errors)


def test_a_cycle_is_caught(project):
    stage(project, "a", '@iv.step(why="a")\ndef build():\n'
                        '    iv.reads("processed/b/", why="b")\n'
                        '    with iv.writes("processed/a/", why="a"):\n        pass\n')
    stage(project, "b", '@iv.step(why="b")\ndef build():\n'
                        '    iv.reads("processed/a/", why="a")\n'
                        '    with iv.writes("processed/b/", why="b"):\n        pass\n')
    errors, _ = _graph.check(g_for(project))
    assert any("CYCLE" in e for e in errors)


def test_a_write_from_a_helper_is_reported(project):
    """The runtime skip check reads the STEP's own source, so a helper's write is invisible
    to it. The project scan can see it, so this is where it gets said — otherwise the hole
    that made the skip check miss an output would simply come back."""
    stage(project, "mid", '''
def _extra():
    with iv.writes("processed/extra/", why="from a helper", terminal=True):
        pass

@iv.step(why="the stage")
def build():
    iv.reads("raw/feed/", why="the feed")
    with iv.writes("processed/mid/", why="its own", terminal=True):
        pass
    _extra()
''')
    errors, _ = _graph.check(g_for(project))
    assert any("WRITE OUTSIDE THE STEP" in e for e in errors)


def test_a_stage_with_no_inputs_is_warned_about(project):
    stage(project, "fetch", '''
@iv.step(why="fetch-once history")
def build():
    iv.external("sports-reference", why="a page that will not change")
    with iv.writes("raw/archive/", why="fetch-once history", terminal=True):
        pass
''')
    errors, warns = _graph.check(g_for(project))
    assert errors == [] and any("RUNS ONCE" in w for w in warns)


def test_constants_are_a_source_and_not_warned_about(project):
    stage(project, "config", 'iv.constants("config/model/", why="the model", v="m1")\n')
    stage(project, "fit", '''
@iv.step(why="the fit")
def build():
    iv.reads("config/model/", why="a model change rebuilds this")
    with iv.writes("processed/xpm/", why="the fit", terminal=True):
        pass
''')
    assert _graph.check(g_for(project)) == ([], [])


def test_a_consumer_running_before_its_producer_is_an_error(project):
    stage(project, "mid", MID)
    stage(project, "end", END)
    (project / "refresh.sh").write_text(
        "uv run python stages/end.py\nuv run python stages/mid.py\n")
    errors, _ = _graph.check(g_for(project, order_from="refresh.sh"))
    assert any("ORDER" in e for e in errors)


def test_a_commented_out_stage_is_not_in_the_order(project):
    stage(project, "mid", MID)
    stage(project, "end", END)
    (project / "refresh.sh").write_text(
        "uv run python stages/mid.py\n# uv run python stages/end.py\n")
    assert _graph.declared_order(iv_for(project, order_from="refresh.sh")) == \
        ["stages/mid.py"]


# ── drift ─────────────────────────────────────────────────────────────────────

def test_an_undeclared_read_is_an_error_and_an_unseen_one_is_a_warning(project):
    stage(project, "mid", MID)
    events = [{"kind": "io", "op": "read", "node": "stages/mid.py", "rel": "raw/secret/"},
              {"kind": "io", "op": "write", "node": "stages/mid.py", "rel": "processed/mid/"}]
    errors, warns = _graph.drift(g_for(project), events)
    assert any("UNDECLARED READ" in e and "raw/secret/" in e for e in errors)
    assert any("raw/feed/" in w for w in warns)


def test_a_prior_read_does_not_count_as_a_trigger(project):
    """An accumulator whose only read is its own last copy runs once and never again.

    `prior=` is excluded from the comparison by design, so it cannot be the thing that
    makes a stage stale — and a stage with nothing else to read is the silent-failure case
    this check exists for.
    """
    stage(project, "log", '''
@iv.step(why="appends to its own last copy")
def build():
    iv.reads("raw/log/", why="yesterday's copy", prior=True, optional=True)
    with iv.writes("raw/log/", why="a running log", terminal=True):
        pass
''')
    errors, warns = _graph.check(g_for(project))
    assert errors == [] and any("RUNS ONCE" in w and "prior=" in w for w in warns)


def test_a_prior_read_is_not_a_dependency_edge(project):
    """Its producer runs LATER; treating it as an edge would be a cycle on paper."""
    stage(project, "draft", '''
@iv.step(why="runs before the fetch, on last run's copy")
def build():
    iv.reads("raw/schedule/", why="the previous run's copy", prior=True)
    with iv.writes("processed/draft/", why="the draft table", terminal=True):
        pass
''')
    stage(project, "fetch", '''
@iv.step(why="fetches the schedule, later in the same run")
def build():
    iv.reads("processed/draft/", why="which classes to fetch")
    with iv.writes("raw/schedule/", why="the schedule"):
        pass
''')
    errors, _ = _graph.check(g_for(project))
    assert not any("CYCLE" in e for e in errors)
    assert g_for(project).parent_map()["stages/draft.py"] == []
