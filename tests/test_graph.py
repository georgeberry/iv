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


def pipe_for(project, **kw):
    return Pipeline(root=project / "data", source_dirs=["stages"],
                    project_root=project, **kw)


def g_for(project, **kw):
    return _graph.build(pipe_for(project, **kw))


CHAIN = '''
    from p import pipe

    @pipe.step("processed/mid/", why="the middle")
    def build(out):
        pipe.reads("raw/feed/", why="the feed")
'''


# ── the scan ──────────────────────────────────────────────────────────────────

def test_a_decorator_and_a_body_read_are_one_stage(project):
    write_stage(project, "stages/mid.py", CHAIN)
    st = _static.scan(pipe_for(project))["stages/mid.py"]
    assert [s.dataset for s in st.outputs] == ["processed/mid/"]
    assert [s.dataset for s in st.inputs] == ["raw/feed/"]
    assert st.outputs[0].why == "the middle"


def test_a_computed_dataset_name_is_refused(project):
    write_stage(project, "stages/bad.py", '''
        from p import pipe
        DS = "processed/x/"
        pipe.reads(DS, why="computed, so unreadable without running it")
    ''')
    with pytest.raises(DeclError, match="string LITERAL"):
        _static.scan(pipe_for(project))


def test_a_call_without_why_is_not_ours(project):
    """The `why=` is what makes a match deliberate; a bare `.reads(...)` is someone else's."""
    write_stage(project, "stages/other.py", '''
        from p import pipe
        conn.reads("not/ours/")
        pipe.reads("raw/feed/", why="ours")
    ''')
    st = _static.scan(pipe_for(project))["stages/other.py"]
    assert [s.dataset for s in st.inputs] == ["raw/feed/"]


def test_a_trailing_slash_is_not_a_second_spelling(project):
    write_stage(project, "stages/a.py", '''
        from p import pipe
        @pipe.step("processed/mid", why="no slash")
        def build(out):
            pipe.reads("raw/feed/", why="the feed")
    ''')
    write_stage(project, "stages/b.py", '''
        from p import pipe
        @pipe.step("processed/end/", why="consumes it")
        def build(out):
            pipe.reads("processed/mid/", why="with a slash")
    ''')
    g = g_for(project)
    assert g.producers_of("processed/mid/") == ["stages/a.py"]
    assert g.consumers_of("processed/mid/") == ["stages/b.py"]


# ── the checks ────────────────────────────────────────────────────────────────

def test_a_clean_pipeline_passes(project):
    write_stage(project, "stages/mid.py", CHAIN)
    write_stage(project, "stages/end.py", '''
        from p import pipe
        @pipe.step("dump/site/", why="the app reads it", terminal=True)
        def build(out):
            pipe.reads("processed/mid/", why="the middle")
    ''')
    errors, warns = _graph.check(g_for(project))
    assert errors == [] and warns == []


def test_a_read_with_no_producer_outside_the_roots_is_an_error(project):
    write_stage(project, "stages/mid.py", '''
        from p import pipe
        @pipe.step("processed/mid/", why="the middle", terminal=True)
        def build(out):
            pipe.reads("processed/nobody_writes/", why="nothing produces this")
    ''')
    errors, _ = _graph.check(g_for(project))
    assert any("READ WITH NO PRODUCER" in e for e in errors)


def test_a_root_prefix_is_how_out_of_band_data_is_declared(project):
    write_stage(project, "stages/mid.py", CHAIN.replace(
        'def build(out):', 'def build(out):').replace(
        '@pipe.step("processed/mid/", why="the middle")',
        '@pipe.step("processed/mid/", why="the middle", terminal=True)'))
    errors, _ = _graph.check(g_for(project))
    assert errors == [], "raw/ is a root, so nothing needs to produce it"


def test_a_write_nobody_reads_needs_terminal(project):
    write_stage(project, "stages/mid.py", CHAIN)
    errors, _ = _graph.check(g_for(project))
    assert any("WRITE WITH NO CONSUMER" in e for e in errors)


def test_two_writers_of_one_dataset_is_an_error(project):
    for name in ("a", "b"):
        write_stage(project, f"stages/{name}.py", f'''
            from p import pipe
            @pipe.step("processed/mid/", why="both think they own it", terminal=True)
            def build_{name}(out):
                pipe.reads("raw/feed/", why="the feed")
        ''')
    errors, _ = _graph.check(g_for(project))
    assert any("TWO WRITERS" in e for e in errors)


def test_a_cycle_is_caught(project):
    write_stage(project, "stages/a.py", '''
        from p import pipe
        @pipe.step("processed/a/", why="a")
        def build(out):
            pipe.reads("processed/b/", why="b")
    ''')
    write_stage(project, "stages/b.py", '''
        from p import pipe
        @pipe.step("processed/b/", why="b")
        def build(out):
            pipe.reads("processed/a/", why="a")
    ''')
    errors, _ = _graph.check(g_for(project))
    assert any("CYCLE" in e for e in errors)


def test_a_stage_with_no_inputs_is_warned_about(project):
    """It can never go stale, which is right for a fetch-once archive and a bug otherwise."""
    write_stage(project, "stages/fetch.py", '''
        from p import pipe
        @pipe.step("raw/archive/", why="fetch-once history", terminal=True)
        def build(out):
            pipe.external("sports-reference", why="a page that will not change")
    ''')
    errors, warns = _graph.check(g_for(project))
    assert errors == [] and any("RUNS ONCE" in w for w in warns)


def test_constants_are_a_source_and_not_warned_about(project):
    """`constants` having no inputs is the point, not a symptom."""
    write_stage(project, "stages/config.py", '''
        from p import pipe
        pipe.constants("config/model/", why="what the fits answer to", v="m1")
    ''')
    write_stage(project, "stages/fit.py", '''
        from p import pipe
        @pipe.step("processed/xpm/", why="the fit", terminal=True)
        def build(out):
            pipe.reads("config/model/", why="a model change rebuilds this")
    ''')
    errors, warns = _graph.check(g_for(project))
    assert errors == [] and warns == []


def test_a_consumer_running_before_its_producer_is_an_error(project):
    write_stage(project, "stages/mid.py", CHAIN)
    write_stage(project, "stages/end.py", '''
        from p import pipe
        @pipe.step("dump/site/", why="the app reads it", terminal=True)
        def build(out):
            pipe.reads("processed/mid/", why="the middle")
    ''')
    (project / "refresh.sh").write_text(
        "uv run python stages/end.py\nuv run python stages/mid.py\n")
    errors, _ = _graph.check(g_for(project, order_from="refresh.sh"))
    assert any("ORDER" in e for e in errors)


def test_a_commented_out_stage_is_not_in_the_order(project):
    write_stage(project, "stages/mid.py", CHAIN)
    write_stage(project, "stages/end.py", '''
        from p import pipe
        @pipe.step("dump/site/", why="the app reads it", terminal=True)
        def build(out):
            pipe.reads("processed/mid/", why="the middle")
    ''')
    (project / "refresh.sh").write_text(
        "uv run python stages/mid.py\n# uv run python stages/end.py\n")
    order = _graph.declared_order(pipe_for(project, order_from="refresh.sh"))
    assert order == ["stages/mid.py"]


# ── drift ─────────────────────────────────────────────────────────────────────

def test_an_undeclared_read_is_an_error_and_an_unseen_one_is_a_warning(project):
    write_stage(project, "stages/mid.py", CHAIN)
    g = g_for(project)
    events = [{"kind": "io", "op": "read", "node": "stages/mid.py", "rel": "raw/secret/"},
              {"kind": "io", "op": "write", "node": "stages/mid.py", "rel": "processed/mid/"}]
    errors, warns = _graph.drift(g, events)
    assert any("UNDECLARED READ" in e and "raw/secret/" in e for e in errors)
    assert any("raw/feed/" in w for w in warns)
