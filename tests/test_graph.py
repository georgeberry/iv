"""The graph and the checks: what the code DECLARES, read without running it.

A declared stage registers itself when its module imports, so these build real pipelines
rather than writing source files and parsing them back. The preflight tests at the bottom
still write files, because pyflakes reads files.
"""
from __future__ import annotations

import polars as pl
import pytest

from iv import Pipeline
from iv import graph as _graph
from iv import static as _static
from iv.errors import DeclError

from tests.conftest import write_stage


@pytest.fixture
def project(tmp_path):
    (tmp_path / "stages").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    return tmp_path


@pytest.fixture
def iv(tmp_path, monkeypatch):
    for var in ("IV_TRACE", "IV_FORCE", "IV_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(root=tmp_path / "data", stage_dir=tmp_path / "stage",
                    project_root=tmp_path)


def frame():
    return pl.DataFrame({"a": [1, 2]})


def check(iv):
    return _graph.check(_graph.build(iv))


# ── the checks ────────────────────────────────────────────────────────────────

def test_a_clean_pipeline_passes(iv):
    @iv.data("processed/mid/", why="the middle")
    def mid(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(m=iv.all_of("processed/mid/", why="the middle")):
        return m

    assert check(iv) == ([], [])


def test_a_read_with_no_producer_outside_the_roots_is_an_error(iv):
    @iv.data("processed/mid/", why="the middle", terminal=True)
    def mid(x=iv.all_of("processed/nobody_writes/", why="nothing produces this")):
        return x

    errors, _ = check(iv)
    assert any("READ WITH NO PRODUCER" in e for e in errors)


def test_an_optional_read_with_no_producer_is_only_a_warning(iv):
    @iv.data("processed/mid/", why="the middle", terminal=True)
    def mid(x=iv.all_of("processed/maybe/", optional=True, why="may not be there")):
        return x

    errors, warns = check(iv)
    assert errors == [] and any("READ WITH NO PRODUCER" in w for w in warns)


def test_a_root_prefix_is_how_out_of_band_data_is_declared(iv):
    """`raw/` is declared a root, so nothing has to produce it."""
    @iv.data("processed/mid/", why="the middle", terminal=True)
    def mid(feed=iv.all_of("raw/feed/", why="arrives out of band")):
        return feed

    assert check(iv)[0] == []


def test_a_write_nobody_reads_needs_terminal(iv):
    @iv.data("processed/mid/", why="nothing downstream reads this")
    def mid(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    errors, _ = check(iv)
    assert any("WRITE WITH NO CONSUMER" in e for e in errors)


def test_a_cycle_is_caught(iv):
    @iv.data("processed/a/", why="a")
    def a(b=iv.all_of("processed/b/", why="b")):
        return b

    @iv.data("processed/b/", why="b")
    def b(a=iv.all_of("processed/a/", why="a")):
        return a

    errors, _ = check(iv)
    assert any("CYCLE" in e for e in errors)


def test_a_stage_with_no_inputs_is_warned_about(iv):
    """`once=True` says it runs a single time. Right for a fetch-once archive, and worth
    saying out loud, because nothing will ever bring it back."""
    @iv.data("raw/archive/", why="fetch-once history", once=True,
             external={"sports-reference": "a page that will not change"})
    def archive():
        return frame()

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(a=iv.all_of("raw/archive/", why="the archive")):
        return a

    errors, warns = check(iv)
    assert errors == [] and any("RUNS ONCE" in w for w in warns)


def test_a_root_that_re_runs_is_not_warned_about(iv):
    """A root with no `once=` runs every time — that is how anything outside the tree gets
    in — so the warning would be false."""
    @iv.data("config/model/", why="the model")
    def model():
        return frame()

    @iv.data("processed/xpm/", why="the fit", terminal=True)
    def xpm(m=iv.all_of("config/model/", why="a model change rebuilds this")):
        return m

    assert check(iv) == ([], [])


def test_an_update_read_does_not_count_as_a_trigger(iv):
    """An accumulator whose only read is its own last copy runs once and never again.

    `own_last_copy` is excluded from the comparison by design, so it cannot be the thing
    that makes a stage stale — and a stage with nothing else to read is the silent-failure
    case this check exists for.
    """
    @iv.data("raw/log/", why="a running log", terminal=True, once=True)
    def log(have=iv.own_last_copy("raw/log/", why="yesterday's copy")):
        return have if have is not None else frame()

    errors, warns = check(iv)
    assert errors == []
    assert any("RUNS ONCE" in w and "update_file_on_disk=" in w for w in warns)


def test_updating_a_dataset_another_stage_writes_is_an_error(iv):
    """The static half of the rule: caught by `iv check`, without running anything."""
    @iv.data("processed/draft/", why="the draft table", terminal=True)
    def draft(prev=iv.own_last_copy("raw/schedule/", why="the previous run's copy")):
        return prev

    @iv.data("raw/schedule/", why="the schedule")
    def fetch(d=iv.all_of("processed/draft/", why="which classes to fetch")):
        return d

    errors, _ = check(iv)
    assert any("UPDATES SOMEONE ELSE" in e and "raw/schedule/" in e for e in errors)


def test_a_consumer_defined_before_its_producer_in_one_file_is_an_error(iv):
    """Within a file, definition order IS run order, so this one is decidable."""
    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(m=iv.all_of("processed/mid/", why="the middle")):
        return m

    @iv.data("processed/mid/", why="the middle")
    def mid(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    errors, _ = check(iv)
    assert any("ORDER" in e for e in errors)


def test_definition_order_is_the_run_order(iv):
    @iv.data("processed/mid/", why="the middle")
    def mid(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(m=iv.all_of("processed/mid/", why="the middle")):
        return m

    order = _graph.build(iv).order()
    assert [n.split("::")[-1] for n in order] == ["mid", "site"]
    assert check(iv) == ([], [])


# ── two stages, one dataset ───────────────────────────────────────────────────

def test_a_second_producer_is_refused_at_declaration(iv):
    """`iv check` used to catch this after the fact. The registry catches it as the second
    stage is declared, which is earlier and says which stage it collides with."""
    @iv.data("processed/mid/", why="the middle")
    def mid():
        return frame()

    with pytest.raises(DeclError, match="already written by 'mid'"):
        @iv.data("processed/mid/", why="the middle again")
        def mid2():
            return frame()


def test_different_partitions_of_one_dataset_are_allowed(iv):
    @iv.data("processed/preds/", part={"completed": "true"}, why="played")
    def played(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    @iv.data("processed/preds/", part={"completed": "false"}, why="not yet played")
    def upcoming(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    @iv.data("dump/site/", why="the app reads it", terminal=True)
    def site(p=iv.all_of("processed/preds/", why="both halves")):
        return p

    assert check(iv) == ([], [])


# ── drift: the code against a recorded run ────────────────────────────────────

def test_an_undeclared_read_is_an_error_and_an_unseen_one_is_a_warning(iv):
    @iv.data("processed/mid/", why="the middle", terminal=True)
    def mid(feed=iv.all_of("raw/feed/", why="the feed")):
        return feed

    node = next(n for n in _graph.build(iv).stages if n.endswith("::mid"))
    events = [{"kind": "io", "op": "read", "node": node, "rel": "raw/secret/"},
              {"kind": "io", "op": "write", "node": node, "rel": "processed/mid/"}]
    errors, warns = _graph.drift(_graph.build(iv), events)
    assert any("UNDECLARED READ" in e and "raw/secret/" in e for e in errors)
    assert any("raw/feed/" in w for w in warns)


# ── preflight reads files, so these write them ────────────────────────────────

def test_preflight_catches_a_name_a_refactor_left_behind(project):
    write_stage(project, "stages/one.py", "x = undefined_thing\n")
    iv = Pipeline(root=project / "data", source_dirs=["stages"], project_root=project)
    names = _static.undefined_names(iv)
    if names is None:
        pytest.skip("pyflakes is not installed")
    assert any("undefined_thing" in n for n in names)


def test_preflight_catches_an_import_of_a_module_that_is_not_there(project):
    write_stage(project, "stages/one.py", "import stages.gone_away\n")
    iv = Pipeline(root=project / "data", source_dirs=["stages"], project_root=project)
    bad = _static.missing_imports(iv)
    assert any("gone_away" in b for b in bad), bad


def test_preflight_ignores_a_third_party_import(project):
    """An absent package is pip's problem and fails the moment anything runs. A local
    module a refactor renamed is a name that looks fine until the stage is reached."""
    write_stage(project, "stages/one.py", "import polars\nimport not_a_real_package\n")
    iv = Pipeline(root=project / "data", source_dirs=["stages"], project_root=project)
    assert _static.missing_imports(iv) == []


def test_preflight_works_when_source_dirs_names_a_file(project):
    """`source_dirs=["pipeline.py"]` is the shape the docstring recommends, and it used to
    raise TypeError: the file branch tried to file its result in a dict, copied from the
    project scan where there was one — but here the target is pyflakes' report."""
    write_stage(project, "one.py", "x = undefined_thing\n")
    iv = Pipeline(root=project / "data", source_dirs=["one.py"], project_root=project)
    names = _static.undefined_names(iv)
    if names is None:
        pytest.skip("pyflakes is not installed")
    assert any("undefined name" in n and "undefined_thing" in n for n in names)
