

from __future__ import annotations

import polars as pl
import pytest

from tyke import Pipeline
from tyke import graph as _graph
from tyke import static as _static
from tyke.errors import DeclError

from tests.conftest import write_stage


@pytest.fixture
def project(tmp_path):
    (tmp_path / "stages").mkdir()
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "d"\nversion = "0"\n')
    return tmp_path


@pytest.fixture
def tyke(tmp_path, monkeypatch):
    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def frame():
    return pl.DataFrame({"a": [1, 2]})


def check(tyke):
    return _graph.check(_graph.build(tyke))


def test_a_clean_pipeline_passes(tyke):
    feed = tyke.source("raw/feed/", why="a fetcher drops it here")
    @tyke.data(dataset="processed/mid/", why="the middle")
    def mid(feed=tyke.all_of(feed, why="the feed")):
        return feed

    @tyke.data(dataset="dump/site/", why="the app reads it")
    def site(m=tyke.all_of(mid, why="the middle")):
        return m

    assert check(tyke) == ([], [])


def test_a_root_prefix_is_how_out_of_band_data_is_declared(tyke):

    feed = tyke.source("raw/feed/", why="a fetcher drops it here")
    @tyke.data(dataset="processed/mid/", why="the middle")
    def mid(feed=tyke.all_of(feed, why="arrives out of band")):
        return feed

    assert check(tyke)[0] == []


def test_a_cycle_cannot_be_written(tyke):


    @tyke.data(dataset="processed/a/", why="a")
    def a():
        return frame()

    with pytest.raises(NameError):
        @tyke.data(dataset="processed/b/", why="b")
        def b(x=tyke.all_of(eval("later_stage"), why="not defined yet")):
            return x


def test_a_stage_with_no_inputs_is_warned_about(tyke):


    @tyke.data(dataset="raw/archive/", why="fetch-once history", once=True,
             external={"sports-reference": "a page that will not change"})
    def archive():
        return frame()

    @tyke.data(dataset="dump/site/", why="the app reads it")
    def site(a=tyke.all_of(archive, why="the archive")):
        return a

    errors, warns = check(tyke)
    assert errors == [] and any("RUNS ONCE" in w for w in warns)


def test_a_root_that_re_runs_is_not_warned_about(tyke):


    @tyke.data(dataset="config/model/", why="the model")
    def model():
        return frame()

    @tyke.data(dataset="processed/xpm/", why="the fit")
    def xpm(m=tyke.all_of(model, why="a model change rebuilds this")):
        return m

    assert check(tyke) == ([], [])


def test_an_update_read_does_not_count_as_a_trigger(tyke):


    @tyke.data(dataset="raw/log/", why="a running log", once=True)
    def log(have=tyke.own_last_copy(why="yesterday's copy")):
        return have if have is not None else frame()

    errors, warns = check(tyke)
    assert errors == []
    assert any("RUNS ONCE" in w and "update_file_on_disk=" in w for w in warns)


def test_updating_a_dataset_another_stage_writes_is_an_error(tyke):

    @tyke.data(dataset="raw/schedule/", why="the schedule")
    def fetch():
        return frame()

    @tyke.data(dataset="processed/draft/", why="the draft table")
    def draft(prev=tyke.own_last_copy(fetch, why="the previous run's copy")):
        return prev

    errors, _ = check(tyke)
    assert any("UPDATES SOMEONE ELSE" in e and "raw/schedule/" in e for e in errors)


def test_definition_order_is_the_run_order(tyke):
    feed = tyke.source("raw/feed/", why="a fetcher drops it here")
    @tyke.data(dataset="processed/mid/", why="the middle")
    def mid(feed=tyke.all_of(feed, why="the feed")):
        return feed

    @tyke.data(dataset="dump/site/", why="the app reads it")
    def site(m=tyke.all_of(mid, why="the middle")):
        return m

    order = _graph.build(tyke).order()
    assert [n.split("::")[-1] for n in order] == ["mid", "site"]
    assert check(tyke) == ([], [])


def test_a_second_producer_is_refused_at_declaration(tyke):


    @tyke.data(dataset="processed/mid/", why="the middle")
    def mid():
        return frame()

    with pytest.raises(DeclError, match="already written by 'mid'"):
        @tyke.data(dataset="processed/mid/", why="the middle again")
        def mid2():
            return frame()


def test_different_partitions_of_one_dataset_are_allowed(tyke):
    feed = tyke.source("raw/feed/", why="a fetcher drops it here")

    @tyke.data(dataset="processed/preds/", part={"completed": "true"}, why="played")
    def played(f=tyke.all_of(feed, why="the feed")):
        return f

    @tyke.data(dataset="processed/preds/", part={"completed": "false"}, why="not yet played")
    def upcoming(f=tyke.all_of(feed, why="the feed")):
        return f

    @tyke.data(dataset="dump/site/", why="the app reads it")
    def site(p=tyke.all_of(upcoming, why="both halves")):
        return p

    assert check(tyke) == ([], [])


def test_an_undeclared_read_is_an_error_and_an_unseen_one_is_a_warning(tyke):
    feed = tyke.source("raw/feed/", why="a fetcher drops it here")
    @tyke.data(dataset="processed/mid/", why="the middle")
    def mid(feed=tyke.all_of(feed, why="the feed")):
        return feed

    node = next(n for n in _graph.build(tyke).stages if n.endswith("::mid"))
    events = [{"kind": "io", "op": "read", "node": node, "rel": "raw/secret/"},
              {"kind": "io", "op": "write", "node": node, "rel": "processed/mid/"}]
    errors, warns = _graph.drift(_graph.build(tyke), events)
    assert any("UNDECLARED READ" in e and "raw/secret/" in e for e in errors)
    assert any("raw/feed/" in w for w in warns)


def test_preflight_catches_a_name_a_refactor_left_behind(project):
    write_stage(project, "stages/one.py", "x = undefined_thing\n")
    tyke = Pipeline(tree=project / "data", code=["stages"], project=project)
    names = _static.undefined_names(tyke)
    if names is None:
        pytest.skip("pyflakes is not installed")
    assert any("undefined_thing" in n for n in names)


def test_preflight_catches_an_import_of_a_module_that_is_not_there(project):
    write_stage(project, "stages/one.py", "import stages.gone_away\n")
    tyke = Pipeline(tree=project / "data", code=["stages"], project=project)
    bad = _static.missing_imports(tyke)
    assert any("gone_away" in b for b in bad), bad


def test_preflight_ignores_a_third_party_import(project):


    write_stage(project, "stages/one.py", "import polars\nimport not_a_real_package\n")
    tyke = Pipeline(tree=project / "data", code=["stages"], project=project)
    assert _static.missing_imports(tyke) == []


def test_preflight_works_when_source_dirs_names_a_file(project):


    write_stage(project, "one.py", "x = undefined_thing\n")
    tyke = Pipeline(tree=project / "data", code=["one.py"], project=project)
    names = _static.undefined_names(tyke)
    if names is None:
        pytest.skip("pyflakes is not installed")
    assert any("undefined name" in n and "undefined_thing" in n for n in names)


def test_a_declared_dataset_nothing_writes_is_warned_about(tyke):


    tyke.dataset("processed/orphan/", why="a table a rename left behind")

    @tyke.data(dataset="processed/real/", why="the one that survived")
    def real():
        return frame()

    _, warns = _graph.check(_graph.build(tyke))
    assert any("DECLARED, NOBODY WRITES  processed/orphan/" in w for w in warns)


def test_a_declared_dataset_a_stage_writes_is_not_warned_about(tyke):
    orphan = tyke.dataset("processed/kept/", why="one of two tables from one fit")

    @tyke.step(output={"a": orphan, "b": "processed/other/"}, why="the fit")
    def fit():
        return {"a": frame(), "b": frame()}

    _, warns = _graph.check(_graph.build(tyke))
    assert not any("DECLARED, NOBODY WRITES" in w for w in warns)
