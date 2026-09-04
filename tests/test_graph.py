

from __future__ import annotations

import polars as pl
import pytest

from iv import Pipeline
from iv import graph as _graph
from iv import static as _static
from iv import viz as _viz
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
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                       project=tmp_path)


def frame():
    return pl.DataFrame({"a": [1, 2]})


def check(iv):
    return _graph.check(_graph.build(iv))


def test_a_clean_pipeline_passes(iv):
    feed = iv.source("raw/feed/", why="a fetcher drops it here")
    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(feed=iv.all_of(feed, why="the feed")):
        return feed

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(m=iv.all_of(mid, why="the middle")):
        return m

    assert check(iv) == ([], [])


def test_a_root_prefix_is_how_out_of_band_data_is_declared(iv):

    feed = iv.source("raw/feed/", why="a fetcher drops it here")
    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(feed=iv.all_of(feed, why="arrives out of band")):
        return feed

    assert check(iv)[0] == []


def test_a_cycle_cannot_be_written(iv):


    @iv.data(dataset="processed/a/", why="a")
    def a():
        return frame()

    with pytest.raises(NameError):
        @iv.data(dataset="processed/b/", why="b")
        def b(x=iv.all_of(eval("later_stage"), why="not defined yet")):
            return x


def test_a_stage_with_no_inputs_is_warned_about(iv):


    @iv.data(dataset="raw/archive/", why="fetch-once history", once=True,
             external={"sports-reference": "a page that will not change"})
    def archive():
        return frame()

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(a=iv.all_of(archive, why="the archive")):
        return a

    errors, warns = check(iv)
    assert errors == [] and any("RUNS ONCE" in w for w in warns)


def test_a_root_that_re_runs_is_not_warned_about(iv):


    @iv.data(dataset="config/model/", why="the model")
    def model():
        return frame()

    @iv.data(dataset="processed/xpm/", why="the fit")
    def xpm(m=iv.all_of(model, why="a model change rebuilds this")):
        return m

    assert check(iv) == ([], [])


def test_an_update_read_does_not_count_as_a_trigger(iv):


    @iv.data(dataset="raw/log/", why="a running log", once=True)
    def log(have=iv.own_last_copy(why="yesterday's copy")):
        return have if have is not None else frame()

    errors, warns = check(iv)
    assert errors == []
    assert any("RUNS ONCE" in w and "update_file_on_disk=" in w for w in warns)


def test_updating_a_dataset_another_stage_writes_is_an_error(iv):

    @iv.data(dataset="raw/schedule/", why="the schedule")
    def fetch():
        return frame()

    @iv.data(dataset="processed/draft/", why="the draft table")
    def draft(prev=iv.own_last_copy(fetch, why="the previous run's copy")):
        return prev

    errors, _ = check(iv)
    assert any("UPDATES SOMEONE ELSE" in e and "raw/schedule/" in e for e in errors)


def test_definition_order_is_the_run_order(iv):
    feed = iv.source("raw/feed/", why="a fetcher drops it here")
    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(feed=iv.all_of(feed, why="the feed")):
        return feed

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(m=iv.all_of(mid, why="the middle")):
        return m

    order = _graph.build(iv).order()
    assert [n.split("::")[-1] for n in order] == ["mid", "site"]
    assert check(iv) == ([], [])


def test_a_second_producer_is_refused_at_declaration(iv):


    @iv.data(dataset="processed/mid/", why="the middle")
    def mid():
        return frame()

    with pytest.raises(DeclError, match="already written by 'mid'"):
        @iv.data(dataset="processed/mid/", why="the middle again")
        def mid2():
            return frame()


def test_different_partitions_of_one_dataset_are_allowed(iv):
    feed = iv.source("raw/feed/", why="a fetcher drops it here")

    @iv.data(dataset="processed/preds/", part={"completed": "true"}, why="played")
    def played(f=iv.all_of(feed, why="the feed")):
        return f

    @iv.data(dataset="processed/preds/", part={"completed": "false"}, why="not yet played")
    def upcoming(f=iv.all_of(feed, why="the feed")):
        return f

    @iv.data(dataset="dump/site/", why="the app reads it")
    def site(p=iv.all_of(upcoming, why="both halves")):
        return p

    assert check(iv) == ([], [])


def test_graph_edges_retain_the_read_rule(iv):
    feed = iv.source("raw/feed/", why="the feed")

    @iv.data(dataset="processed/mid/", part="season", why="the middle")
    def mid(season, f=iv.same_part(feed, why="the matching season")):
        return f

    @iv.data(dataset="processed/end/", part="season", why="the end")
    def end(m=iv.before_part(mid, why="prior seasons only")):
        return m

    g = _graph.build(iv)
    edges = [(u, v, d["rule"]) for u, v, d in _viz.to_networkx(g).edges(data=True)]
    assert {rule for _, _, rule in edges} == {"same_part", "before_part"}


def test_an_undeclared_read_is_an_error_and_an_unseen_one_is_a_warning(iv):
    feed = iv.source("raw/feed/", why="a fetcher drops it here")
    @iv.data(dataset="processed/mid/", why="the middle")
    def mid(feed=iv.all_of(feed, why="the feed")):
        return feed

    node = next(n for n in _graph.build(iv).stages if n.endswith("::mid"))
    events = [{"kind": "io", "op": "read", "node": node, "rel": "raw/secret/"},
              {"kind": "io", "op": "write", "node": node, "rel": "processed/mid/"}]
    errors, warns = _graph.drift(_graph.build(iv), events)
    assert any("UNDECLARED READ" in e and "raw/secret/" in e for e in errors)
    assert any("raw/feed/" in w for w in warns)


def test_preflight_catches_a_name_a_refactor_left_behind(project):
    write_stage(project, "stages/one.py", "x = undefined_thing\n")
    iv = Pipeline(tree=project / "data", code=["stages"], project=project)
    names = _static.undefined_names(iv)
    if names is None:
        pytest.skip("pyflakes is not installed")
    assert any("undefined_thing" in n for n in names)


def test_preflight_catches_an_import_of_a_module_that_is_not_there(project):
    write_stage(project, "stages/one.py", "import stages.gone_away\n")
    iv = Pipeline(tree=project / "data", code=["stages"], project=project)
    bad = _static.missing_imports(iv)
    assert any("gone_away" in b for b in bad), bad


def test_preflight_ignores_a_third_party_import(project):


    write_stage(project, "stages/one.py", "import polars\nimport not_a_real_package\n")
    iv = Pipeline(tree=project / "data", code=["stages"], project=project)
    assert _static.missing_imports(iv) == []


def test_preflight_works_when_source_dirs_names_a_file(project):


    write_stage(project, "one.py", "x = undefined_thing\n")
    iv = Pipeline(tree=project / "data", code=["one.py"], project=project)
    names = _static.undefined_names(iv)
    if names is None:
        pytest.skip("pyflakes is not installed")
    assert any("undefined name" in n and "undefined_thing" in n for n in names)


def test_a_declared_dataset_nothing_writes_is_warned_about(iv):


    iv.dataset("processed/orphan/", why="a table a rename left behind")

    @iv.data(dataset="processed/real/", why="the one that survived")
    def real():
        return frame()

    _, warns = _graph.check(_graph.build(iv))
    assert any("DECLARED, NOBODY WRITES  processed/orphan/" in w for w in warns)


def test_a_declared_dataset_a_stage_writes_is_not_warned_about(iv):
    orphan = iv.dataset("processed/kept/", why="one of two tables from one fit")

    @iv.step(output={"a": orphan, "b": "processed/other/"}, why="the fit")
    def fit():
        return {"a": frame(), "b": frame()}

    _, warns = _graph.check(_graph.build(iv))
    assert not any("DECLARED, NOBODY WRITES" in w for w in warns)
