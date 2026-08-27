from __future__ import annotations

import polars as pl
import pytest

from tyke.core import Pipeline
from tyke.errors import DeclError


@pytest.fixture
def tyke(tmp_path, monkeypatch):
    for var in ("TYKE_TRACE", "TYKE_FORCE", "TYKE_STAGE"):
        monkeypatch.delenv(var, raising=False)
    return Pipeline(tree=tmp_path / "data", stage_dir=tmp_path / "stage",
                    project=tmp_path)


def frame() -> pl.DataFrame:
    return pl.DataFrame({"a": [1, 2, 3]})


def seeded(tyke):
    @tyke.data(dataset="raw/feed/", why="the feed")
    def feed():
        return frame()
    feed()
    return tyke.resolve_out("raw/feed/")


@pytest.mark.parametrize("probe", ["exists", "is_file", "is_dir"])
def test_a_probe_of_the_tree_inside_a_stage_is_refused(tyke, probe):
    d = seeded(tyke)

    @tyke.data(dataset="processed/out/", why="probes behind tyke's back")
    def out():
        getattr(d, probe)()
        return frame()

    with pytest.raises(DeclError, match="asks the data tree a question"):
        out()


@pytest.mark.parametrize("verb", ["iterdir", "glob", "rglob"])
def test_listing_the_tree_inside_a_stage_is_refused(tyke, verb):
    d = seeded(tyke)

    @tyke.data(dataset="processed/out/", why="lists behind tyke's back")
    def out():
        list(getattr(d, verb)("*") if verb != "iterdir" else d.iterdir())
        return frame()

    with pytest.raises(DeclError, match="asks the data tree a question"):
        out()


def test_a_probe_outside_a_stage_is_allowed(tyke):
    d = seeded(tyke)
    assert d.exists()
    assert list(d.iterdir())


def test_a_missing_directory_cannot_answer_a_stage_with_silence(tyke):
    gone = tyke.resolve_out("raw/never_built/")

    @tyke.data(dataset="processed/out/", why="branches on a directory that is not there")
    def out():
        return frame() if list(gone.glob("*.parquet")) else pl.DataFrame()

    with pytest.raises(DeclError, match="asks the data tree a question"):
        out()


def test_a_declared_read_is_still_allowed_inside_a_stage(tyke):
    seeded(tyke)

    @tyke.data(dataset="processed/out/", why="reads what it declares")
    def out(feed=None):
        return pl.read_parquet(tyke.reads("raw/feed/", why="declared"))

    assert out().height == 3


def test_writing_a_frame_into_the_tree_inside_a_stage_is_refused(tyke):
    target = tyke.resolve_out("raw/feed/") / "smuggled.parquet"

    @tyke.data(dataset="processed/out/", why="writes a parquet behind tyke's back")
    def out():
        frame().write_parquet(str(target))
        return frame()

    with pytest.raises(DeclError, match="outside tyke.writes"):
        out()


@pytest.mark.parametrize("verb", ["mkdir", "touch"])
def test_creating_a_path_in_the_tree_inside_a_stage_is_refused(tyke, verb):
    target = tyke.resolve_out("raw/feed/") / "smuggled"

    @tyke.data(dataset="processed/out/", why="creates a path behind tyke's back")
    def out():
        getattr(target, verb)()
        return frame()

    with pytest.raises(DeclError, match="outside tyke.writes"):
        out()


def test_removing_a_handed_back_shard_outside_a_stage_is_allowed(tyke):
    seeded(tyke)
    shard = tyke.reads("raw/feed/", why="declared")[0]
    shard.unlink()
    assert not list(tyke.resolve_out("raw/feed/").iterdir())


def test_removing_a_tree_path_inside_a_stage_is_refused(tyke):
    seeded(tyke)
    shard = tyke.reads("raw/feed/", why="declared")[0]

    @tyke.data(dataset="processed/out/", why="deletes behind tyke's back")
    def out():
        shard.unlink()
        return frame()

    with pytest.raises(DeclError, match="outside tyke.writes"):
        out()


def test_a_cloudpath_is_named_by_its_uri_not_its_local_mirror():
    from cloudpathlib.local import LocalS3Path

    from tyke.core import _path_text

    p = LocalS3Path("s3://bucket/some/key.parquet")
    assert _path_text(p) == "s3://bucket/some/key.parquet"


def test_a_concrete_cloudpath_subclass_is_patched(tyke):
    from cloudpathlib.local import LocalS3Path

    assert getattr(LocalS3Path.exists, "_iv_read_checked", False)
    assert getattr(LocalS3Path.mkdir, "_iv_checked", False)


def test_remote_storage_outside_every_tree_is_refused_inside_a_stage(tyke):
    from cloudpathlib.local import LocalS3Path

    elsewhere = LocalS3Path("s3://somewhere-else/feed.parquet")

    @tyke.data(dataset="processed/out/", why="reaches a bucket nobody declared")
    def out():
        elsewhere.exists()
        return frame()

    with pytest.raises(DeclError, match="outside every declared tree"):
        out()


def test_remote_storage_a_stage_declares_as_external_is_allowed(tyke):
    from cloudpathlib.local import LocalS3Path

    dest = LocalS3Path("s3://declared-bucket/out.json")

    @tyke.data(dataset="processed/out/", why="ships a file out of the pipeline",
             external={"s3://declared-bucket": "where the app reads it at runtime"})
    def out():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}")
        return frame()

    assert out().height == 3
    assert dest.read_text() == "{}"


def test_an_external_declared_by_another_stage_does_not_carry_over(tyke):
    from cloudpathlib.local import LocalS3Path

    dest = LocalS3Path("s3://declared-bucket/out.json")

    @tyke.data(dataset="processed/first/", why="declares the bucket",
             external={"s3://declared-bucket": "where the app reads it"})
    def first():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}")
        return frame()

    @tyke.data(dataset="processed/second/", why="declares nothing")
    def second():
        dest.write_text("{}")
        return frame()

    first()
    with pytest.raises(DeclError, match="outside every declared tree"):
        second()
