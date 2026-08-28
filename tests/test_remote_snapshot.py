from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor

import pytest
from typer.testing import CliRunner

from iv.cli import app
from iv.core import Pipeline
from iv.errors import StateError
from iv.paths import local_tree_snapshot


class Remote:
    def __init__(self, files=None, rel="", events=None, fail_upload=None,
                 fail_remove=None):
        self.files = files if files is not None else {}
        self.rel = rel
        self.events = events if events is not None else []
        self.fail_upload = fail_upload
        self.fail_remove = fail_remove

    def __str__(self):
        return "gs://bucket/archive" + (f"/{self.rel}" if self.rel else "")

    def __truediv__(self, rel):
        joined = "/".join(x for x in (self.rel, str(rel)) if x)
        return Remote(self.files, joined, self.events, self.fail_upload, self.fail_remove)

    def download_to(self, destination):
        self.events.append(("download", self.rel))
        Path(destination).write_bytes(self.files[self.rel])

    def rglob(self, pattern):
        assert pattern == "*" and not self.rel
        return [self / rel for rel in self.files]

    def is_file(self):
        return self.rel in self.files

    def upload_from(self, source):
        self.events.append(("upload", self.rel))
        if self.rel == self.fail_upload:
            raise OSError("upload interrupted")
        self.files[self.rel] = Path(source).read_bytes()

    def unlink(self):
        self.events.append(("unlink", self.rel))
        if self.rel == self.fail_remove:
            raise OSError("delete interrupted")
        self.files.pop(self.rel, None)

    def exists(self):
        return not self.rel or self.rel in self.files


class BatchRemote(Remote):
    @property
    def client(self):
        outer = self

        class Client:
            def _list_dir(self, remote, recursive=False):
                assert recursive
                return [(outer / rel, False) for rel in outer.files]

        return Client()

    def rglob(self, pattern):
        raise AssertionError("the provider's batch listing should be used")


class WorkerRemote(Remote):
    def __truediv__(self, rel):
        joined = "/".join(x for x in (self.rel, str(rel)) if x)
        return WorkerRemote(self.files, joined, self.events, self.fail_upload,
                            self.fail_remove)

    def upload_from(self, source):
        self.events.append(("upload", self.rel))
        with ThreadPoolExecutor(max_workers=1) as pool:
            self.files[self.rel] = pool.submit(Path(source).read_bytes).result()


def pipeline(remote):
    state = SimpleNamespace(depth=0, publishing=0)

    @contextmanager
    def bookkeeping():
        state.depth += 1
        try:
            yield
        finally:
            state.depth -= 1

    @contextmanager
    def publication():
        state.publishing += 1
        try:
            yield
        finally:
            state.publishing -= 1

    return SimpleNamespace(tree=remote, out_tree=remote, bookkeeping=bookkeeping,
                           publication=publication, bookkeeping_state=state)


def test_one_download_then_local_execution_and_successful_publication():
    remote = Remote({"raw/old.parquet": b"old"})
    iv = pipeline(remote)

    with local_tree_snapshot(iv) as result:
        local = iv.out_tree
        assert isinstance(iv.tree, Path) and iv.tree == iv.out_tree
        (local / "raw/old.parquet").unlink()
        new = local / "raw/new.parquet"
        new.parent.mkdir(parents=True, exist_ok=True)
        new.write_bytes(b"new")

    assert result.downloaded == 1
    assert [event for event in remote.events if event[0] == "download"] == [
        ("download", "raw/old.parquet")]
    assert remote.files == {"raw/new.parquet": b"new"}
    assert remote.events[-2:] == [("upload", "raw/new.parquet"),
                                  ("unlink", "raw/old.parquet")]
    assert iv.tree is remote and iv.out_tree is remote
    assert not local.exists(), "the temporary archive should be cleaned up"


def test_remote_reads_are_scheduled_as_one_parallel_batch():
    remote = Remote({f"raw/{n}.parquet": str(n).encode() for n in range(8)})
    messages = []
    with local_tree_snapshot(pipeline(remote), report=messages.append) as result:
        pass

    assert result.files_downloaded == 8
    assert any(message.startswith("remote snapshot · local directory ")
               for message in messages)
    assert "remote snapshot · 8 file(s) · 8 workers · parallel downloader" in messages
    assert "remote snapshot · downloaded 8/8 file(s)" in messages


def test_provider_listing_avoids_one_metadata_probe_per_file():
    remote = BatchRemote({f"raw/{n}.parquet": b"x" for n in range(4)})
    with local_tree_snapshot(pipeline(remote)) as result:
        pass
    assert result.files_downloaded == 4


def test_gcs_uses_the_supported_transfer_manager_for_small_file_batch(monkeypatch):
    remote = Remote({"raw/a.parquet": b"a", "raw/b.parquet": b"b"})
    calls = []

    class TransferManager:
        THREAD = "thread"

        @staticmethod
        def download_many_to_path(bucket, names, **kw):
            calls.append((bucket, names, kw))
            destination = Path(kw["destination_directory"])
            for name in names:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(remote.files[name])

    import iv.paths as paths
    monkeypatch.setattr(paths, "_gcs_transfer",
                        lambda root: (TransferManager, "bucket", "archive/"))
    with local_tree_snapshot(pipeline(remote)) as result:
        pass

    assert result.files_downloaded == 2
    assert calls == [("bucket", ["raw/a.parquet", "raw/b.parquet"], {
        "destination_directory": str(calls[0][2]["destination_directory"]),
        "blob_name_prefix": "archive/", "worker_type": "thread",
        "max_workers": 2, "raise_exception": True,
    })]


def test_a_failed_run_never_mutates_remote_storage():
    remote = Remote({"raw/old.parquet": b"old"})
    iv = pipeline(remote)

    with pytest.raises(RuntimeError, match="stage failed"):
        with local_tree_snapshot(iv):
            (iv.out_tree / "raw/new.parquet").write_bytes(b"new")
            raise RuntimeError("stage failed")

    assert remote.files == {"raw/old.parquet": b"old"}
    assert remote.events == [("download", "raw/old.parquet")]
    assert iv.tree is remote and iv.out_tree is remote


def test_a_local_dev_output_is_never_published_to_the_remote_tree(tmp_path):
    remote = Remote({"raw/input.parquet": b"input"})
    iv = pipeline(remote)
    dev = tmp_path / "evaluation"
    from iv.cli import _dev_output

    with _dev_output(iv, dev):
        with local_tree_snapshot(iv):
            target = iv.out_tree / "processed/model.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"model")

    assert (dev / "processed/model.parquet").read_bytes() == b"model"
    assert remote.files == {"raw/input.parquet": b"input"}
    assert not any(op in {"upload", "unlink"} for op, _ in remote.events)
    assert iv.out_tree is remote


def test_a_failed_run_publishes_only_its_last_completed_stage():
    remote = Remote({"raw/old.parquet": b"old"})
    iv = pipeline(remote)

    with pytest.raises(RuntimeError, match="later stage failed"):
        with local_tree_snapshot(iv) as result:
            (iv.out_tree / "raw/old.parquet").unlink()
            completed = iv.out_tree / "derived/completed.parquet"
            completed.parent.mkdir(parents=True, exist_ok=True)
            completed.write_bytes(b"completed")
            iv._remote_checkpoint()

            completed.unlink()
            partial = iv.out_tree / "derived/partial.parquet"
            partial.write_bytes(b"partial")
            raise RuntimeError("later stage failed")

    assert result.partial
    assert remote.files == {"derived/completed.parquet": b"completed"}
    assert "derived/partial.parquet" not in remote.files


@pytest.mark.parametrize("fail_after_checkpoint", [False, True])
def test_remote_publication_runs_inside_cross_thread_scope(monkeypatch,
                                                           fail_after_checkpoint):
    remote = Remote({"raw/old.parquet": b"old"})
    iv = pipeline(remote)
    depths = []
    import iv.paths as paths
    real = paths._publish

    def checked(*args, **kwargs):
        depths.append(iv.bookkeeping_state.publishing)
        return real(*args, **kwargs)

    monkeypatch.setattr(paths, "_publish", checked)
    error = pytest.raises(RuntimeError) if fail_after_checkpoint else nullcontext()
    with error:
        with local_tree_snapshot(iv):
            target = iv.out_tree / "derived/completed.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"completed")
            iv._remote_checkpoint()
            if fail_after_checkpoint:
                raise RuntimeError("later stage failed")

    assert depths == [1]


def test_publication_allows_transfer_workers_to_read_the_local_mirror(tmp_path):
    remote = WorkerRemote({"raw/old.parquet": b"old"})
    iv = Pipeline(tree=remote, project=tmp_path, stage_dir=tmp_path / "stage")

    with local_tree_snapshot(iv):
        target = iv.out_tree / "derived/new.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"new")

    assert remote.files["derived/new.parquet"] == b"new"


def test_an_interrupted_upload_rolls_back_additions_before_any_removal():
    remote = Remote({"raw/old.parquet": b"old"}, fail_upload="raw/z.parquet")
    iv = pipeline(remote)

    with pytest.raises(StateError, match="Existing remote shards were not removed"):
        with local_tree_snapshot(iv):
            (iv.out_tree / "raw/old.parquet").unlink()
            for name in ("a.parquet", "z.parquet"):
                target = iv.out_tree / "raw" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(name.encode())

    assert remote.files == {"raw/old.parquet": b"old"}
    assert ("unlink", "raw/old.parquet") not in remote.events
    assert ("unlink", "raw/a.parquet") in remote.events


def test_iv_run_enters_the_local_snapshot_before_planning(monkeypatch):
    remote = Remote({"raw/old.parquet": b"old"})
    iv = pipeline(remote)
    seen = []

    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)

    def run_local(got, *args):
        seen.append((got.tree, got.out_tree))

    monkeypatch.setattr(cli, "_run_local", run_local)
    result = CliRunner().invoke(app, ["run"])
    assert result.exit_code == 0, result.output
    assert len(seen) == 1 and all(isinstance(root, Path) for root in seen[0])
    assert remote.events == [("download", "raw/old.parquet")]
    assert "remote snapshot · listing gs://bucket/archive" in result.output
    assert "remote snapshot ready · 1 file(s)" in result.output
    assert "remote publish complete" in result.output
    assert "remote snapshot · local cleanup complete" in result.output


def test_iv_fetch_downloads_remote_state_to_a_new_local_directory(tmp_path,
                                                                  monkeypatch):
    remote = Remote({"raw/a.parquet": b"a", "derived/b.json": b"b"})
    iv = pipeline(remote)
    destination = tmp_path / "state"
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)

    result = CliRunner().invoke(app, ["fetch", str(destination)])
    assert result.exit_code == 0, result.output
    assert (destination / "raw/a.parquet").read_bytes() == b"a"
    assert (destination / "derived/b.json").read_bytes() == b"b"
    assert f"remote fetch complete · 2 file(s) · {destination}" in result.output
    assert not any(op in {"upload", "unlink"} for op, _ in remote.events)


def test_iv_fetch_refuses_to_merge_over_an_existing_directory(tmp_path, monkeypatch):
    remote = Remote({"raw/a.parquet": b"remote"})
    iv = pipeline(remote)
    destination = tmp_path / "state"
    destination.mkdir()
    existing = destination / "notes.txt"
    existing.write_text("mine")
    import iv.cli as cli
    monkeypatch.setattr(cli, "_load", lambda: iv)

    result = CliRunner().invoke(app, ["fetch", str(destination)])
    assert result.exit_code == 1
    assert "destination already exists" in result.output
    assert existing.read_text() == "mine"


def test_iv_fetch_cleans_up_an_interrupted_staging_directory(tmp_path, monkeypatch):
    remote = Remote({"raw/a.parquet": b"a"})
    destination = tmp_path / "state"
    import iv.paths as paths

    def interrupted(source, staged, report=None):
        staged.mkdir(parents=True)
        (staged / "partial").write_text("partial")
        raise StateError("download interrupted")

    monkeypatch.setattr(paths, "_download", interrupted)
    with pytest.raises(StateError, match="download interrupted"):
        paths.fetch_tree(remote, destination)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
