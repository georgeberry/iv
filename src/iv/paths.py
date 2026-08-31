from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError, StateError


def mkpath(spec, project: Path | None):
    if not isinstance(spec, str):
        return spec
    if "://" not in spec:
        p = Path(spec)
        return p if p.is_absolute() or project is None else project / p
    try:
        from cloudpathlib import AnyPath
    except ImportError as e:
        raise ConfigError(
            f"root {spec!r} is a URI, which needs cloudpathlib installed "
            f"(pip install 'cloudpathlib[gs]')") from e
    return AnyPath(spec)


def is_remote(path) -> bool:
    return "://" in str(path) and hasattr(path, "download_to")


def _manifest(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


@dataclass
class RemoteSnapshot:
    downloaded: int = 0
    files_downloaded: int = 0
    bytes_downloaded: int = 0
    uploaded: int = 0
    removed: int = 0
    partial: bool = False


def _say(report, message: str) -> None:
    if report is not None:
        report(message)


def _remote_files(remote) -> list[tuple[str, object]]:
    prefix = str(remote).rstrip("/") + "/"
    listing = getattr(getattr(remote, "client", None), "_list_dir", None)
    if listing is not None:
        paths = (path for path, is_dir in listing(remote, recursive=True) if not is_dir)
    else:
        paths = (path for path in remote.rglob("*") if path.is_file())
    return sorted(
        ((str(path)[len(prefix):], path) for path in paths),
        key=lambda item: item[0],
    )


def _download_workers(count: int) -> int:
    raw = os.environ.get("IV_DOWNLOAD_WORKERS", "64")
    try:
        configured = int(raw)
    except ValueError as e:
        raise ConfigError(f"IV_DOWNLOAD_WORKERS must be an integer, got {raw!r}.") from e
    if configured < 1:
        raise ConfigError(f"IV_DOWNLOAD_WORKERS must be positive, got {configured}.")
    return min(configured, max(1, count))


def _gcs_transfer(remote):
    if not str(remote).startswith("gs://"):
        return None
    try:
        from google.cloud.storage import transfer_manager
        client = remote.client.client
        bucket = client.bucket(remote.bucket)
    except (ImportError, AttributeError):
        return None
    prefix = remote.blob.rstrip("/")
    return transfer_manager, bucket, (prefix + "/" if prefix else "")


def _download(remote, local: Path, report=None) -> set[str]:
    local.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _say(report, f"remote snapshot · listing {remote}")
    try:
        files = _remote_files(remote)
    except Exception as e:
        try:
            missing = not remote.exists()
        except Exception:
            missing = False
        if not missing:
            raise StateError(f"could not snapshot remote data tree {remote}: {e}") from e
        files = []
    workers = _download_workers(len(files))
    gcs = _gcs_transfer(remote)
    engine = "GCS transfer manager" if gcs is not None else "parallel downloader"
    _say(report, f"remote snapshot · {len(files)} file(s) · {workers} workers · {engine}")

    def fetch(item):
        rel, source = item
        target = local / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        source.download_to(target)
        return target.stat().st_size

    done = total_bytes = 0
    try:
        if gcs is not None and files:
            transfer_manager, bucket, prefix = gcs
            transfer_manager.download_many_to_path(
                bucket, [rel for rel, _ in files], destination_directory=str(local),
                blob_name_prefix=prefix, worker_type=transfer_manager.THREAD,
                max_workers=workers, raise_exception=True)
            done = len(files)
            total_bytes = sum((local / rel).stat().st_size for rel, _ in files)
            _say(report, f"remote snapshot · downloaded {done}/{len(files)} file(s)")
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="iv-download") as pool:
                futures = [pool.submit(fetch, item) for item in files]
                for future in as_completed(futures):
                    total_bytes += future.result()
                    done += 1
                    if done == len(files) or done % 100 == 0:
                        _say(report, f"remote snapshot · downloaded {done}/{len(files)} file(s)")
    except Exception as e:
        raise StateError(f"could not snapshot remote data tree {remote}: {e}") from e
    _say(report, f"remote snapshot ready · {done} file(s), {total_bytes / 1_000_000:.1f} MB "
                 f"in {time.perf_counter() - started:.2f}s")
    return {rel for rel, _ in files}


def fetch_tree(remote, destination: Path, report=None, *, replace: bool = False) -> set[str]:
    if not is_remote(remote):
        raise ConfigError(f"iv fetch needs a remote pipeline tree, got {remote}.")
    target = destination.expanduser().resolve()
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ConfigError(
            f"iv fetch destination must be a directory, not a file or symlink: {target}.")
    if target.exists() and not replace:
        raise ConfigError(
            f"iv fetch destination already exists: {target}. Pass --replace to replace "
            "the complete directory after a successful download.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.iv-fetch-",
                                     dir=target.parent) as tmp:
        staged = Path(tmp) / "tree"
        files = _download(remote, staged, report)
        previous = Path(tmp) / "previous"
        if target.exists():
            _say(report, f"remote fetch · replacing existing directory {target}")
            target.rename(previous)
        try:
            staged.rename(target)
        except BaseException:
            if previous.exists() and not target.exists():
                previous.rename(target)
            raise
    _say(report, f"remote fetch complete · {len(files)} file(s) · {target}")
    return files


def _publish(remote, local: Path, before: set[str], result: RemoteSnapshot,
             report=None, after: set[str] | None = None) -> None:
    after = _manifest(local) if after is None else after
    additions = sorted(after - before)
    removals = sorted(before - after)
    _say(report, f"remote publish · {len(additions)} upload(s), {len(removals)} removal(s)")
    uploaded = []
    try:
        gcs = _gcs_transfer(remote)
        if gcs is not None and additions:
            transfer_manager, bucket, prefix = gcs
            workers = _download_workers(len(additions))
            transfer_manager.upload_many_from_filenames(
                bucket, additions, source_directory=str(local), blob_name_prefix=prefix,
                worker_type=transfer_manager.THREAD, max_workers=workers,
                raise_exception=True)
            uploaded = [remote / rel for rel in additions]
            result.uploaded = len(uploaded)
            _say(report, f"remote publish · uploaded {result.uploaded}/{len(additions)}")
        else:
            for rel in additions:
                target = remote / rel
                target.upload_from(local / rel)
                uploaded.append(target)
                result.uploaded += 1
                if result.uploaded == len(additions) or result.uploaded % 100 == 0:
                    _say(report, f"remote publish · uploaded {result.uploaded}/{len(additions)}")
    except Exception as e:
        for target in reversed(uploaded or [remote / rel for rel in additions]):
            try:
                target.unlink()
            except Exception:
                pass
        raise StateError(
            f"could not publish local run to {remote}: {e}. Existing remote shards "
            "were not removed; run `iv gc` if the failed upload left a new shard behind.") from e
    for rel in removals:
        try:
            (remote / rel).unlink()
            result.removed += 1
        except Exception as e:
            raise StateError(
                f"uploaded the completed run but could not remove superseded remote shard "
                f"{remote / rel}: {e}. No data was lost; run `iv gc` to finish cleanup.") from e
    _say(report, f"remote publish complete · {result.uploaded} uploaded, "
                 f"{result.removed} superseded removed")


@contextmanager
def local_tree_snapshot(iv, report=None):
    """Run against local copies of remote roots and publish output on success."""
    remote_tree = iv.tree if is_remote(iv.tree) else None
    remote_out = iv.out_tree if is_remote(iv.out_tree) else None
    result = RemoteSnapshot()
    if remote_tree is None and remote_out is None:
        yield result
        return

    original_tree, original_out = iv.tree, iv.out_tree
    with tempfile.TemporaryDirectory(prefix="iv-remote-") as tmp:
        root = Path(tmp)
        _say(report, f"remote snapshot · local directory {root}")
        local_tree = root / "tree"
        same_root = remote_tree is not None and str(original_tree) == str(original_out)
        before_tree = (_download(remote_tree, local_tree, report)
                       if remote_tree is not None else set())
        if remote_tree is not None:
            result.downloaded += 1
            result.files_downloaded += len(before_tree)
            result.bytes_downloaded += sum((local_tree / rel).stat().st_size
                                           for rel in before_tree)
        if same_root:
            local_out, before_out = local_tree, before_tree
        elif remote_out is not None:
            local_out = root / "out"
            before_out = _download(remote_out, local_out, report)
            result.downloaded += 1
            result.files_downloaded += len(before_out)
            result.bytes_downloaded += sum((local_out / rel).stat().st_size
                                           for rel in before_out)
        else:
            local_out, before_out = original_out, set()
        iv.tree = local_tree if remote_tree is not None else original_tree
        iv.out_tree = local_out
        checkpoint_dir = root / "checkpoint"
        checkpoint: set[str] | None = None
        previous_checkpoint = getattr(iv, "_remote_checkpoint", None)

        def save_checkpoint() -> None:
            nonlocal checkpoint
            checkpoint = _manifest(local_out)
            for rel in checkpoint - before_out:
                source, target = local_out / rel, checkpoint_dir / rel
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)

        iv._remote_checkpoint = save_checkpoint
        try:
            yield result
        except BaseException:
            if remote_out is not None and checkpoint is not None and checkpoint != before_out:
                result.partial = True
                _say(report, "remote publish · preserving completed stages after failure")
                with iv.publication():
                    _publish(remote_out, checkpoint_dir, before_out, result, report,
                             after=checkpoint)
            raise
        else:
            if remote_out is not None:
                with iv.publication():
                    _publish(remote_out, local_out, before_out, result, report)
        finally:
            if previous_checkpoint is None:
                del iv._remote_checkpoint
            else:
                iv._remote_checkpoint = previous_checkpoint
            iv.tree, iv.out_tree = original_tree, original_out
            _say(report, "remote snapshot · local cleanup complete")
