import os
from pathlib import Path

import polars as pl
import pytest
from iv import Pipeline
from iv.core import _check_declared, _check_write
from iv.errors import DeclError


def test_permissions_are_separate_and_stage_can_commit(tmp_path, monkeypatch):
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    (runtime / 'config').write_text('runtime')
    iv = Pipeline(tree=tmp_path/'tree', project=tmp_path, stage_dir=tmp_path/'stage',
                  allow_reads=['runtime'], allow_writes=['scratch/'])
    monkeypatch.chdir(tmp_path)

    @iv.data(dataset='result/', why='exercise runtime I/O and staged commit')
    def result():
        assert Path('runtime/config').read_text() == 'runtime'
        Path('scratch').mkdir()
        Path('scratch/cache').write_text('cache')
        with pytest.raises(DeclError, match='undeclared local read'):
            Path('scratch/cache').read_text()
        with pytest.raises(DeclError, match='undeclared local write'):
            Path('runtime/config').write_text('changed')
        with pytest.raises(DeclError, match='undeclared local read'):
            Path('runtime-neighbor/config').exists()
        return pl.DataFrame({'value': [1]})

    assert result()['value'].to_list() == [1]
    assert iv.verify("result/") == []


def test_exact_files_and_symlink_boundaries(tmp_path):
    directory = tmp_path/'runtime'
    directory.mkdir()
    outside = tmp_path/'outside'
    outside.mkdir()
    (directory/'escape').symlink_to(outside, target_is_directory=True)
    iv = Pipeline(tree=tmp_path/'tree', stage_dir=tmp_path/'stage', project=tmp_path,
                  allow_reads=[directory, 'missing-device'], allow_writes=['output'])

    @iv.data(dataset='result/', why='validate exact and recursive permission boundaries')
    def result():
        assert not (tmp_path/'missing-device').exists()
        (tmp_path/'output').write_text('ok')
        for path in (tmp_path/'missing-device/child', directory/'../outside/data',
                     directory/'escape/data'):
            with pytest.raises(DeclError, match='undeclared local read'):
                path.exists()
        with pytest.raises(DeclError, match='undeclared local write'):
            _check_write(tmp_path/'output/child')
        return pl.DataFrame({'value': [1]})

    result()


def test_permissions_do_not_override_data_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    iv = Pipeline(tree=tmp_path/'tree', out_tree=tmp_path/'outputs',
                  stage_dir=tmp_path/'stage', project=tmp_path,
                  allow_reads=[tmp_path], allow_writes=[tmp_path])

    @iv.data(dataset='result/', why='keep data tree declarations mandatory')
    def result():
        for root in (iv.tree, iv.out_tree, Path('tree')):
            with pytest.raises(DeclError, match='inside the data tree'):
                _check_declared(root/'hidden')
            with pytest.raises(DeclError, match='inside the data tree'):
                _check_write(root/'hidden')
        return pl.DataFrame({'value': [1]})

    result()


@pytest.mark.parametrize('operation', ['text', 'os_open', 'mkdir', 'parquet'])
def test_unapproved_local_writes_fail(tmp_path, operation):
    iv = Pipeline(tree=tmp_path/'tree', stage_dir=tmp_path/'stage')
    target = tmp_path/'hidden'

    @iv.data(dataset='result/', why='block side writes')
    def result():
        if operation == 'text': target.write_text('hidden')
        elif operation == 'os_open': os.close(os.open(target, os.O_CREAT | os.O_WRONLY))
        elif operation == 'mkdir': target.mkdir()
        else: pl.DataFrame({'value': [1]}).write_parquet(target)
        return pl.DataFrame({'value': [1]})

    with pytest.raises(DeclError, match='undeclared local write'):
        result()
    assert not target.exists()


@pytest.mark.parametrize('operation', ['open', 'os_open', 'path_open'])
def test_read_write_handles_require_both_permissions(tmp_path, operation):
    target = tmp_path/'cache'
    target.write_text('secret')
    iv = Pipeline(tree=tmp_path/'tree', stage_dir=tmp_path/'stage', allow_writes=[target])

    @iv.data(dataset='result/', why='read-write mode must not bypass read guard')
    def result():
        if operation == 'os_open': os.close(os.open(target, os.O_RDWR))
        elif operation == 'path_open': target.open('r+').close()
        else: open(target, 'r+').close()
        return pl.DataFrame({'value': [1]})

    with pytest.raises(DeclError, match='undeclared local read'):
        result()


@pytest.mark.parametrize('value', ['file', ['gs://bucket/key'], ['']])
def test_reject_invalid_permission_lists(tmp_path, value):
    with pytest.raises(DeclError, match='local paths'):
        Pipeline(tree=tmp_path/'tree', allow_reads=value)


def test_inactive_pipeline_does_not_grant_runtime_access(tmp_path):
    Pipeline(tree=tmp_path/'idle-tree', allow_reads=[tmp_path], allow_writes=[tmp_path])
    iv = Pipeline(tree=tmp_path/'tree', stage_dir=tmp_path/'stage')

    @iv.data(dataset='result/', why='permissions belong to the running pipeline')
    def result():
        with pytest.raises(DeclError, match='undeclared local read'):
            _check_declared(tmp_path/'hidden')
        with pytest.raises(DeclError, match='undeclared local write'):
            _check_write(tmp_path/'hidden')
        for resource in ('/dev/nvidia0', '/sys/kernel/mm/transparent_hugepage/enabled'):
            with pytest.raises(DeclError, match='undeclared local read'):
                _check_declared(resource, probe='exists')
        return pl.DataFrame({'value': [1]})

    result()


def test_guard_installation_is_idempotent(tmp_path):
    from cloudpathlib.local import LocalS3Path

    Pipeline(tree=tmp_path/'first')
    original = (Path.open, LocalS3Path.open)
    Pipeline(tree=tmp_path/'second')
    assert (Path.open, LocalS3Path.open) == original
