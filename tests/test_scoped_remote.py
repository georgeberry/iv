from pathlib import Path
import polars as pl
import pytest
from typer.testing import CliRunner
from iv import Pipeline, cli, paths
from iv.errors import IvError
from test_remote_snapshot import Remote, pipeline

@pytest.mark.parametrize('fail', [False, True])
def test_scoped_publish_never_deletes_omitted_files(fail):
    remote = Remote({'wanted/old.json': b'old', 'large/source.json': b'unrelated'})
    iv = pipeline(remote)
    try:
        with paths.local_tree_snapshot(iv, include=('wanted/',)):
            assert not (iv.tree/'large').exists()
            (iv.out_tree/'wanted/old.json').unlink()
            (iv.out_tree/'wanted/new.json').write_bytes(b'new')
            iv._remote_checkpoint()
            if fail:
                raise RuntimeError('later stage failed')
    except RuntimeError:
        assert fail
    assert remote.files == {'wanted/new.json': b'new', 'large/source.json': b'unrelated'}
    assert ('download', 'large/source.json') not in remote.events
    assert ('unlink', 'large/source.json') not in remote.events


@pytest.mark.parametrize('force', [False, True])
def test_from_downloads_boundary_inputs_and_selected_outputs(tmp_path, monkeypatch, force):
    iv = Pipeline(tree=tmp_path/'seed', project=tmp_path, stage_dir=tmp_path/'stage')
    @iv.data(dataset='raw/', why='raw', ext='.json')
    def raw(): return {'v': 1}
    @iv.data(dataset='parent/', why='parent', ext='.json')
    def parent(value=iv.all_of(raw, why='raw')): return value
    @iv.data(dataset='selected/', why='selected', ext='.json')
    def selected(value=iv.all_of(parent, why='parent')): return value
    @iv.data(dataset='child/', why='child', ext='.json')
    def child(value=iv.all_of(selected, why='selected')): return value
    raw(); parent(); selected(); child()
    remote=MetadataRemote({p.relative_to(iv.tree).as_posix():p.read_bytes() for ds in ['raw/','parent/','selected/','child/'] for p in iv.reads(ds,why='remote fixture')})
    remote.files['unrelated/huge.json']=b'unrelated'
    iv.tree=iv.out_tree=remote
    monkeypatch.setattr(cli,'_load',lambda:iv)
    result=CliRunner().invoke(cli.app,['run','--from','selected']+(['--force'] if force else []))
    assert result.exit_code==0, result.output
    downloaded={rel.split('/')[0] for op,rel in remote.events if op=='download'}
    assert downloaded=={'parent','selected','child'}
    assert 'unrelated/huge.json' in remote.files
    assert any(k.startswith('raw/') for k in remote.files)


def test_remote_upstream_checked_before_partial_download(tmp_path, monkeypatch):
    iv=Pipeline(tree=Remote(),project=tmp_path,stage_dir=tmp_path/'stage')
    @iv.data(dataset='output/',why='output',ext='.json')
    def output():return {}
    monkeypatch.setattr(cli,'_load',lambda:iv)
    def reject(iv,g,selected,filters,target):
        assert paths.is_remote(iv.tree)
        assert iv.tree.events==[]
        raise IvError('stale upstream')
    monkeypatch.setattr(cli,'_check_upstream',reject)
    result=CliRunner().invoke(cli.app,['run','--only','output'])
    assert result.exit_code!=0
    assert 'stale upstream' in result.output
    assert iv.tree.events==[]


class MetadataRemote(Remote):
    @property
    def name(self): return self.rel.rsplit('/', 1)[-1]
    @property
    def suffix(self): return Path(self.name).suffix
    def __truediv__(self, rel):
        return MetadataRemote(self.files, '/'.join(x for x in (self.rel,str(rel).strip('/')) if x), self.events)
    def iterdir(self):
        prefix=self.rel.rstrip('/')+'/'
        return [self / k[len(prefix):] for k in self.files if k.startswith(prefix) and '/' not in k[len(prefix):]]
