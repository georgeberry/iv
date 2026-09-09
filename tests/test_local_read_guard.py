import os
from pathlib import Path

import polars as pl
import pytest
from iv import Pipeline
from iv.errors import DeclError


@pytest.mark.parametrize('operation', ['bytes', 'text', 'parquet', 'exists', 'glob', 'relative', 'os_exists', 'os_listdir'])
def test_local_data_outside_tree_requires_a_declared_source(tmp_path, monkeypatch, operation):
    iv = Pipeline(tree=tmp_path/'tree', stage_dir=tmp_path/'stage', project=tmp_path)
    directory=tmp_path/'manual_data'
    directory.mkdir()
    source=directory/'curated.parquet'
    pl.DataFrame({'value':[42]}).write_parquet(source)
    monkeypatch.chdir(tmp_path)

    @iv.data(dataset='derived/result/', why='attempt undeclared manual data')
    def result():
        if operation == 'bytes': source.read_bytes()
        elif operation == 'text': source.read_text()
        elif operation == 'parquet': pl.read_parquet(source)
        elif operation == 'exists': (directory/'absent.json').exists()
        elif operation == 'glob': list(directory.glob('*.parquet'))
        elif operation == 'relative': Path('manual_data/curated.parquet').read_bytes()
        elif operation == 'os_exists': os.path.exists(directory/'absent.json')
        elif operation == 'os_listdir': os.listdir(directory)
        return pl.DataFrame({'value':[1]})

    with pytest.raises(DeclError, match='undeclared local read'):
        result()


def test_importing_the_same_curated_values_as_a_source_is_valid(tmp_path):
    iv=Pipeline(tree=tmp_path/'tree', stage_dir=tmp_path/'stage', project=tmp_path)
    source=iv.source('raw/curated/', why='reviewed manual values')
    with iv.writes(source.dataset, why='one-time source import') as out:
        pl.DataFrame({'value':[42]}).write_parquet(out)

    @iv.data(dataset='derived/result/', why='declared manual input')
    def result(data=iv.all_of(source, why='reviewed values')):
        return data

    assert result()['value'].to_list()==[42]
