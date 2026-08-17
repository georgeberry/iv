"""The fingerprint is the bottom of the recursion, so its properties are the package's."""
from __future__ import annotations

import polars as pl
import pytest

from dagio import fingerprint as fp
from dagio.errors import FingerprintError

FRAME = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_same_data_rewritten_gives_the_same_id(tmp_path):
    """The property the whole design rests on: a no-op refetch invalidates nothing."""
    p, q = tmp_path / "a.parquet", tmp_path / "b.parquet"
    FRAME.write_parquet(p, compression="zstd")
    FRAME.write_parquet(q, compression="snappy")   # different bytes on purpose
    assert p.read_bytes() != q.read_bytes()
    assert fp.compute(p, "data") == fp.compute(q, "data")
    assert fp.compute(p, "bytes") != fp.compute(q, "bytes")


def test_one_changed_cell_gives_a_new_id(tmp_path):
    p, q = tmp_path / "a.parquet", tmp_path / "b.parquet"
    FRAME.write_parquet(p)
    FRAME.with_columns(pl.col("a").replace(3, 4)).write_parquet(q)
    assert fp.compute(p, "data") != fp.compute(q, "data")


def test_row_order_is_data_only_under_data_order(tmp_path):
    p, q = tmp_path / "a.parquet", tmp_path / "b.parquet"
    FRAME.write_parquet(p)
    FRAME.reverse().write_parquet(q)
    assert fp.compute(p, "data") == fp.compute(q, "data")
    assert fp.compute(p, "data_order") != fp.compute(q, "data_order")


def test_schema_change_moves_an_empty_frames_id(tmp_path):
    """An empty frame still has a schema, and a schema change IS a change."""
    p, q = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pl.DataFrame({"a": []}).write_parquet(p)
    pl.DataFrame({"b": []}).write_parquet(q)
    assert fp.compute(p, "data") != fp.compute(q, "data")


def test_rows_is_blind_to_a_same_shape_content_change(tmp_path):
    """Stated as a test because it is the hazard, not an accident: `rows` on a derived
    artifact lets a real change through."""
    p, q = tmp_path / "a.parquet", tmp_path / "b.parquet"
    FRAME.write_parquet(p)
    FRAME.with_columns(pl.col("a") * 100).write_parquet(q)
    assert fp.compute(p, "rows") == fp.compute(q, "rows")
    assert fp.compute(p, "data") != fp.compute(q, "data")


def test_unknown_strategy_raises_with_no_fallback(tmp_path):
    p = tmp_path / "a.parquet"
    FRAME.write_parquet(p)
    with pytest.raises(FingerprintError, match="unknown fingerprint strategy"):
        fp.compute(p, "mtime")


def test_data_on_a_non_tabular_file_says_what_to_use_instead(tmp_path):
    p = tmp_path / "a.json"
    p.write_text('{"a": 1}')
    with pytest.raises(FingerprintError, match="fp='bytes'"):
        fp.compute(p, "data")


def test_a_custom_callable_is_accepted(tmp_path):
    p = tmp_path / "a.parquet"
    FRAME.write_parquet(p)
    assert fp.compute(p, lambda _: "constant") == "constant"


def test_a_callable_returning_nothing_raises(tmp_path):
    p = tmp_path / "a.parquet"
    FRAME.write_parquet(p)
    with pytest.raises(FingerprintError, match="expected a non-empty str"):
        fp.compute(p, lambda _: "")
