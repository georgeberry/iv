

from __future__ import annotations

import hashlib

import polars as pl
import pytest

from tyke import shards as sh
from tyke.errors import DeclError, StateError


def dg(tag: str) -> str:

    return hashlib.sha256(tag.encode()).hexdigest()[:sh.DIGEST_LEN]


def frame(n=3, extra=0):
    return pl.DataFrame({"a": list(range(n)), "b": [x + extra for x in range(n)]})


def put(d, name, f=None):
    d.mkdir(parents=True, exist_ok=True)
    (f if f is not None else frame()).write_parquet(d / name)
    return d / name


def shard(d, part, tag, f=None):
    return put(d, sh.shard_name(part, dg(tag)), f)


def test_a_name_is_a_partition_and_a_fingerprint_and_nothing_else(tmp_path):
    n = sh.shard_name({"season": 2026}, dg("c"))
    assert n == f"season=2026.{dg('c')}.parquet"
    got = sh.parse_name(tmp_path / n)
    assert (got.part_str, got.fp, got.part) == ("season=2026", dg("c"), {"season": "2026"})

    bare = sh.parse_name(tmp_path / sh.shard_name(None, dg("c")))
    assert (bare.part_str, bare.fp, bare.part) == ("", dg("c"), {})


def test_multi_key_part_preserves_declared_order(tmp_path):
    n = sh.shard_name({"school": "duke", "season": 2019}, dg("c"))
    assert n.startswith("school=duke__season=2019.")
    assert sh.parse_name(tmp_path / n).part == {"school": "duke", "season": "2019"}


def test_a_separator_in_a_value_raises_rather_than_being_escaped():
    for bad in ({"season": "20.26"}, {"season": "a=b"}, {"k": "a__b"}, {"k": "a/b"}):
        with pytest.raises(DeclError):
            sh.encode_part(bad)


def test_an_underscore_in_a_value_is_fine(tmp_path):

    n = sh.shard_name({"dataset": "player_box", "season": 2026}, dg("c"))
    assert sh.parse_name(tmp_path / n).part == {"dataset": "player_box", "season": "2026"}


def test_a_fingerprint_must_look_like_one():
    with pytest.raises(DeclError, match="hex"):
        sh.shard_name({"season": 2026}, "short")


def test_non_shard_files_are_not_ours(tmp_path):
    for name in ("_index.json", "notes.txt", ".parquet", "aaaa.parquet",
                 "notes.backup.parquet", "season=2019.parquet"):
        assert sh.parse_name(tmp_path / name) is None


def test_a_stray_file_of_any_shape_stops_the_run(tmp_path):

    d = tmp_path / "ds"
    shard(d, {"season": 2026}, "a")
    for junk in ("notes.backup.parquet", "README.md", "_tmp-1.parquet", ".DS_Store"):
        (d / junk).write_text("x")
        with pytest.raises(StateError, match="not a shard"):
            sh.list_shards(d)
        (d / junk).unlink()


def test_listing_a_missing_dataset_does_not_probe_then_list(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    real = type(missing).exists
    probes = []
    monkeypatch.setattr(type(missing), "exists",
                        lambda self: (probes.append(self), real(self))[1])
    assert sh.list_shards(missing) == {}
    assert probes == []


def test_numeric_partitions_sort_numerically_not_lexically(tmp_path):
    d = tmp_path / "ds"
    for gid in (1, 2, 9, 10, 100):
        shard(d, {"game_id": gid}, str(gid))
    assert [s.part["game_id"] for s in sh.select(sh.current_shards(d))] == \
        ["1", "2", "9", "10", "100"]


def test_read_order_is_stable_across_repeated_listings(tmp_path):

    d = tmp_path / "ds"
    for s in (2011, 2006, 2026, 2019):
        shard(d, {"season": s}, str(s))
    assert len({tuple(x.name for x in sh.select(sh.current_shards(d))) for _ in range(6)}) == 1
    assert [x.part["season"] for x in sh.select(sh.current_shards(d))] == \
        ["2006", "2011", "2019", "2026"]


def test_sort_is_total_so_ties_cannot_reorder():
    a, b = sh.sort_key("season=2026"), sh.sort_key("season=2026__run=2")
    assert a != b and (a < b or b < a)


def test_a_permutation_is_not_new_data():
    f = frame(5)
    assert sh.fingerprint(f) == sh.fingerprint(f.reverse())


def test_the_fingerprint_moves_on_a_schema_change_alone():
    f = frame(3)
    assert sh.fingerprint(f) != sh.fingerprint(f.rename({"b": "c"}))
    assert sh.fingerprint(f) != sh.fingerprint(f.with_columns(pl.col("b").cast(pl.Float64)))


def test_the_fingerprint_sees_a_string_only_change():

    a = pl.DataFrame({"n": [1, 2], "team": ["LAS", "NYL"]})
    b = pl.DataFrame({"n": [1, 2], "team": ["SEA", "CHI"]})
    assert sh.fingerprint(a) != sh.fingerprint(b)


def test_the_fingerprint_is_stable_and_matches_the_on_disk_form(tmp_path):
    f = frame(4)
    p = put(tmp_path / "ds", f"{dg('x')}.parquet", f)
    assert sh.fingerprint(f) == sh.fingerprint(f)
    assert sh.fingerprint_of_file(p) == sh.fingerprint(f)


def test_empty_frames_differ_by_schema():
    assert sh.fingerprint(pl.DataFrame(schema={"a": pl.Int64})) != \
        sh.fingerprint(pl.DataFrame(schema={"a": pl.Utf8}))


def test_dataset_id_folds_fingerprints_and_ignores_file_order(tmp_path):
    d = tmp_path / "ds"
    for s in ("2025", "2026"):
        shard(d, {"season": s}, s)
    sel = sh.select(sh.current_shards(d))
    assert sh.dataset_id(sel) == sh.dataset_id(list(reversed(sel)))
    assert sh.dataset_id([]) == "data:(empty)"


def test_a_selection_has_its_own_id(tmp_path):
    d = tmp_path / "ds"
    for s in ("2024", "2025", "2026"):
        shard(d, {"season": s}, s)
    all_ = sh.current_shards(d)
    upto = sh.select(all_, {"season": {"le": "2025"}})
    assert len(upto) == 2 and sh.dataset_id(upto) != sh.dataset_id(sh.select(all_))


def test_a_later_shard_cannot_move_an_earlier_selection(tmp_path):

    d = tmp_path / "ds"
    for s in ("2024", "2025"):
        shard(d, {"season": s}, s)
    upto = {"season": {"le": "2025"}}
    before = sh.dataset_id(sh.select(sh.current_shards(d), upto))
    shard(d, {"season": "2026"}, "2026")
    assert sh.dataset_id(sh.select(sh.current_shards(d), upto)) == before


def test_an_explicit_list_is_a_coverage_claim(tmp_path):
    d = tmp_path / "ds"
    for s in ("2024", "2025"):
        shard(d, {"season": s}, s)
    assert [x.part["season"] for x in
            sh.select(sh.current_shards(d), {"season": ["2024", "2025"]})] == ["2024", "2025"]
    with pytest.raises(StateError, match="2026"):
        sh.select(sh.current_shards(d), {"season": ["2025", "2026"]})


def test_a_predicate_may_match_nothing(tmp_path):
    d = tmp_path / "ds"
    shard(d, {"season": "2024"}, "c")
    assert sh.select(sh.current_shards(d), {"season": {"gt": "2100"}}) == []


def test_selection_on_a_missing_directory_is_empty(tmp_path):
    assert sh.current_shards(tmp_path / "nope") == {} and sh.select({}, None) == []


def test_commit_names_by_fingerprint_and_drops_what_it_replaced(tmp_path):
    d = tmp_path / "ds"
    f = frame(3)
    tmp = sh.stage("1", tmp_path)
    f.write_parquet(tmp)
    first = sh.commit(tmp, d, part={"season": 2026})
    assert first.name == f"season=2026.{sh.fingerprint(f)}.parquet"

    f2 = frame(3, extra=100)
    tmp2 = sh.stage("2", tmp_path)
    f2.write_parquet(tmp2)
    second = sh.commit(tmp2, d, part={"season": 2026})
    assert not first.exists() and [p.name for p in d.iterdir()] == [second.name]


def test_identical_data_is_the_identical_file_and_touches_nothing(tmp_path):

    d = tmp_path / "ds"
    f = frame(3)
    names = []
    for i in range(2):
        tmp = sh.stage(i, tmp_path)
        f.write_parquet(tmp)
        names.append(sh.commit(tmp, d, part={"season": 2026}).name)
        assert not tmp.exists()
    assert names[0] == names[1]
    assert [p.name for p in d.iterdir()] == [f"season=2026.{sh.fingerprint(f)}.parquet"]


def test_commit_reports_only_a_content_change(tmp_path):
    d = tmp_path / "ds"
    changed = []
    for key, extra in (("a" * 16, 0), ("b" * 16, 0), ("c" * 16, 1)):
        tmp = sh.stage(key, tmp_path)
        frame(3, extra=extra).write_parquet(tmp)
        sh.commit(tmp, d, part={"season": 2026}, key=key,
                  on_commit=lambda _, moved: changed.append(moved))
    assert changed == [True, False, True]


def test_commit_leaves_other_partitions_alone(tmp_path):
    d = tmp_path / "ds"
    shard(d, {"season": 2025}, "old")
    tmp = sh.stage("x", tmp_path)
    frame().write_parquet(tmp)
    sh.commit(tmp, d, part={"season": 2026})
    assert set(sh.current_shards(d)) == {"season=2025", "season=2026"}


def test_staging_is_local_so_the_fingerprint_never_downloads_what_it_just_uploaded(tmp_path):
    p = sh.stage("1", tmp_path)
    assert p.parent.parent == tmp_path and "ds" not in str(p)


def test_the_committed_fingerprint_is_what_a_READER_gets_back(tmp_path):
    d = tmp_path / "ds"
    tmp = sh.stage("x", tmp_path)
    frame(4).write_parquet(tmp)
    final = sh.commit(tmp, d, part={"season": 2026})
    assert sh.parse_name(final).fp == sh.fingerprint(pl.read_parquet(final))


def test_two_shards_for_one_partition_raise_rather_than_being_guessed(tmp_path):
    d = tmp_path / "ds"
    shard(d, {"season": 2026}, "a")
    shard(d, {"season": 2026}, "b")
    assert len(sh.list_shards(d)["season=2026"]) == 2
    with pytest.raises(StateError, match="interrupted"):
        sh.current_shards(d)


def test_gc_drops_what_is_not_kept(tmp_path):
    d = tmp_path / "ds"
    keep = shard(d, {"season": 2026}, "a")
    stale = shard(d, {"season": 2026}, "b")
    assert sh.gc(d, keep={keep.name}) == [stale.name]
    assert sh.current_shards(d)["season=2026"].name == keep.name


def test_every_comparison_is_answered_from_FILENAMES_alone(tmp_path):


    d = tmp_path / "ds"
    for s in ("2024", "2025", "2026"):
        tmp = sh.stage(s, tmp_path)
        frame(3, extra=int(s)).write_parquet(tmp)
        sh.commit(tmp, d, part={"season": s})

    upto = {"season": {"le": "2025"}}
    live = sh.select(sh.current_shards(d))
    before = (
        [x.name for x in live],
        sh.dataset_id(live),
        sh.dataset_id(sh.select(sh.current_shards(d), upto)),
    )
    for x in live:
        x.path.write_bytes(b"not a parquet file")
    with pytest.raises(Exception):
        pl.read_parquet(live[0].path)

    again = sh.select(sh.current_shards(d))
    assert ([x.name for x in again], sh.dataset_id(again),
            sh.dataset_id(sh.select(sh.current_shards(d), upto))) == before
