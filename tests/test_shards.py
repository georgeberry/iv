"""`shards.py` alone: names, ordering, digests, selection, commit, gc, index.

No Invalidator, no static scan, no bucket. A shard is a file in a temp directory and every
question here is answerable from the directory listing, which is the point of the module.
"""
from __future__ import annotations

import hashlib
import json

import polars as pl
import pytest

from iv import shards as sh
from iv.errors import DeclError, StateError


def dg(tag: str) -> str:
    """A stand-in digest of the real shape — 16 hex chars, which the names require."""
    return hashlib.sha256(tag.encode()).hexdigest()[:sh.DIGEST_LEN]


def frame(n=3, extra=0):
    return pl.DataFrame({"a": list(range(n)), "b": [x + extra for x in range(n)]})


def put(d, name, f=None):
    d.mkdir(parents=True, exist_ok=True)
    (f if f is not None else frame()).write_parquet(d / name)
    return d / name


# ── names round-trip ──────────────────────────────────────────────────────────

def test_name_round_trips_with_and_without_a_partition(tmp_path):
    c, r = dg("c"), dg("r")
    n = sh.shard_name({"season": 2026}, c, r)
    assert n == f"season=2026.{c}.{r}.parquet"
    got = sh.parse_name(tmp_path / n)
    assert (got.part_str, got.content, got.recipe) == ("season=2026", c, r)
    assert got.part == {"season": "2026"}

    got2 = sh.parse_name(tmp_path / sh.shard_name(None, c, r))
    assert (got2.part_str, got2.content, got2.recipe, got2.part) == ("", c, r, {})


def test_multi_key_part_preserves_declared_order(tmp_path):
    n = sh.shard_name({"school": "duke", "season": 2019}, dg("c"), dg("r"))
    assert n.startswith("school=duke__season=2019.")
    assert sh.parse_name(tmp_path / n).part == {"school": "duke", "season": "2019"}


def test_settled_names_carry_no_digests(tmp_path):
    n = sh.shard_name({"season": 2019}, digested=False)
    assert n == "season=2019.parquet"
    got = sh.parse_name(tmp_path / n, digested=False)
    assert got.part_str == "season=2019" and got.content == "" and got.recipe == ""

    assert sh.shard_name(None, digested=False) == "_.parquet"
    assert sh.parse_name(tmp_path / "_.parquet", digested=False).part == {}


def test_policy_decides_how_a_name_parses_not_the_name(tmp_path):
    """One filename, two datasets, two answers — and neither is a guess.

    `<hex>.<hex>.parquet` is a content+recipe pair on a tracked dataset. On a settled one
    it is not a shard at all, because a settled name is a partition and a partition is
    always `key=value`. Deciding by the dataset's POLICY is what stops a name parsing two
    ways, and a wrong guess here would be a wrong id rather than an error.
    """
    name = f"{dg('a')}.{dg('b')}.parquet"
    tracked = sh.parse_name(tmp_path / name, digested=True)
    assert (tracked.content, tracked.recipe) == (dg("a"), dg("b"))
    assert sh.parse_name(tmp_path / name, digested=False) is None


def test_a_separator_in_a_value_raises_rather_than_being_escaped():
    for bad in ({"season": "20.26"}, {"season": "a=b"}, {"k": "a__b"}, {"k": "a/b"}):
        with pytest.raises(DeclError):
            sh.encode_part(bad)


def test_a_digest_must_look_like_one():
    with pytest.raises(DeclError, match="hex"):
        sh.shard_name({"season": 2026}, "short", dg("r"))


def test_junk_in_a_dataset_directory_is_skipped_not_fatal(tmp_path):
    """`list_shards` walks whatever is there, so "not ours" has to be an answer."""
    d = tmp_path / "ds"
    keep = put(d, sh.shard_name({"season": 2026}, dg("a"), dg("r")))
    put(d, "notes.backup.parquet")
    put(d, "season=2019.parquet")             # settled-shaped, in a tracked dataset
    (d / "README.md").write_text("hi")
    (d / sh.INDEX_NAME).write_text("{}")
    assert [s.name for v in sh.list_shards(d).values() for s in v] == [keep.name]


def test_non_shard_files_are_ignored(tmp_path):
    assert sh.parse_name(tmp_path / "_index.json") is None
    assert sh.parse_name(tmp_path / "notes.txt") is None
    assert sh.parse_name(tmp_path / ".parquet") is None


# ── ordering ──────────────────────────────────────────────────────────────────

def test_numeric_partitions_sort_numerically_not_lexically(tmp_path):
    d = tmp_path / "ds"
    for gid in (1, 2, 9, 10, 100):
        put(d, sh.shard_name({"game_id": gid}, dg(str(gid)), dg("r")))
    assert [s.part["game_id"] for s in sh.select(sh.current_shards(d))] == \
        ["1", "2", "9", "10", "100"]


def test_read_order_is_stable_across_repeated_listings(tmp_path):
    """Row order is a model input; a listing that reorders is the bug this rules out."""
    d = tmp_path / "ds"
    for s in (2011, 2006, 2026, 2019):
        put(d, sh.shard_name({"season": s}, dg(str(s)), dg("r")))
    orders = {tuple(x.name for x in sh.select(sh.current_shards(d))) for _ in range(6)}
    assert len(orders) == 1
    assert [x.part["season"] for x in sh.select(sh.current_shards(d))] == \
        ["2006", "2011", "2019", "2026"]


def test_sort_is_total_so_ties_cannot_reorder():
    a, b = sh.sort_key("season=2026"), sh.sort_key("season=2026__run=2")
    assert a != b and (a < b or b < a)


# ── digests ───────────────────────────────────────────────────────────────────

def test_content_digest_is_order_sensitive():
    f = frame(5)
    assert sh.content_digest(f) != sh.content_digest(f.reverse())


def test_content_digest_moves_on_a_schema_change_alone():
    f = frame(3)
    assert sh.content_digest(f) != sh.content_digest(f.rename({"b": "c"}))
    assert sh.content_digest(f) != sh.content_digest(f.with_columns(pl.col("b").cast(pl.Float64)))


def test_content_digest_is_stable_and_matches_the_on_disk_form(tmp_path):
    f = frame(4)
    p = put(tmp_path / "ds", f"{dg('x')}.{dg('y')}.parquet", f)
    assert sh.content_digest(f) == sh.content_digest(f)
    assert sh.content_digest_of_file(p) == sh.content_digest(f)


def test_empty_frames_differ_by_schema():
    a = pl.DataFrame(schema={"a": pl.Int64})
    b = pl.DataFrame(schema={"a": pl.Utf8})
    assert sh.content_digest(a) != sh.content_digest(b)


def test_recipe_is_insensitive_to_input_discovery_order():
    x = sh.recipe_digest("meta", {"a/": "1", "b/": "2"})
    assert x == sh.recipe_digest("meta", {"b/": "2", "a/": "1"})
    assert x != sh.recipe_digest("meta", {"a/": "1", "b/": "3"})
    assert x != sh.recipe_digest("other", {"a/": "1", "b/": "2"})


def test_dataset_id_folds_content_and_ignores_file_order(tmp_path):
    d = tmp_path / "ds"
    for s in ("2025", "2026"):
        put(d, sh.shard_name({"season": s}, dg(s), dg("r")))
    sel = sh.select(sh.current_shards(d))
    assert sh.dataset_id(sel) == sh.dataset_id(list(reversed(sel)))
    assert sh.dataset_id([]) == "ds:(empty)"


def test_a_selection_has_its_own_id(tmp_path):
    d = tmp_path / "ds"
    for s in ("2024", "2025", "2026"):
        put(d, sh.shard_name({"season": s}, dg(s), dg("r")))
    all_ = sh.current_shards(d)
    upto25 = sh.select(all_, {"season": lambda v: v <= "2025"})
    assert len(upto25) == 2
    assert sh.dataset_id(upto25) != sh.dataset_id(sh.select(all_))


def test_a_later_shard_cannot_move_an_earlier_selection(tmp_path):
    """The walk-forward guarantee, and the reason `fp_of=` is not needed."""
    d = tmp_path / "ds"
    for s in ("2024", "2025"):
        put(d, sh.shard_name({"season": s}, dg(s), dg("r")))
    upto = {"season": lambda v: v <= "2025"}
    before = sh.dataset_id(sh.select(sh.current_shards(d), upto))
    put(d, sh.shard_name({"season": "2026"}, dg("2026"), dg("r")))
    assert sh.dataset_id(sh.select(sh.current_shards(d), upto)) == before


# ── selection ─────────────────────────────────────────────────────────────────

def test_an_explicit_list_is_a_coverage_claim(tmp_path):
    d = tmp_path / "ds"
    for s in ("2024", "2025"):
        put(d, sh.shard_name({"season": s}, dg(s), dg("r")))
    got = sh.select(sh.current_shards(d), {"season": ["2024", "2025"]})
    assert [x.part["season"] for x in got] == ["2024", "2025"]
    with pytest.raises(StateError, match="2026"):
        sh.select(sh.current_shards(d), {"season": ["2025", "2026"]})


def test_a_predicate_may_match_nothing(tmp_path):
    d = tmp_path / "ds"
    put(d, sh.shard_name({"season": "2024"}, dg("c"), dg("r")))
    assert sh.select(sh.current_shards(d), {"season": lambda v: v > "2100"}) == []


def test_selection_on_a_missing_directory_is_empty(tmp_path):
    assert sh.current_shards(tmp_path / "nope") == {}
    assert sh.select({}, None) == []


# ── commit ────────────────────────────────────────────────────────────────────

def test_commit_names_by_digest_and_drops_what_it_replaced(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    f = frame(3)
    tmp = d / "tmp-1.parquet"
    f.write_parquet(tmp)
    final = sh.commit(tmp, d, part={"season": 2026},
                      content=sh.content_digest(f), recipe=dg("r1"))
    assert final.name == f"season=2026.{sh.content_digest(f)}.{dg('r1')}.parquet"

    f2 = frame(3, extra=100)
    tmp2 = d / "tmp-2.parquet"
    f2.write_parquet(tmp2)
    final2 = sh.commit(tmp2, d, part={"season": 2026},
                       content=sh.content_digest(f2), recipe=dg("r2"))
    assert not final.exists()
    assert [p.name for p in d.iterdir()] == [final2.name]


def test_identical_content_under_a_new_recipe_keeps_the_content_segment(tmp_path):
    """Early cutoff: the CONTENT segment does not move, so dependants see no change."""
    d = tmp_path / "ds"
    d.mkdir()
    f = frame(3)
    c = sh.content_digest(f)
    for recipe in ("r1", "r2"):
        tmp = d / f"tmp-{recipe}.parquet"
        f.write_parquet(tmp)
        sh.commit(tmp, d, part={"season": 2026}, content=c, recipe=dg(recipe))
    assert [p.name for p in d.iterdir()] == [f"season=2026.{c}.{dg('r2')}.parquet"]
    assert sh.current_shards(d)["season=2026"].content == c


def test_recommitting_the_identical_shard_is_a_no_op(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    f = frame(3)
    c = sh.content_digest(f)
    for i in range(2):
        tmp = d / f"tmp-{i}.parquet"
        f.write_parquet(tmp)
        sh.commit(tmp, d, part={"season": 2026}, content=c, recipe=dg("r"))
    assert [p.name for p in d.iterdir()] == [f"season=2026.{c}.{dg('r')}.parquet"]


def test_commit_leaves_other_partitions_alone(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    put(d, sh.shard_name({"season": 2025}, dg("old"), dg("r")))
    tmp = d / "tmp.parquet"
    frame().write_parquet(tmp)
    sh.commit(tmp, d, part={"season": 2026}, content=dg("new"), recipe=dg("r"))
    assert set(sh.current_shards(d)) == {"season=2025", "season=2026"}


# ── an interrupted commit ─────────────────────────────────────────────────────

def test_two_shards_for_one_partition_raise_rather_than_being_guessed(tmp_path):
    d = tmp_path / "ds"
    put(d, sh.shard_name({"season": 2026}, dg("a"), dg("r1")))
    put(d, sh.shard_name({"season": 2026}, dg("b"), dg("r2")))
    assert len(sh.list_shards(d)["season=2026"]) == 2
    with pytest.raises(StateError, match="interrupted"):
        sh.current_shards(d)


def test_gc_drops_what_is_not_kept(tmp_path):
    d = tmp_path / "ds"
    keep = put(d, sh.shard_name({"season": 2026}, dg("a"), dg("r1")))
    stale = put(d, sh.shard_name({"season": 2026}, dg("b"), dg("r2")))
    assert sh.gc(d, keep={keep.name}) == [stale.name]
    assert sh.current_shards(d)["season=2026"].name == keep.name


# ── the index is advisory ─────────────────────────────────────────────────────

def test_index_round_trips(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    sh.write_entry(d, "season=2026", {"content": dg("c"), "seconds": 1.5})
    assert sh.read_index(d)["shards"]["season=2026"]["seconds"] == 1.5


def test_a_corrupt_or_missing_index_costs_an_explanation_not_a_decision(tmp_path):
    d = tmp_path / "ds"
    put(d, sh.shard_name({"season": 2026}, dg("a"), dg("r1")))
    assert sh.read_index(d) == {}
    for junk in ("{not json", json.dumps({"v": 999, "shards": {}})):
        (d / sh.INDEX_NAME).write_text(junk)
        assert sh.read_index(d) == {}
    # The decision is the filename, and it is unaffected by any of the above.
    assert sh.current_shards(d)["season=2026"].recipe == dg("r1")


def test_the_index_never_takes_down_a_build(tmp_path):
    sh.write_entry(tmp_path / "does" / "not" / "exist", "p", {"a": 1})
