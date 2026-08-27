

from __future__ import annotations

import pytest

from tyke import decl
from tyke.errors import DeclError
from tyke.decl import PART


class _Declared:


    def __init__(self, dataset="processed/features/"):
        self.dataset = dataset


SHAPES = [
    ("all_of",       decl.all_of(_Declared("d/"), why="x"),                              (), ()),
    ("same_part",    decl.same_part(_Declared("d/"), why="x"),
     (("season", ("in", (PART,))),), ()),
    ("before_part",  decl.before_part(_Declared("d/"), why="x"),
     (("season", ("range", (("lt", PART),))),), ()),
    ("before_part inclusive", decl.before_part(_Declared("d/"), why="x", inclusive=True),
     (("season", ("range", (("le", PART),))),), ()),
    ("after_part",   decl.after_part(_Declared("d/"), why="x"),
     (("season", ("range", (("gt", PART),))),), ()),
    ("between",      decl.between(_Declared("d/"), why="x", ge="2020", lt=PART),
     (("season", ("range", (("ge", "2020"), ("lt", PART)))),), ()),
    ("parts",        decl.parts(_Declared("d/"), why="x", season=["2025", "2024"]),
     (("season", ("in", ("2024", "2025"))),), (("season", ("2024", "2025")),)),
]


@pytest.mark.parametrize("label,read,sel,where", SHAPES, ids=[s[0] for s in SHAPES])
def test_a_declared_selector_is_the_value_key_of_hashes(label, read, sel, where):
    bound = read.bound_to("season")
    assert bound.sel() == sel
    assert bound.where() == where, "the DAG edge test reads only outright values"


def test_a_bound_relative_to_the_partition_is_not_an_outright_value():


    assert decl.same_part(_Declared("d/"), why="x").bound_to("season").where() == ()
    assert decl.before_part(_Declared("d/"), why="x").bound_to("season").where() == ()


def test_a_triple_is_what_key_of_consumes():
    r = decl.before_part(_Declared("processed/features/"), why="prior seasons").bound_to("season")
    assert r.triple() == ("processed/features/",
                          (("season", ("range", (("lt", PART),))),), False)


def test_optional_rides_along():
    r = decl.all_of(_Declared("raw/feed/"), why="x", optional=True)
    assert r.triple()[2] is True


def test_the_partition_key_comes_from_the_stage():
    r = decl.same_part(_Declared("raw/box/"), why="x")
    assert r.key is None, "the helper does not know the key"
    assert r.bound_to("season").sel() == (("season", ("in", (PART,))),)


def test_an_explicit_key_is_not_overwritten():
    r = decl.parts(_Declared("raw/box/"), why="x", season=["2024"])
    assert r.bound_to("year").key == "season"


def test_a_partition_relative_read_needs_a_partition():
    with pytest.raises(DeclError, match="only means something where there is a partition"):
        decl.same_part(_Declared("raw/box/"), why="x").bound_to(None)


def test_a_whole_dataset_read_does_not_need_one():
    assert decl.all_of(_Declared("raw/box/"), why="x").bound_to(None).sel() == ()
    assert decl.own_last_copy(_Declared("raw/log/"), why="x").bound_to(None).sel() == ()


def test_own_last_copy_is_marked_and_optional():
    r = decl.own_last_copy(_Declared("raw/odds_log/"), why="yesterday's copy")
    assert r.is_own and r.optional and r.sel() == ()


def test_why_is_required():
    for make in (decl.all_of, decl.same_part, decl.before_part):
        with pytest.raises(DeclError, match="needs why="):
            make(_Declared("raw/box/"), why="")


def test_why_is_one_tweet_or_shorter():
    with pytest.raises(DeclError, match="280"):
        decl.all_of(_Declared(), why="x" * 281)
    assert decl.all_of(_Declared(), why="x" * 280).why == "x" * 280


def test_a_dataset_must_be_a_relative_path():
    with pytest.raises(DeclError, match="not a relative dataset path"):
        decl.all_of(_Declared("s3://bucket/box/"), why="x")


def test_between_refuses_an_unknown_bound():
    with pytest.raises(DeclError, match="unknown bound"):
        decl.between(_Declared("raw/box/"), why="x", approximately="2020")


def test_between_needs_a_bound():
    with pytest.raises(DeclError, match="needs at least one"):
        decl.between(_Declared("raw/box/"), why="x")


def test_parts_can_name_multiple_keys():
    r = decl.parts(_Declared("raw/box/"), why="x", season=["2024"], week=["1"])
    assert r.sel() == (("season", ("in", ("2024",))), ("week", ("in", ("1",))))
    with pytest.raises(DeclError, match="at least one partition key"):
        decl.parts(_Declared("raw/box/"), why="x")


def test_a_composite_range_keeps_the_other_dimension_equal():
    r = decl.before_part(_Declared("raw/box/"), why="earlier seasons", key="season")
    assert r.bound_to(("league", "season")).sel() == (
        ("season", ("range", (("lt", decl.PART),))),
        ("league", ("in", (decl.PART,))),
    )


def test_parts_refuses_an_empty_coverage_claim():
    with pytest.raises(DeclError, match="no values"):
        decl.parts(_Declared("raw/box/"), why="x", season=[])


def test_a_scalar_is_a_set_of_one():
    assert decl.parts(_Declared("raw/box/"), why="x", season="2024").body == ("2024",)


def test_values_are_stringified_like_the_scan_does():
    assert decl.parts(_Declared("raw/box/"), why="x", season=[2025, 2024]).body == ("2024", "2025")


def test_a_read_names_a_declaration():


    assert decl.all_of(_Declared(), why="x").dataset == "processed/features/"


def test_a_dataset_nothing_produces_is_declared_too():


    assert decl.all_of(_Declared("raw_data/pbp_official/"),
                       why="x").dataset == "raw_data/pbp_official/"


def test_every_helper_takes_either_form():
    for make in (decl.all_of, decl.same_part, decl.before_part, decl.after_part):
        assert make(_Declared(), why="x").dataset == "processed/features/"
    assert decl.parts(_Declared(), why="x", season=["2024"]).dataset == "processed/features/"
    assert decl.between(_Declared(), why="x", lt="2021").dataset == "processed/features/"


def test_a_bare_path_is_refused_and_says_what_to_do():
    with pytest.raises(DeclError, match="names a DECLARED dataset"):
        decl.all_of("raw/box/", why="x")
    with pytest.raises(DeclError, match="names a DECLARED dataset"):
        decl.all_of(len, why="x")


def test_as_paths_defaults_to_the_contents():
    assert decl.all_of(_Declared("d/"), why="x").as_paths is False
    assert decl.all_of(_Declared("d/"), why="x", as_paths=True).as_paths is True


def test_as_paths_survives_being_bound_to_a_partition():
    r = decl.same_part(_Declared("d/"), why="x", as_paths=True).bound_to("season")
    assert r.as_paths is True and r.sel() == (("season", ("in", (PART,))),)


def test_own_last_copy_names_nothing_until_the_stage_says():
    bare = decl.own_last_copy(why="yesterday's copy")
    assert bare.dataset is None and bare.is_own
    assert bare.against("raw/log/").dataset == "raw/log/"


def test_own_last_copy_may_still_name_one():
    named = decl.own_last_copy(_Declared("raw/log/"), why="x")
    assert named.against("something/else/").dataset == "raw/log/", "an explicit one wins"


def test_a_bare_own_last_copy_on_a_stage_writing_several_says_so():
    with pytest.raises(DeclError, match="writes several"):
        decl.own_last_copy(why="x").against(None)
