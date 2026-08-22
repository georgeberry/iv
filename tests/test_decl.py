"""A declared selector is a VALUE, and these are the values.

`key_of` hashes them and `_resolve_sel` turns them back into a `where=`, so a shape that
differs by so much as a sort order is a stage that rebuilds forever, or one that never
does. Nothing else in the package pins them down, so they are written out in full here
rather than compared against another expression of the same thing.
"""
from __future__ import annotations

import pytest

from iv import decl
from iv.errors import DeclError
from iv.decl import PART


SHAPES = [
    ("all_of",       decl.all_of("d/", why="x"),                              (), ()),
    ("same_part",    decl.same_part("d/", why="x"),
     (("season", ("in", (PART,))),), ()),
    ("before_part",  decl.before_part("d/", why="x"),
     (("season", ("range", (("lt", PART),))),), ()),
    ("before_part inclusive", decl.before_part("d/", why="x", inclusive=True),
     (("season", ("range", (("le", PART),))),), ()),
    ("after_part",   decl.after_part("d/", why="x"),
     (("season", ("range", (("gt", PART),))),), ()),
    ("between",      decl.between("d/", why="x", ge="2020", lt=PART),
     (("season", ("range", (("ge", "2020"), ("lt", PART)))),), ()),
    ("parts",        decl.parts("d/", why="x", season=["2025", "2024"]),
     (("season", ("in", ("2024", "2025"))),), (("season", ("2024", "2025")),)),
]


@pytest.mark.parametrize("label,read,sel,where", SHAPES, ids=[s[0] for s in SHAPES])
def test_a_declared_selector_is_the_value_key_of_hashes(label, read, sel, where):
    bound = read.bound_to("season")
    assert bound.sel() == sel
    assert bound.where() == where, "the DAG edge test reads only outright values"


def test_a_bound_relative_to_the_partition_is_not_an_outright_value():
    """`where()` rules an edge OUT, so only a value stated on both sides counts. A bound
    that is not known until the shard is chosen cannot, and a missing edge is a wrong DAG.
    """
    assert decl.same_part("d/", why="x").bound_to("season").where() == ()
    assert decl.before_part("d/", why="x").bound_to("season").where() == ()


def test_a_triple_is_what_key_of_consumes():
    r = decl.before_part("processed/features/", why="prior seasons").bound_to("season")
    assert r.triple() == ("processed/features/",
                          (("season", ("range", (("lt", PART),))),), False)


def test_optional_rides_along():
    r = decl.all_of("raw/feed/", why="x", optional=True)
    assert r.triple()[2] is True


# ── the partition key is filled in, never repeated ────────────────────────────

def test_the_partition_key_comes_from_the_stage():
    r = decl.same_part("raw/box/", why="x")
    assert r.key is None, "the helper does not know the key"
    assert r.bound_to("season").sel() == (("season", ("in", (PART,))),)


def test_an_explicit_key_is_not_overwritten():
    r = decl.parts("raw/box/", why="x", season=["2024"])
    assert r.bound_to("year").key == "season"


def test_a_partition_relative_read_needs_a_partition():
    with pytest.raises(DeclError, match="only means something where there is a partition"):
        decl.same_part("raw/box/", why="x").bound_to(None)


def test_a_whole_dataset_read_does_not_need_one():
    assert decl.all_of("raw/box/", why="x").bound_to(None).sel() == ()
    assert decl.own_last_copy("raw/log/", why="x").bound_to(None).sel() == ()


# ── an own-copy read is lineage, never a trigger ──────────────────────────────

def test_own_last_copy_is_marked_and_optional():
    r = decl.own_last_copy("raw/odds_log/", why="yesterday's copy")
    assert r.is_own and r.optional and r.sel() == ()


# ── what is refused ───────────────────────────────────────────────────────────

def test_why_is_required():
    for make in (decl.all_of, decl.same_part, decl.before_part):
        with pytest.raises(DeclError, match="needs why="):
            make("raw/box/", why="")


def test_a_dataset_must_be_a_relative_path():
    with pytest.raises(DeclError, match="not a relative dataset path"):
        decl.all_of("s3://bucket/box/", why="x")


def test_between_refuses_an_unknown_bound():
    with pytest.raises(DeclError, match="unknown bound"):
        decl.between("raw/box/", why="x", approximately="2020")


def test_between_needs_a_bound():
    with pytest.raises(DeclError, match="needs at least one"):
        decl.between("raw/box/", why="x")


def test_parts_names_exactly_one_key():
    with pytest.raises(DeclError, match="exactly one partition key"):
        decl.parts("raw/box/", why="x", season=["2024"], week=["1"])
    with pytest.raises(DeclError, match="exactly one partition key"):
        decl.parts("raw/box/", why="x")


def test_parts_refuses_an_empty_coverage_claim():
    with pytest.raises(DeclError, match="no values"):
        decl.parts("raw/box/", why="x", season=[])


def test_a_scalar_is_a_set_of_one():
    assert decl.parts("raw/box/", why="x", season="2024").body == ("2024",)


def test_values_are_stringified_like_the_scan_does():
    assert decl.parts("raw/box/", why="x", season=[2025, 2024]).body == ("2024", "2025")
