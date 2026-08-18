"""The graph read out of the source, and every check it can fail."""
from __future__ import annotations

import pytest

from conftest import write_stage
from invalidator import Invalidator
from invalidator import graph as G
from invalidator.errors import DeclError

GOOD = {
    "stages/build_stats.py": '''
        import polars as pl
        from mypipe import iv

        @iv.step("processed/team_stats.parquet", why="season points by team")
        def build(out):
            games = pl.read_parquet(iv.reads(
                "raw/games.parquet", why="one row per team per game", fp="rows"))
            games.write_parquet(out)
    ''',
    "stages/build_ratings.py": '''
        import polars as pl
        from mypipe import iv

        @iv.step("processed/ratings.parquet",
                 why="what the app renders", terminal=True)
        def build(out):
            stats = pl.read_parquet(iv.reads(
                "processed/team_stats.parquet", why="season points by team"))
            stats.write_parquet(out)
    ''',
}


def make(project, files: dict[str, str], stages=None, **kw):
    for rel, body in files.items():
        write_stage(project, rel, body)
    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project,
                     stages=stages, **kw)
    return G.build(iv)


# ── the scan ──────────────────────────────────────────────────────────────────

def test_it_finds_calls_on_an_instance_whatever_it_is_named(project):
    """The receiver is an arbitrary name imported from elsewhere, so matching is on the
    method name plus the required `why=` literal."""
    g = make(project, GOOD)
    assert set(g.stages) == set(GOOD)
    assert g.producers_of("processed/team_stats.parquet") == ["stages/build_stats.py"]
    assert g.consumers_of("processed/team_stats.parquet") == ["stages/build_ratings.py"]


def test_a_differently_named_instance_still_matches(project):
    g = make(project, {"stages/a.py": '''
        from somewhere.else_ import PIPELINE as whatever
        @whatever.step("processed/x.parquet", why="an oddly named instance",
                       terminal=True)
        def build(out):
            whatever.reads("raw/y.parquet", why="the source")
    '''})
    assert g.producers_of("processed/x.parquet") == ["stages/a.py"]


def test_metadata_survives_the_scan(project):
    g = make(project, GOOD)
    site = g.sites["raw/games.parquet"][0]
    assert site.why == "one row per team per game"
    assert site.fp == "rows"
    assert g.is_terminal("processed/ratings.parquet")


def test_a_clean_pipeline_has_no_findings(project):
    g = make(project, GOOD, stages=list(GOOD))
    errors, warns = G.check(g)
    assert errors == []
    assert warns == []


# ── the literal rule, enforced rather than requested ──────────────────────────

def test_a_computed_path_is_an_error_that_names_the_line(project):
    with pytest.raises(DeclError, match=r"stages/bad\.py:4.*string LITERAL"):
        make(project, {"stages/bad.py": '''
            from mypipe import iv
            NAME = "xpm"
            def build():
                iv.reads(f"processed/{NAME}.parquet", why="computed on purpose")
        '''})


def test_a_call_without_why_is_simply_not_ours(project):
    """`why=` is what identifies a call as ours. Something else's `.reads(...)` is not a
    declaration and must not be mistaken for one."""
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        import zipfile

        @iv.step("processed/x.parquet", why="the real one", terminal=True)
        def build(out):
            iv.reads("raw/y.parquet", why="the source")
            zipfile.ZipFile("z.zip").reads("not-a-declaration")
    '''})
    assert sorted(s.path for s in g.stages["stages/a.py"].sites) == [
        "processed/x.parquet", "raw/y.parquet"]


def test_a_template_without_part_is_an_error(project):
    with pytest.raises(DeclError, match=r"placeholder\(s\) \['season'\] but no part="):
        make(project, {"stages/bad.py": '''
            from mypipe import iv
            def build():
                iv.reads("raw/box/{season}.parquet", why="per season")
        '''})


def test_part_keys_must_match_the_template(project):
    with pytest.raises(DeclError, match="part= supplies"):
        make(project, {"stages/bad.py": '''
            from mypipe import iv
            def build(year):
                iv.reads("raw/box/{season}.parquet", why="per season",
                         part={"year": year})
        '''})


# ── the checks ────────────────────────────────────────────────────────────────

def test_write_with_no_consumer(project):
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        @iv.step("processed/dead.parquet", why="nobody reads this")
        def build(out):
            iv.reads("raw/games.parquet", why="the source")
    '''})
    errors, _ = G.check(g)
    assert any("WRITE WITH NO CONSUMER  processed/dead.parquet" in e for e in errors)


def test_terminal_makes_no_consumer_correct(project):
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        @iv.step("dump/site.json", why="the app renders it", terminal=True, fp="bytes")
        def build(out):
            iv.reads("raw/games.parquet", why="the source")
    '''})
    errors, _ = G.check(g)
    assert not any("NO CONSUMER" in e for e in errors)


def test_read_with_no_producer(project):
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        @iv.step("dump/out.json", why="terminal", terminal=True)
        def build(out):
            iv.reads("processed/nobody_makes_this.parquet", why="a missing stage")
    '''})
    errors, _ = G.check(g)
    assert any("READ WITH NO PRODUCER" in e for e in errors)


def test_two_writers(project):
    body = '''
        from mypipe import iv
        @iv.step("processed/x.parquet", why="the same artifact", terminal=True)
        def build(out):
            iv.reads("raw/games.parquet", why="the source")
    '''
    g = make(project, {"stages/a.py": body, "stages/b.py": body})
    errors, _ = G.check(g)
    assert any("TWO WRITERS" in e for e in errors)


def test_policy_conflict(project):
    g = make(project, {
        "stages/a.py": '''
            from mypipe import iv
            def build():
                iv.reads("raw/games.parquet", why="the source")
                with iv.updates("processed/x.parquet", why="first pass",
                                policy="manual", terminal=True) as p:
                    pass
        ''',
        "stages/b.py": '''
            from mypipe import iv
            def build():
                iv.reads("raw/games.parquet", why="the source")
                with iv.updates("processed/x.parquet", why="second pass",
                                policy="tracked", terminal=True) as p:
                    pass
        ''',
    })
    errors, _ = G.check(g)
    assert any("POLICY CONFLICT" in e and "policy=" in e for e in errors)


def test_a_cycle(project):
    g = make(project, {
        "stages/a.py": '''
            from mypipe import iv
            @iv.step("processed/a.parquet", why="to a")
            def build(out):
                iv.reads("processed/b.parquet", why="from b")
        ''',
        "stages/b.py": '''
            from mypipe import iv
            @iv.step("processed/b.parquet", why="to b")
            def build(out):
                iv.reads("processed/a.parquet", why="from a")
        ''',
    })
    errors, _ = G.check(g)
    assert any(e.startswith("CYCLE") for e in errors)


def test_updates_is_not_a_cycle(project):
    """A stage that reads its own output to decide what to skip is an incremental cache,
    not a cycle. Without updates() the only ways out are a false alarm or an allowlist."""
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        def build():
            iv.reads("raw/games.parquet", why="the source")
            with iv.updates("processed/cache.parquet",
                            why="reads its own output to decide what to skip",
                            terminal=True) as p:
                pass
    '''})
    errors, _ = G.check(g)
    assert not any(e.startswith("CYCLE") for e in errors), errors


def test_order_violation_is_found(project):
    g = make(project, GOOD,
             stages=["stages/build_ratings.py", "stages/build_stats.py"])
    errors, _ = G.check(g)
    assert any(e.startswith("ORDER") for e in errors)


def test_prior_makes_a_late_producer_legal(project):
    files = dict(GOOD)
    files["stages/build_ratings.py"] = files["stages/build_ratings.py"].replace(
        'why="season points by team"))',
        'why="season points by team", prior=True))')
    g = make(project, files,
             stages=["stages/build_ratings.py", "stages/build_stats.py"])
    errors, _ = G.check(g)
    assert not any(e.startswith("ORDER") for e in errors), errors


def test_guarding_a_fetch_is_an_error(project):
    """The guard against this package's own silent failure.

    Nothing in a fetched artifact's id can move on its own, so guarding the stage that
    produces it means it runs once and never again — which looks exactly like the cache
    working perfectly.
    """
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        @iv.step("raw/games.parquet", why="as fetched", terminal=True)
        def build(out):
            iv.external("vendor/feed", why="the upstream drop")
    '''})
    errors, _ = G.check(g)
    assert any("GUARDED FETCH  raw/games.parquet" in e for e in errors), errors


def test_an_unguarded_fetch_is_fine(project):
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        iv.external("vendor/feed", why="the upstream drop")
        with iv.writes("raw/games.parquet", why="as fetched", terminal=True) as p:
            pass
    '''})
    errors, warns = G.check(g)
    assert not any("GUARDED FETCH" in e for e in errors), errors
    assert not any("NO PROVENANCE" in w for w in warns), warns


def test_settled_says_the_staleness_question_does_not_apply(project):
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        @iv.step("raw/history.parquet", why="fetch-once history",
                 policy="settled", terminal=True)
        def build(out):
            iv.external("vendor/archive", why="a season that will never change")
    '''})
    errors, _ = G.check(g)
    assert not any("GUARDED FETCH" in e for e in errors), errors


def test_an_artifact_from_nothing_warns_about_provenance(project):
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        with iv.writes("processed/from_nowhere.parquet",
                       why="conjured out of nothing", terminal=True) as p:
            pass
    '''})
    errors, warns = G.check(g)
    assert any("NO PROVENANCE" in w for w in warns), (errors, warns)


def test_external_shows_up_in_the_stage_card(project):
    from invalidator import render
    g = make(project, {"stages/a.py": '''
        from mypipe import iv
        iv.external("espn/scoreboard", why="pulled every morning")
        with iv.writes("raw/games.parquet", why="as fetched", terminal=True) as p:
            pass
    '''})
    card = render.stage_card("stages/a.py", g, color=False)
    assert "from  external:espn/scoreboard" in card
    assert "pulled every morning" in card


def test_no_declared_order_warns_rather_than_passing_quietly(project):
    g = make(project, GOOD)
    errors, warns = G.check(g)
    assert errors == []
    assert any(w.startswith("ORDER  not checked") for w in warns)


# ── export ────────────────────────────────────────────────────────────────────

def test_export_is_the_dbt_manifest_shape(project):
    g = make(project, GOOD, stages=list(GOOD))
    out = G.export(g)
    assert set(out) >= {"nodes", "parent_map", "stage_parent_map"}
    assert out["data_version"] == "v1"
    assert out["nodes"]["processed/ratings.parquet"]["terminal"] is True
    assert out["stage_parent_map"]["stages/build_ratings.py"] == ["stages/build_stats.py"]


def test_a_steps_inputs_are_its_own_not_the_files(project):
    """The decorator clears the read set on entry, so at runtime a step's inputs are
    exactly the reads inside it. The scan has to say the same thing — otherwise a second
    step in the same file contributes a phantom input and every check reports
    'input added' forever."""
    from invalidator import static
    g = make(project, {"stages/two.py": '''
        from mypipe import iv

        @iv.step("processed/a.parquet", why="the first")
        def a(out):
            iv.reads("raw/x.parquet", why="only a reads this")

        @iv.step("processed/b.parquet", why="the second", terminal=True)
        def b(out):
            iv.reads("processed/a.parquet", why="only b reads this")
    '''})
    assert static.inputs_for_artifact(g.iv, "processed/a.parquet") == {"raw/x.parquet": "data"}
    assert static.inputs_for_artifact(g.iv, "processed/b.parquet") == \
        {"processed/a.parquet": "data"}


def test_a_steps_inputs_follow_its_helpers(project):
    """The first stage migrated in anger factored its one read into a helper shared by two
    steps. Attributed lexically, neither step owned any read and both were permanently
    stale — so a step's inputs follow same-file calls, transitively."""
    from invalidator import static
    g = make(project, {"stages/two.py": '''
        from mypipe import iv

        def _source():
            return iv.reads("raw/x.parquet", why="shared by both steps")

        def _extra():
            return iv.reads("raw/only_b.parquet", why="reached only from b")

        @iv.step("processed/a.parquet", why="the first")
        def a(out):
            _source()

        @iv.step("processed/b.parquet", why="the second", terminal=True)
        def b(out):
            _source()
            _extra()
    '''})
    assert static.inputs_for_artifact(g.iv, "processed/a.parquet") == \
        {"raw/x.parquet": "data"}
    assert static.inputs_for_artifact(g.iv, "processed/b.parquet") == \
        {"raw/x.parquet": "data", "raw/only_b.parquet": "data"}


def test_a_function_passed_by_name_is_reached(project):
    """`for_each(seasons, build_one, ...)` never calls build_one syntactically, but its
    reads are the artifact's inputs."""
    from invalidator import static
    g = make(project, {"stages/p.py": '''
        from mypipe import iv

        def build_one(season):
            iv.reads("raw/box/{season}.parquet", why="one season",
                     part={"season": season})

        iv.for_each(["2024"], build_one, output="processed/totals.parquet",
                    key="season", why="per-season totals")
    '''})
    assert static.inputs_for_artifact(g.iv, "processed/totals.parquet") == \
        {"raw/box/{season}.parquet": "data"}


def test_a_steps_inputs_follow_calls_into_other_modules(project):
    """A real pipeline reads through its own library. wvorp's stages call
    `wvorp.data.load`, so a walk that stopped at the file boundary would report them as
    having no inputs and every artifact would be permanently stale."""
    from invalidator import static
    g = make(project, {
        "stages/lib.py": '''
            from mypipe import iv

            def load_panel():
                return iv.reads("processed/panel.parquet", why="read via the library")
        ''',
        "stages/use.py": '''
            from mypipe import iv
            from stages.lib import load_panel

            @iv.step("processed/out.parquet", why="uses the library", terminal=True)
            def build(out):
                load_panel()
        ''',
    })
    assert static.inputs_for_artifact(g.iv, "processed/out.parquet") == \
        {"processed/panel.parquet": "data"}


def test_a_module_attribute_call_is_followed_too(project):
    """`from x import y` then `y.load(...)` records the ATTRIBUTE name, so the module
    itself has to be tried as a target as well as the bare name."""
    from invalidator import static
    g = make(project, {
        "stages/lib.py": '''
            from mypipe import iv

            def load(name):
                return iv.reads("processed/panel.parquet", why="read via the library")
        ''',
        "stages/use.py": '''
            from mypipe import iv
            from stages import lib

            @iv.step("processed/out.parquet", why="uses the library", terminal=True)
            def build(out):
                lib.load("player_box")
        ''',
    })
    assert static.inputs_for_artifact(g.iv, "processed/out.parquet") == \
        {"processed/panel.parquet": "data"}


def test_reads_inside_a_dependency_are_not_ours_to_declare(project):
    """Only modules inside the scanned source dirs are followed."""
    from invalidator import static
    g = make(project, {"stages/use.py": '''
        from mypipe import iv
        import polars as pl

        @iv.step("processed/out.parquet", why="reads nothing of ours", terminal=True)
        def build(out):
            pl.read_parquet("/somewhere/else.parquet")
    '''})
    assert static.inputs_for_artifact(g.iv, "processed/out.parquet") == {}


def test_a_computed_why_is_an_error_not_a_silent_skip(project, iv):
    """The call looks exactly like a declaration and the scan cannot read it, so the
    artifact quietly loses an input. Found the hard way, by a helper taking `why=why`."""
    from conftest import write_stage
    from invalidator.errors import DeclError
    import pytest

    write_stage(project, "stages/roll.py", '''
        from pipeline import iv

        def _declare(path, why):
            iv.reads(path, why=why)

        @iv.step("processed/out.parquet", why="the rollup")
        def build(out):
            _declare("raw/box.parquet", "one raw feed")
    ''')
    (project / "pipeline.py").write_text(
        'from invalidator import Invalidator\n'
        'iv = Invalidator(data_root="data", data_version="v1", source_dirs=["stages"])\n')

    from invalidator import static
    with pytest.raises(DeclError, match="not a string literal"):
        static.scan(iv)


def test_every_declaring_method_is_known_to_the_scan(iv):
    """A method that declares at RUNTIME but is missing from `METHODS` is invisible to
    the scan, and the artifact silently loses the input — the runtime record still looks
    right, which is what makes it hard to spot. `iv.frame` shipped that way for an hour."""
    from invalidator import static

    declaring = {"reads", "frame", "collection", "writes", "updates", "external",
                 "step", "for_each", "partitions"}
    assert declaring <= set(static.METHODS), declaring - set(static.METHODS)
    for name in declaring:
        assert callable(getattr(iv, name)), name


def test_a_repeated_field_must_match_the_same_word():
    """`raw/{dataset}/{dataset}_{season}.parquet` says the directory and the filename
    prefix are the same word. Two independent wildcards also matched
    `raw/predictions/rollforward_v2.parquet`, which drew an edge from that file's writer
    to every reader of a raw feed and made wvorp's whole graph one cycle."""
    from invalidator.static import matches

    t = "raw/{dataset}/{dataset}_{season}.parquet"
    assert matches(t, "raw/player_box/player_box_2026.parquet")
    assert not matches(t, "raw/predictions/rollforward_v2.parquet")
    assert not matches(t, "raw/predictions/predictions_2026.parquet") is False


def test_viz_draws_a_temporal_loop_by_cutting_the_later_edge(project):
    """A ring the run order resolves is still a ring on paper. It gets drawn, with the
    cut edge named — refusing outright would make `iv viz` useless on a pipeline whose
    last stage amends a file an earlier one read."""
    import networkx as nx
    from invalidator.viz import find_cycle

    d = nx.DiGraph()
    d.add_edge("a", "b", stage="early.py")
    d.add_edge("b", "a", stage="late.py")
    assert find_cycle(d) is not None

    order = {"early.py": 0, "late.py": 1}
    pairs = [("a", "b"), ("b", "a")]
    cut = max(pairs, key=lambda e: order[d.edges[e]["stage"]])
    assert cut == ("b", "a"), "the amendment is the edge to cut, not the dependency"


def test_a_slice_says_an_artifact_is_not_one_population(project):
    """`game_predictions` holds the seasons that have been played and the one that has
    not. Calibrating on the first while another stage writes the second is two real
    edges and not a cycle — an argument that lived in a comment until it could be said
    in the declaration."""
    import polars as pl
    from conftest import write_stage
    from invalidator import Invalidator, graph as _graph

    (project / "data" / "raw").mkdir(parents=True)
    pl.DataFrame({"x": [1]}).write_parquet(project / "data" / "raw" / "seed.parquet")
    write_stage(project, "stages/first.py", '''
        import polars as pl
        from pipeline import iv

        @iv.step("processed/preds.parquet", why="predictions", slice="played")
        def build(out):
            pl.read_parquet(iv.reads("raw/seed.parquet", why="the seed")).write_parquet(out)
        build()
    ''')
    write_stage(project, "stages/second.py", '''
        import polars as pl
        from pipeline import iv

        @iv.step("processed/ratings.parquet", why="ratings")
        def build(out):
            iv.reads("processed/preds.parquet", why="predictions", slice="played")
            pl.DataFrame({"r": [1]}).write_parquet(out)
        build()
    ''')
    write_stage(project, "stages/third.py", '''
        import polars as pl
        from pipeline import iv

        iv.reads("processed/ratings.parquet", why="ratings")
        with iv.updates("processed/preds.parquet", why="predictions",
                        slice="upcoming") as p:
            pass
    ''')
    iv = Invalidator(data_root=project / "data", data_version="v1",
                     source_dirs=["stages"], project_root=project,
                     stages=["stages/first.py", "stages/second.py", "stages/third.py"])
    g = _graph.build(iv)
    assert _graph.find_cycle(g) is None, "disjoint slices are not a cycle"
    # The reader of `played` depends on the writer of `played`, and on nobody else.
    assert g.producers_of("processed/preds.parquet", "played") == ["stages/first.py"]
