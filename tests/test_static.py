"""The graph read out of the source, and every check it can fail."""
from __future__ import annotations

import textwrap

import pytest

from dagio import graph as G
from dagio import static
from dagio.errors import DeclError

GOOD = {
    "stages/build_stats.py": '''
        import dagio as dg
        import polars as pl

        def build():
            games = pl.read_parquet(dg.reads(
                "raw/games.parquet", why="one row per team per game", fp="rows"))
            with dg.writes("processed/team_stats.parquet",
                           why="season points by team") as p:
                games.write_parquet(p)
    ''',
    "stages/build_ratings.py": '''
        from dagio import reads, writes
        import polars as pl

        def build():
            stats = pl.read_parquet(reads("processed/team_stats.parquet",
                                          why="season points by team"))
            with writes("processed/ratings.parquet",
                        why="what the app renders", terminal=True) as p:
                stats.write_parquet(p)
    ''',
}


def make(project, files: dict[str, str], order: list[str] | None = None):
    for rel, body in files.items():
        p = project / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body).lstrip())
    if order is not None:
        pyproject = (project / "pyproject.toml").read_text()
        stages = ", ".join(f'"{s}"' for s in order)
        (project / "pyproject.toml").write_text(
            pyproject.replace('data_root = "data"',
                              f'data_root = "data"\nstages = [{stages}]'))
    from dagio import config
    config.reset()
    static.reset()
    return G.build()


# ── the scan ──────────────────────────────────────────────────────────────────

def test_it_finds_calls_through_both_import_styles(project):
    g = make(project, GOOD)
    assert set(g.stages) == set(GOOD)
    assert g.producers_of("processed/team_stats.parquet") == ["stages/build_stats.py"]
    assert g.consumers_of("processed/team_stats.parquet") == ["stages/build_ratings.py"]


def test_metadata_survives_the_scan(project):
    g = make(project, GOOD)
    site = g.sites["raw/games.parquet"][0]
    assert site.why == "one row per team per game"
    assert site.fp == "rows"
    assert g.is_terminal("processed/ratings.parquet")


def test_a_clean_pipeline_has_no_findings(project):
    g = make(project, GOOD, order=list(GOOD))
    errors, warns = G.check(g)
    assert errors == []
    assert warns == []


# ── the literal rule, enforced rather than requested ──────────────────────────

def test_a_computed_path_is_an_error_that_names_the_line(project):
    with pytest.raises(DeclError, match=r"stages/bad\.py:4.*string LITERAL"):
        make(project, {"stages/bad.py": '''
            import dagio as dg
            NAME = "xpm"
            def build():
                dg.reads(f"processed/{NAME}.parquet", why="computed on purpose")
        '''})


def test_a_missing_why_is_an_error(project):
    with pytest.raises(DeclError, match="why= is required"):
        make(project, {"stages/bad.py": '''
            import dagio as dg
            def build():
                dg.reads("raw/games.parquet")
        '''})


def test_a_template_without_part_is_an_error(project):
    with pytest.raises(DeclError, match=r"placeholder\(s\) \['season'\] but no part="):
        make(project, {"stages/bad.py": '''
            import dagio as dg
            def build():
                dg.reads("raw/box/{season}.parquet", why="per season")
        '''})


def test_part_keys_must_match_the_template(project):
    with pytest.raises(DeclError, match="part= supplies"):
        make(project, {"stages/bad.py": '''
            import dagio as dg
            def build(year):
                dg.reads("raw/box/{season}.parquet", why="per season",
                         part={"year": year})
        '''})


# ── the checks ────────────────────────────────────────────────────────────────

def test_write_with_no_consumer(project):
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        def build():
            dg.reads("raw/games.parquet", why="the source")
            with dg.writes("processed/dead.parquet", why="nobody reads this") as p:
                pass
    '''})
    errors, _ = G.check(g)
    assert any("WRITE WITH NO CONSUMER  processed/dead.parquet" in e for e in errors)


def test_terminal_makes_no_consumer_correct(project):
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        def build():
            dg.reads("raw/games.parquet", why="the source")
            with dg.writes("dump/site.json", why="the app renders it",
                           terminal=True, fp="bytes") as p:
                pass
    '''})
    errors, _ = G.check(g)
    assert not any("NO CONSUMER" in e for e in errors)


def test_read_with_no_producer(project):
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        def build():
            dg.reads("processed/nobody_makes_this.parquet", why="a missing stage")
            with dg.writes("dump/out.json", why="terminal", terminal=True) as p:
                pass
    '''})
    errors, _ = G.check(g)
    assert any("READ WITH NO PRODUCER" in e for e in errors)


def test_two_writers(project):
    body = '''
        import dagio as dg
        def build():
            dg.reads("raw/games.parquet", why="the source")
            with dg.writes("processed/x.parquet", why="the same artifact",
                           terminal=True) as p:
                pass
    '''
    g = make(project, {"stages/a.py": body, "stages/b.py": body})
    errors, _ = G.check(g)
    assert any("TWO WRITERS" in e for e in errors)


def test_policy_conflict(project):
    g = make(project, {
        "stages/a.py": '''
            import dagio as dg
            def build():
                dg.reads("raw/games.parquet", why="the source")
                with dg.updates("processed/x.parquet", why="first pass",
                                policy="manual", terminal=True) as p:
                    pass
        ''',
        "stages/b.py": '''
            import dagio as dg
            def build():
                dg.reads("raw/games.parquet", why="the source")
                with dg.updates("processed/x.parquet", why="second pass",
                                policy="tracked", terminal=True) as p:
                    pass
        ''',
    })
    errors, _ = G.check(g)
    assert any("POLICY CONFLICT" in e and "policy=" in e for e in errors)


def test_a_cycle(project):
    g = make(project, {
        "stages/a.py": '''
            import dagio as dg
            def build():
                dg.reads("processed/b.parquet", why="from b")
                with dg.writes("processed/a.parquet", why="to a") as p:
                    pass
        ''',
        "stages/b.py": '''
            import dagio as dg
            def build():
                dg.reads("processed/a.parquet", why="from a")
                with dg.writes("processed/b.parquet", why="to b") as p:
                    pass
        ''',
    })
    errors, _ = G.check(g)
    assert any(e.startswith("CYCLE") for e in errors)


def test_updates_is_not_a_cycle(project):
    """A stage that reads its own output to decide what to skip is an incremental cache,
    not a cycle. Without updates() the only ways out are a false alarm or an allowlist."""
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        def build():
            dg.reads("raw/games.parquet", why="the source")
            with dg.updates("processed/cache.parquet",
                            why="reads its own output to decide what to skip",
                            terminal=True) as p:
                pass
    '''})
    errors, _ = G.check(g)
    assert not any(e.startswith("CYCLE") for e in errors), errors


def test_order_violation_is_found(project):
    g = make(project, GOOD, order=["stages/build_ratings.py", "stages/build_stats.py"])
    errors, _ = G.check(g)
    assert any(e.startswith("ORDER") for e in errors)


def test_prior_makes_a_late_producer_legal(project):
    files = dict(GOOD)
    files["stages/build_ratings.py"] = files["stages/build_ratings.py"].replace(
        'why="season points by team"))',
        'why="season points by team", prior=True))')
    g = make(project, files, order=["stages/build_ratings.py", "stages/build_stats.py"])
    errors, _ = G.check(g)
    assert not any(e.startswith("ORDER") for e in errors), errors


def test_guarding_a_fetch_is_an_error(project):
    """The guard against this package's own silent failure.

    Nothing in a fetched artifact's id can move on its own, so guarding the stage that
    produces it means it runs once and never again — which looks exactly like the cache
    working perfectly.
    """
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        def build():
            dg.external("vendor/feed", why="the upstream drop")
            with dg.writes("raw/games.parquet", why="as fetched", terminal=True) as p:
                pass
        dg.build_if_needed("raw/games.parquet", build, if_needed=True)
    '''})
    errors, _ = G.check(g)
    assert any("GUARDED FETCH  raw/games.parquet" in e for e in errors), errors


def test_an_unguarded_fetch_is_fine(project):
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        dg.external("vendor/feed", why="the upstream drop")
        with dg.writes("raw/games.parquet", why="as fetched", terminal=True) as p:
            pass
    '''})
    errors, warns = G.check(g)
    assert not any("GUARDED FETCH" in e for e in errors), errors
    assert not any("NO PROVENANCE" in w for w in warns), warns


def test_settled_says_the_staleness_question_does_not_apply(project):
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        def build():
            dg.external("vendor/archive", why="a season that will never change")
            with dg.writes("raw/history.parquet", why="fetch-once history",
                           policy="settled", terminal=True) as p:
                pass
        dg.build_if_needed("raw/history.parquet", build, if_needed=True)
    '''})
    errors, _ = G.check(g)
    assert not any("GUARDED FETCH" in e for e in errors), errors


def test_an_artifact_from_nothing_warns_about_provenance(project):
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        with dg.writes("processed/from_nowhere.parquet",
                       why="conjured out of nothing", terminal=True) as p:
            pass
    '''})
    errors, warns = G.check(g)
    assert any("NO PROVENANCE" in w for w in warns), (errors, warns)


def test_external_shows_up_in_the_stage_card(project):
    from dagio import render
    g = make(project, {"stages/a.py": '''
        import dagio as dg
        dg.external("espn/scoreboard", why="pulled every morning")
        with dg.writes("raw/games.parquet", why="as fetched", terminal=True) as p:
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
    g = make(project, GOOD, order=list(GOOD))
    out = G.export(g)
    assert set(out) >= {"nodes", "parent_map", "stage_parent_map"}
    assert out["nodes"]["processed/ratings.parquet"]["terminal"] is True
    assert out["stage_parent_map"]["stages/build_ratings.py"] == ["stages/build_stats.py"]
