"""Every shape a real pipeline uses, in one file. Run it: `uv run python example.py`

Modelled on a working sports-model pipeline, so the awkward cases are here rather than the
tidy ones:

    config/today/            the clock, as a file — a "re-run daily" policy, stored
    config/model/            the knobs, as a file — a model version, stored
    raw/box_settled/         finished seasons, fetched once each
    raw/box_live/            the season being played, polled — ONE shard
    raw/box/                 the two halves, per season, put together
    derived/schedule/        an UPDATE: read the copy on disk, amend it, write it back
    derived/box_features/    ONE computation, split into a shard per season
    derived/college/         THREE stages, three blocks, one dataset
    processed/xpm/ + 3 more  ONE joint fit, four tables, three of them terminal
    processed/rapm_fit/      not every artifact is a table: a pickled model object
    processed/xpm_eoy/       walk-forward, INCLUSIVE: fit through the end of season T
    processed/rookie/        walk-forward, EXCLUSIVE: fit on seasons strictly before T
    processed/predictions/   two stages, the played and unplayed halves
    dump/site/               terminal JSON, written through `out`

THERE IS NO INDEX. A derived shard is named `<part>.<key>.<fp>`: the key is a digest of the
upstreams it was built from, the fp a digest of its bytes. "Is it current" recomputes the
key from the declared upstreams and the files on disk and asks whether that name is here —
so the record cannot go missing, go stale, or disagree with the tree.

WHAT IS DECLARED. Upstreams are parameter defaults, so `inspect.signature` reads the whole
declaration — dataset, selector, partition key — with nothing executed and no source text
needed. The walk-forward bounds below are DATA, which is what lets a shard's key be
computed before its body runs.
"""

import datetime as dt
import json
import pathlib
import shutil
import tempfile

import polars as pl

from iv import Invalidator

root = pathlib.Path(tempfile.mkdtemp()) / "data"
iv = Invalidator(tree=root, project=root.parent, sources=("raw/", "config/"))

SEASONS = ["2022", "2023", "2024"]
LIVE = "2024"          # the season being played; the rest have settled
TODAY = dt.date(2026, 8, 21)
RIDGE = 4.0
PTS = {"2022": 10, "2023": 20, "2024": 30, "2025": 40}
ran = []


# ── metadata is a file, and a root always runs ───────────────────────────────


@iv.data(
    "config/today/", why="the day, so a polled feed re-fetches once a day", ext=".json"
)
def today():
    """No upstream, so nothing on disk could say the date moved — it runs every time.

    That is what lets the outside world in. The commit is content-addressed, so a run on
    the same day writes the same bytes and moves nothing downstream.
    """
    return {"date": TODAY.isoformat()}


@iv.data("config/model/", why="the knobs the fit shape depends on", ext=".json")
def model_config():
    """Not a version string, not a label — a shard, declared as an upstream by whoever
    answers to it. `iv graph` draws the edge; a change moves exactly those stages."""
    return {"ridge": RIDGE, "epochs": 300, "seed": 0}


# ── the settled half and the live half, and putting them together ─────────────
#
# A feed has two halves that behave nothing alike, and the shape below is the whole reason
# a daily run stays cheap. A FINISHED season never changes: fetch it once. The season being
# played changes all day: poll it. Declared as one partitioned dataset reading the clock,
# every season would be stale the moment the day turned — twenty years of history rebuilt
# to pick up one afternoon. Declared as two datasets, the clock touches exactly one shard.


@iv.data(
    "raw/box_settled/",
    why="raw box scores for one finished season",
    part="season",
    once=True,
    external={"espn/feeds": "ESPN's per-season files"},
)
def box_settled(season):
    """`once=True` is PER SHARD: a season already fetched is left alone, one added to the
    list is fetched. Nothing polls it, because nothing about it moves."""
    ran.append(f"settled:{season}")
    return pl.DataFrame(
        {"season": [season] * 2, "player": [1, 2],
         "pts": [PTS[season], PTS[season] + 5]}
    )


@iv.data(
    "raw/box_live/",
    why="raw box scores for the season being played",
    part={"season": LIVE},
    external={"espn/feeds": "ESPN's live feed"},
)
def box_live(clock=iv.all_of("config/today/", load=False, why="poll once a day")):
    """ONE shard, named outright with a literal `part=`. It reads the clock, so it re-fetches
    when the day turns — and it is the only thing that does."""
    ran.append("live")
    return pl.DataFrame(
        {"season": [LIVE] * 2, "player": [1, 2], "pts": [PTS[LIVE], PTS[LIVE] + 5]}
    )


@iv.data("raw/box/", why="one season of box scores, from whichever half has it",
         part="season")
def box(
    season,
    settled=iv.same_part("raw/box_settled/", optional=True, why="a finished season"),
    live=iv.same_part("raw/box_live/", optional=True, why="the season being played"),
):
    """Both halves are declared and one is used. A declaration resolves every read before
    the body runs, so the half this season is not in has to be optional rather than absent —
    and because the selector is `same_part`, a season only ever depends on its OWN shard of
    either half. The day turning moves `raw/box_live/`, which moves this season and no other.
    """
    ran.append(f"box:{season}")
    return live if season == LIVE else settled


# ── an update: read your own last copy, amend it, write it back ───────────────


@iv.data(
    "derived/schedule/",
    why="one row per game, with scores patched in as they land",
    external={"espn/scoreboard": "final scores, which land all day"},
)
def schedule(
    clock=iv.all_of("config/today/", why="scores land all day; re-check daily"),
    was=iv.own_last_copy("derived/schedule/", why="the copy this amends"),
):
    """READ-MODIFY-WRITE, which used to need its own primitive.

    `own_last_copy` says: this is the copy on disk I am about to overwrite. It is recorded
    for lineage and EXCLUDED from the comparison — otherwise the stage would be permanently
    stale against its own last output, one step behind itself, forever.
    """
    ran.append("schedule")
    old = (
        was if was is not None else pl.DataFrame(schema={"day": pl.Utf8, "n": pl.Int64})
    )
    new = pl.DataFrame({"day": [TODAY.isoformat()], "n": [len(old) + 1]})
    return pl.concat([old, new]).unique(subset="day", keep="last").sort("day")


# ── one computation, split into a shard per season ───────────────────────────


@iv.data(
    "derived/box_features/",
    why="the per-(season, player) box matrix",
    part="season",
    split=True,
)
def box_features(
    box=iv.all_of("raw/box/", why="every season of raw box scores"),
    sched=iv.all_of("derived/schedule/", why="which games count"),
):
    """The features have career-cumulative terms, so they are built in ONE pass and split.

    `split=True` says the body returns {partition: value} — one expensive computation, many
    shards, each of which downstream can then depend on separately.
    """
    ran.append("box_features")
    bf = box.with_columns((pl.col("pts") * 2).alias("z"))
    return {str(s): rows for (s,), rows in bf.group_by("season", maintain_order=True)}


# ── three stages, three blocks, one dataset ──────────────────────────────────


@iv.data(
    "derived/college/",
    why="the NCAA block of the college feature table",
    part={"source": "ncaa"},
)
def ncaa_block(
    bf=iv.all_of("derived/box_features/", why="the pro side to rank against")
):
    """A LITERAL part= is how several stages share one dataset. Each owns exactly one shard,
    so the graph can see they do not collide — without it, whichever ran last would win.
    """
    ran.append("ncaa")
    return pl.DataFrame({"source": ["ncaa"], "n": [bf.height]})


@iv.data("derived/college/", why="the G-League block", part={"source": "gleague"})
def gleague_block(bf=iv.all_of("derived/box_features/", why="the pro side")):
    ran.append("gleague")
    return pl.DataFrame({"source": ["gleague"], "n": [1]})


@iv.data(
    "derived/college/",
    why="the international block",
    part={"source": "intl"},
    external={"basketball-reference/international": "the international player pages"},
)
def intl_block(bf=iv.all_of("derived/box_features/", why="the pro side")):
    ran.append("intl")
    return pl.DataFrame({"source": ["intl"], "n": [1]})


# ── one fit, several outputs ─────────────────────────────────────────────────


@iv.step(
    outputs={
        "ratings": "processed/xpm/",
        "career": iv.output("processed/xpm_career/", terminal=True),
        "summary": iv.output("processed/xpm_summary/", terminal=True),
        "levels": iv.output("processed/xpm_levels/", terminal=True),
    },
    why="the joint fit and the tables that fall out of it",
)
def xpm(
    knobs=iv.all_of("config/model/", why="a knob change must refit"),
    bf=iv.all_of("derived/box_features/", why="the box prior, every season at once"),
    college=iv.all_of("derived/college/", why="the college block, all three sources"),
):
    """ONE expensive computation, four tables. Declaring them together is what stops the fit
    being run once per output — and losing any one of them brings the whole fit back.

    No selector on `bf`: every season at once. That is what makes this a JOINT fit rather
    than a per-season one, and it is visible in the signature rather than buried in a body.
    """
    ran.append("xpm")
    r = bf.group_by("player", maintain_order=True).agg(
        (pl.col("z") * knobs["ridge"]).mean()
    )
    return {
        "ratings": r,
        "career": r.head(1),
        "summary": r.select(pl.col("z").mean().alias("mean_z")),
        "levels": r.head(1),
    }


@iv.data(
    "processed/rapm_fit/",
    why="the fitted model object, so nothing refits it twice",
    ext=".pkl",
)
def rapm_fit(
    knobs=iv.all_of("config/model/", why="the fit shape"),
    bf=iv.all_of("derived/box_features/", why="the design matrix"),
):
    """Not every artifact is a table. `.pkl` round trips whatever the body returned."""
    ran.append("rapm_fit")
    return {"coefs": [1.0, 2.0], "ridge": knobs["ridge"]}


# ── walk-forward, two bounds ─────────────────────────────────────────────────


@iv.data(
    "processed/xpm_eoy/",
    why="one end-of-year rating per season, frozen once it ends",
    part="season",
)
def xpm_eoy(
    season,
    knobs=iv.all_of("config/model/", why="a knob change must refit this season"),
    bf=iv.before_part(
        "derived/box_features/",
        inclusive=True,
        why="the box matrix through the END of this season",
    ),
):
    """INCLUSIVE: `le`, so season T's own rows are in its own fit and nothing later is."""
    ran.append(f"eoy:{season}")
    return bf.select(pl.lit(season).alias("season"), pl.col("z").mean())


@iv.data(
    "processed/rookie/",
    why="a projection per cohort, on prior seasons only",
    part="season",
)
def rookie(
    bf=iv.before_part("derived/box_features/", why="strictly before this cohort"),
    college=iv.all_of("derived/college/", why="the college block"),
):
    """EXCLUSIVE: `lt`. The bound picks FILES, so a cohort physically cannot open its own
    season or a later one. A season backfilled BELOW the bound is picked up; one added
    above it is not."""
    ran.append("rookie")
    return bf.select(pl.col("z").mean().alias("prior_mean"))


# ── two stages, the played and unplayed halves of one dataset ────────────────


@iv.data(
    "processed/predictions/",
    why="one predicted margin per game already played",
    part={"completed": "true"},
)
def predict_played(
    x=iv.all_of("processed/xpm/", why="the ratings each game is priced off"),
    sched=iv.all_of("derived/schedule/", why="which games were played"),
):
    ran.append("predict_played")
    return pl.DataFrame({"game": [1], "margin": [3.5]})


@iv.data(
    "processed/predictions/",
    why="one predicted margin per game not yet played",
    part={"completed": "false"},
)
def predict_upcoming(
    x=iv.all_of("processed/xpm/", why="the ratings"),
    sched=iv.all_of("derived/schedule/", why="the remaining schedule"),
):
    ran.append("predict_upcoming")
    return pl.DataFrame({"game": [2], "margin": [-1.5]})


@iv.data(
    "processed/calibration/",
    why="the sigma the predictions are calibrated with",
    allow_missing=True,
)
def calibration(
    played=iv.parts(
        "processed/predictions/",
        completed=["true"],
        why="played games only — an unplayed one has no residual",
    )
):
    """`parts()` is an explicit COVERAGE CLAIM: a named partition that is not there is an
    error rather than a quietly shorter read. `allow_missing=True` says producing nothing
    is legitimate — the shard stays absent and the next run tries again."""
    ran.append("calibration")
    return (
        None if played.height == 0 else pl.DataFrame({"sigma": [float(played.height)]})
    )


# ── terminal, written through `out` ──────────────────────────────────────────


@iv.data("dump/site/", why="the payload the app renders", ext=".json", terminal=True)
def site(
    out,
    x=iv.all_of("processed/xpm/", why="the leaderboard"),
    fit=iv.all_of("processed/rapm_fit/", why="the fit, for the ridge it used"),
    eoy=iv.all_of("processed/xpm_eoy/", why="the end-of-year column"),
    rk=iv.all_of("processed/rookie/", why="the rookie projections"),
    preds=iv.all_of("processed/predictions/", why="today's games"),
    cal=iv.all_of(
        "processed/calibration/", optional=True, why="the sigma, once there is one"
    ),
):
    """A body that takes `out` writes the file itself, and nothing is inferred about a
    return value. That is the escape hatch for anything the formats do not cover."""
    ran.append("site")
    out.write_text(
        json.dumps(
            {
                "players": x.height,
                "eoy": eoy.height,
                "rookies": rk.height,
                "games": preds.height,
                "ridge": fit["ridge"],  # a .pkl, round-tripped
                "calibrated": cal is not None,
            },
            indent=1,
        )
    )


# ── run it ────────────────────────────────────────────────────────────────────


def build_all():
    today()
    model_config()
    box_settled.for_each([s for s in SEASONS if s != LIVE])
    box_live()
    box.for_each(SEASONS)
    schedule()
    box_features()
    ncaa_block()
    gleague_block()
    intl_block()
    xpm()
    rapm_fit()
    xpm_eoy.for_each(SEASONS[:-1])
    rookie.for_each(SEASONS[1:])
    predict_played()
    predict_upcoming()
    calibration()
    site()


def run(label):
    print(f"\n=== {label} ===")
    ran.clear()
    build_all()
    print(f"    ran: {ran or ['nothing']}")


def tree():
    print("\n--- what is on disk ---")
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        print(f"    {f.relative_to(root)}")


run("first run")
tree()

run("nothing changed")

TODAY = dt.date(2026, 8, 22)
run("a new day: the schedule is amended, and only what reads it follows")

# A settled season is fetched ONCE, so a correction upstream is not seen and iv does not
# pretend otherwise — `iv check` warns that raw/box_settled/ runs once, which is this said
# in advance. Picking it up is a deliberate act: IV_FORCE=1, or delete the shard.
PTS["2022"] = 999
run("a SETTLED season is corrected upstream: fetched once, so nothing follows")

TODAY = dt.date(2026, 8, 23)
run("another day, and nothing else moved: only the poll re-runs")

PTS[LIVE] = 999
TODAY = dt.date(2026, 8, 24)
run("a score lands in the LIVE season: one shard moves, and the tail follows")

SEASONS.append("2025")
run("a new season lands")

RIDGE = 3.5
run("a knob changes: the fits move, the features do not")

shutil.rmtree(root / "processed/xpm_summary")
run("one of the fit's four outputs is deleted")

tree()


# ── what the tool knows, having run nothing ──────────────────────────────────

from iv import graph as _graph  # noqa: E402
from iv import render as _render  # noqa: E402

print("\n\n=== the graph, from the declarations alone ===\n")
g = _graph.build(iv)
order = g.order()
print(_render.render(order, _render.transitive_reduction(order, g.parent_map())))

errors, warns = _graph.check(g)
print(f"\ncheck: {len(errors)} error(s), {len(warns)} warning(s)")
for w in warns:
    print("  warn ", w.splitlines()[0])
for e in errors:
    print("  ERROR", e.splitlines()[0])


# ── what it looks like when something is wrong ────────────────────────────────
#
# Every one of these used to pass quietly. They are the reason the pipeline can be trusted
# to skip: anything the tool cannot account for stops the run instead of guessing.

print("\n\n=== things that crash, and what they say ===")


def shows(label, fn):
    try:
        fn()
        print(f"\n  {label}\n      NO ERROR — that is a bug")
    except Exception as e:
        print(
            f"\n  {label}\n      {type(e).__name__}: {' '.join(str(e).split())[:270]}"
        )


def dict_to_parquet():
    @iv.data("processed/bad_knobs/", why="a dict, but no ext=")
    def bad_knobs():
        return {"a": 1}

    bad_knobs()


def part_relative_with_no_partition():
    @iv.data("processed/bad_bound/", why="not partitioned, but reads as if it were")
    def bad_bound(bf=iv.same_part("derived/box_features/", why="this season")):
        return bf


def a_second_writer_of_the_same_shard():
    @iv.data("derived/college/", why="a second NCAA block", part={"source": "ncaa"})
    def ncaa_again(bf=iv.all_of("derived/box_features/", why="the pro side")):
        return bf


def an_output_that_was_not_returned():
    @iv.data("processed/two_out/", why="declares two, returns one")
    def _unused():
        return None

    @iv.step(
        outputs={"a": "processed/out_a/", "b": "processed/out_b/"},
        why="declares two outputs and returns one",
    )
    def two(bf=iv.all_of("derived/box_features/", why="the box prior")):
        return {"a": bf}

    two()


def building_a_stage_from_inside_one():
    @iv.data("processed/reaches/", why="reaches for a stage instead of declaring it")
    def reaches(unused=iv.all_of("config/today/", why="declared, then ignored")):
        return box("2024")

    reaches()


def a_parameter_iv_cannot_supply():
    @iv.data("processed/mystery/", why="takes something unexplained")
    def mystery(what_is_this):
        return what_is_this


shows("a dict returned to a .parquet dataset", dict_to_parquet)
shows("a partition-relative read with no part=", part_relative_with_no_partition)
shows("two stages writing the same shard", a_second_writer_of_the_same_shard)
shows("a declared output the body did not return", an_output_that_was_not_returned)
shows("building one stage from inside another", building_a_stage_from_inside_one)
shows("a parameter iv cannot supply", a_parameter_iv_cannot_supply)
shows(
    "a read that selects nothing",
    lambda: iv.reads("raw/nothing_here/", why="not there"),
)
