

import datetime as dt
import json
import pathlib
import shutil
import tempfile

import polars as pl

from iv import Pipeline

root = pathlib.Path(tempfile.mkdtemp()) / "data"
iv = Pipeline(tree=root, project=root.parent)

SEASONS = ["2022", "2023", "2024"]
LIVE = "2024"
TODAY = dt.date(2026, 8, 21)
RIDGE = 4.0
PTS = {"2022": 10, "2023": 20, "2024": 30, "2025": 40}
ran = []


bios = iv.source(
    "raw/bios/",
    why="heights and weights, dropped in by hand once a year",
    external={"basketball-reference/players": "the player pages"},
)


@iv.data(
    dataset="config/today/",
    why="the day, so a polled feed re-fetches once a day",
    ext=".json",
)
def today():


    return {"date": TODAY.isoformat()}


@iv.data(dataset="config/model/", why="the knobs the fit shape depends on", ext=".json")
def model_config():


    return {"ridge": RIDGE, "epochs": 300, "seed": 0}


@iv.data(
    dataset="raw/box_settled/",
    why="raw box scores for one finished season",
    part="season",
    once=True,
    external={"espn/feeds": "ESPN's per-season files"},
)
def box_settled(season):


    ran.append(f"settled:{season}")
    return pl.DataFrame(
        {"season": [season] * 2, "player": [1, 2],
         "pts": [PTS[season], PTS[season] + 5]}
    )


@iv.data(
    dataset="raw/box_live/",
    why="raw box scores for the season being played",
    part={"season": LIVE},
    external={"espn/feeds": "ESPN's live feed"},
)
def box_live(clock=iv.all_of(today, as_paths=True, why="poll once a day")):


    ran.append("live")
    return pl.DataFrame(
        {"season": [LIVE] * 2, "player": [1, 2], "pts": [PTS[LIVE], PTS[LIVE] + 5]}
    )


@iv.data(dataset="raw/box/", why="one season of box scores, from whichever half has it",
         part="season")
def box(
    season,
    settled=iv.same_part(box_settled, optional=True, why="a finished season"),
    live=iv.same_part(box_live, optional=True, why="the season being played"),
):


    ran.append(f"box:{season}")
    return live if season == LIVE else settled


@iv.data(
    dataset="derived/schedule/",
    why="one row per game, with scores patched in as they land",
    external={"espn/scoreboard": "final scores, which land all day"},
)
def schedule(
    clock=iv.all_of(today, why="scores land all day; re-check daily"),
    was=iv.own_last_copy(why="the copy this amends"),
):


    ran.append("schedule")
    old = (
        was if was is not None else pl.DataFrame(schema={"day": pl.Utf8, "n": pl.Int64})
    )
    new = pl.DataFrame({"day": [TODAY.isoformat()], "n": [len(old) + 1]})
    return pl.concat([old, new]).unique(subset="day", keep="last").sort("day")


@iv.data(
    dataset="derived/box_features/",
    why="the per-(season, player) box matrix",
    part="season",
    split=True,
)
def box_features(
    box=iv.all_of(box, why="every season of raw box scores"),
    sched=iv.all_of(schedule, why="which games count"),
    bio=iv.all_of(bios, as_paths=True, why="the body-shape columns"),
):


    ran.append("box_features")
    bf = box.with_columns((pl.col("pts") * 2).alias("z"))
    return {str(s): rows for (s,), rows in bf.group_by("season", maintain_order=True)}


COLLEGE = iv.dataset(
    "derived/college/", why="one row per amateur source, ranked against the pros"
)


@iv.data(
    dataset=COLLEGE,
    why="the NCAA block of the college feature table",
    part={"source": "ncaa"},
)
def ncaa_block(
    bf=iv.all_of(box_features, why="the pro side to rank against")
):


    ran.append("ncaa")
    return pl.DataFrame({"source": ["ncaa"], "n": [bf.height]})


@iv.data(dataset=COLLEGE, why="the G-League block", part={"source": "gleague"})
def gleague_block(bf=iv.all_of(box_features, why="the pro side")):
    ran.append("gleague")
    return pl.DataFrame({"source": ["gleague"], "n": [1]})


@iv.data(
    dataset=COLLEGE,
    why="the international block",
    part={"source": "intl"},
    external={"basketball-reference/international": "the international player pages"},
)
def intl_block(bf=iv.all_of(box_features, why="the pro side")):
    ran.append("intl")
    return pl.DataFrame({"source": ["intl"], "n": [1]})


XPM = iv.dataset("processed/xpm/", why="a rating per player per season")
XPM_CAREER = iv.dataset("processed/xpm_career/", why="one row per player, career to date")
XPM_SUMMARY = iv.dataset("processed/xpm_summary/", why="what the fit did, per season")
XPM_LEVELS = iv.dataset("processed/xpm_levels/", why="the level each season sits at")


@iv.step(
    output={
        "ratings": XPM,
        "career": XPM_CAREER,
        "summary": XPM_SUMMARY,
        "levels": XPM_LEVELS,
    },
    why="the joint fit and the tables that fall out of it",
)
def xpm(
    knobs=iv.all_of(model_config, why="a knob change must refit"),
    bf=iv.all_of(box_features, why="the box prior, every season at once"),
    college=iv.all_of(COLLEGE, why="the college block, all three sources"),
):


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
    dataset="processed/rapm_fit/",
    why="the fitted model object, so nothing refits it twice",
    ext=".pkl",
)
def rapm_fit(
    knobs=iv.all_of(model_config, why="the fit shape"),
    bf=iv.all_of(box_features, why="the design matrix"),
):

    ran.append("rapm_fit")
    return {"coefs": [1.0, 2.0], "ridge": knobs["ridge"]}


@iv.data(
    dataset="processed/xpm_eoy/",
    why="one end-of-year rating per season, frozen once it ends",
    part="season",
)
def xpm_eoy(
    season,
    knobs=iv.all_of(model_config, why="a knob change must refit this season"),
    bf=iv.before_part(
        box_features,
        inclusive=True,
        why="the box matrix through the END of this season",
    ),
):

    ran.append(f"eoy:{season}")
    return bf.select(pl.lit(season).alias("season"), pl.col("z").mean())


@iv.data(
    dataset="processed/rookie/",
    why="a projection per cohort, on prior seasons only",
    part="season",
)
def rookie(
    bf=iv.before_part(box_features, why="strictly before this cohort"),
    college=iv.all_of(COLLEGE, why="the college block"),
):


    ran.append("rookie")
    return bf.select(pl.col("z").mean().alias("prior_mean"))


PREDICTIONS = iv.dataset("processed/predictions/", why="one predicted margin per game")


@iv.data(
    dataset=PREDICTIONS,
    why="one predicted margin per game already played",
    part={"completed": "true"},
)
def predict_played(
    x=iv.all_of(XPM, why="the ratings each game is priced off"),
    sched=iv.all_of(schedule, why="which games were played"),
):
    ran.append("predict_played")
    return pl.DataFrame({"game": [1], "margin": [3.5]})


@iv.data(
    dataset=PREDICTIONS,
    why="one predicted margin per game not yet played",
    part={"completed": "false"},
)
def predict_upcoming(
    x=iv.all_of(XPM, why="the ratings"),
    sched=iv.all_of(schedule, why="the remaining schedule"),
):
    ran.append("predict_upcoming")
    return pl.DataFrame({"game": [2], "margin": [-1.5]})


@iv.data(
    dataset="processed/calibration/",
    why="the sigma the predictions are calibrated with",
    allow_missing=True,
)
def calibration(
    played=iv.parts(
        PREDICTIONS,
        completed=["true"],
        why="played games only — an unplayed one has no residual",
    )
):


    ran.append("calibration")
    return (
        None if played.height == 0 else pl.DataFrame({"sigma": [float(played.height)]})
    )


@iv.data(dataset="dump/site/", why="the payload the app renders", ext=".json")
def site(
    out,
    x=iv.all_of(XPM, why="the leaderboard"),
    fit=iv.all_of(rapm_fit, why="the fit, for the ridge it used"),
    eoy=iv.all_of(xpm_eoy, why="the end-of-year column"),
    rk=iv.all_of(rookie, why="the rookie projections"),
    preds=iv.all_of(PREDICTIONS, why="every game, played and upcoming"),
    cal=iv.all_of(calibration, optional=True, why="the sigma, once there is one"),
):


    ran.append("site")
    out.write_text(
        json.dumps(
            {
                "players": x.height,
                "eoy": eoy.height,
                "rookies": rk.height,
                "games": preds.height,
                "ridge": fit["ridge"],
                "calibrated": cal is not None,
            },
            indent=1,
        )
    )


def build_all():
    if not (root / "raw/bios").exists():
        with iv.writes("raw/bios/", why="dropped in by hand") as out:
            pl.DataFrame({"player": [1, 2], "cm": [180, 191]}).write_parquet(out)
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


from iv import graph as _graph
from iv import render as _render

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
    @iv.data(dataset="processed/bad_knobs/", why="a dict, but no ext=")
    def bad_knobs():
        return {"a": 1}

    bad_knobs()


def part_relative_with_no_partition():
    @iv.data(dataset="processed/bad_bound/",
             why="not partitioned, but reads as if it were")
    def bad_bound(bf=iv.same_part(box_features, why="this season")):
        return bf


def a_second_writer_of_the_same_shard():
    @iv.data(dataset="derived/college/", why="a second NCAA block",
             part={"source": "ncaa"})
    def ncaa_again(bf=iv.all_of(box_features, why="the pro side")):
        return bf


def an_output_that_was_not_returned():
    @iv.data(dataset="processed/two_out/", why="declares two, returns one")
    def _unused():
        return None

    @iv.step(
        output={"a": "processed/out_a/", "b": "processed/out_b/"},
        why="declares two outputs and returns one",
    )
    def two(bf=iv.all_of(box_features, why="the box prior")):
        return {"a": bf}

    two()


def building_a_stage_from_inside_one():
    @iv.data(dataset="processed/reaches/",
             why="reaches for a stage instead of declaring it")
    def reaches(unused=iv.all_of(today, why="declared, then ignored")):
        return box("2024")

    reaches()


def a_parameter_iv_cannot_supply():
    @iv.data(dataset="processed/mystery/", why="takes something unexplained")
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
