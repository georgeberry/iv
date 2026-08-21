"""One small pipeline exercising every shape wvorp has. Run it: `uv run python repro.py`.

config/today/           the clock, as a file — what a "re-run daily" policy used to be
config/hyperparams/     metadata, as a file — what a model version used to be
raw/box/                sharded by season, one file per season
raw/odds_log/           an UPDATE: read yesterday's copy, append, write it back
processed/features/     sharded by season, built one season at a time
processed/xpm/          a JOINT fit — reads every season at once
processed/xpm_summary/  ...a second output of the same fit
dump/site/              terminal: something outside the pipeline reads it
"""

import datetime as dt
import pathlib
import shutil
import tempfile

import polars as pl

from iv.core import Pipeline

root = pathlib.Path(tempfile.mkdtemp()) / "data"
iv = Pipeline(root=root, project_root=root.parent)

SEASONS = ["2024", "2025"]
TODAY = dt.date(2026, 8, 21)
HALF_LIFE = 4.0
ran = []


# ── metadata is a file ────────────────────────────────────────────────────────


def config():
    """Not a version string, not a label — a shard, declared as an upstream by whoever
    answers to it. `iv graph` draws the edge; a change moves exactly those stages."""
    iv.constants("config/today/", why="poll once a day", date=TODAY.isoformat())
    iv.constants(
        "config/hyperparams/",
        why="the knobs the fit shape depends on",
        half_life=HALF_LIFE,
        epochs=300,
        seed=0,
    )


# ── partitions ────────────────────────────────────────────────────────────────


def fetch(season, pts):
    """One shard per season. `part=` lands in the FILENAME, not in the dataset path."""
    with iv.writes(
        "raw/box/", why="raw box scores for one season", part={"season": season}
    ) as out:
        pl.DataFrame({"player": [1, 2], "pts": [pts, pts + 5]}).write_parquet(out)


def features():
    """One shard per season. Only the seasons that moved get rebuilt."""

    def one(season, out):
        ran.append(f"features:{season}")
        box = pl.read_parquet(
            iv.reads(
                "raw/box/", why="raw box for this season", where={"season": [season]}
            )
        )
        box.with_columns((pl.col("pts") * 2).alias("z")).write_parquet(out)

    return iv.for_each(
        SEASONS,
        one,
        dataset="processed/features/",
        key="season",
        why="per-(season, player) box features",
        quiet=True,
    )


# ── an update: read your own last copy, amend it, write it back ───────────────


@iv.step(why="a running log of market lines, appended to once a day")
def append_odds():
    """READ-MODIFY-WRITE, which used to need its own primitive.

    `prior=True` says: this is the previous run's copy. It is recorded for lineage and
    EXCLUDED from the comparison — otherwise the stage would be permanently stale against
    its own last output, one step behind itself, forever.

    The clock is what makes it re-run at all. Without that read nothing could ever move it,
    which is right for a fetch-once archive and wrong for anything polled.
    """
    ran.append("append_odds")
    iv.reads("config/today/", why="append once a day")
    iv.external("some-sportsbook", why="today's closing lines")
    have = iv.reads("raw/odds_log/", why="yesterday's copy", prior=True, optional=True)
    old = (
        pl.read_parquet(have)
        if have
        else pl.DataFrame(schema={"day": pl.Utf8, "line": pl.Float64})
    )
    new = pl.DataFrame({"day": [TODAY.isoformat()], "line": [float(len(old) + 1)]})
    with iv.writes("raw/odds_log/", why="a running log of market lines") as out:
        pl.concat([old, new]).unique(subset="day", keep="last").sort(
            "day"
        ).write_parquet(out)


# ── walk-forward: a cohort fit only on what came before it ───────────────────


def cohorts():
    """`where=` picks FILES, so a cohort physically cannot open a later season.

    The rule is DATA, not a lambda, and that is what makes it exact later: it goes into the
    record and is re-run against the dataset as it stands now. A season backfilled BELOW the
    bound is picked up; one added above it is not.
    """

    def one(season, out):
        ran.append(f"cohort:{season}")
        past = iv.reads(
            "processed/features/",
            why="every season before this cohort",
            where={"season": {"lt": season}},
        )
        pl.read_parquet(past).group_by("player", maintain_order=True).agg(
            pl.col("z").mean()
        ).write_parquet(out)

    return iv.for_each(
        SEASONS[1:],
        one,
        dataset="processed/cohorts/",
        key="season",
        why="a fit per cohort, on prior seasons only",
        quiet=True,
    )


# ── one computation, several outputs ──────────────────────────────────────────


@iv.step(why="one joint fit over every season, and the two tables it produces")
def fit():
    ran.append("fit")
    iv.reads("config/hyperparams/", why="a knob change must refit")
    # No `where=`: every season at once. That is what makes this a JOINT fit rather than a
    # per-season one, and it is visible here rather than buried in the builder.
    bf = pl.read_parquet(iv.reads("processed/features/", why="every season's features"))
    with iv.writes("processed/xpm/", why="one rating per player") as out:
        bf.group_by("player", maintain_order=True).agg(
            pl.col("z").mean()
        ).write_parquet(out)
    with iv.writes("processed/xpm_summary/", why="league level, same fit") as out:
        bf.select(pl.col("z").mean().alias("mean_z")).write_parquet(out)


@iv.step(why="what the app renders")
def dump():
    ran.append("dump")
    xpm = pl.read_parquet(iv.reads("processed/xpm/", why="the fit"))
    odds = pl.read_parquet(iv.reads("raw/odds_log/", why="the market, for context"))
    with iv.writes("dump/site/", why="what the app renders", terminal=True) as out:
        xpm.with_columns(pl.lit(odds.height).alias("days_of_odds")).write_parquet(out)


# ── run it ────────────────────────────────────────────────────────────────────


def run(label):
    print(f"\n=== {label} ===")
    ran.clear()
    config()
    for s, p in zip(SEASONS, [10, 20, 30]):
        fetch(s, p)
    features()
    cohorts()
    append_odds()
    fit()
    dump()
    print(f"    ran: {ran or ['nothing']}")


def tree():
    print("\n--- what is on disk ---")
    for f in sorted(root.rglob("*.parquet")):
        print(f"    {f.relative_to(root)}")


run("first run")
tree()

run("nothing changed")

TODAY = dt.date(2026, 8, 22)
run("a new day: the log appends, and only what reads it follows")

SEASONS.append("2026")
run("a new season lands")

HALF_LIFE = 3.5
run("a hyperparameter changes")

shutil.rmtree(root / "processed/xpm_summary")
run("one of the fit's two outputs is deleted")

tree()


# ── what it looks like when something is wrong ────────────────────────────────
#
# Every one of these used to pass quietly. They are the reason the pipeline can be trusted
# to skip: anything the tool cannot account for stops the run instead of guessing.

print("\n\n=== things that now crash, and what they say ===")


def shows(label, fn):
    try:
        fn()
        print(f"\n  {label}\n      NO ERROR — that is a bug")
    except Exception as e:
        first = " ".join(str(e).split())
        print(f"\n  {label}\n      {type(e).__name__}: {first[:300]}")


shows(
    "a stray file in a dataset directory",
    lambda: (
        (root / "raw/box/notes.txt").write_text("x"),
        iv.reads("raw/box/", why="every season"),
    )[1],
)
(root / "raw/box/notes.txt").unlink()

shows(
    "a lambda selector, which could not be replayed",
    lambda: iv.reads(
        "raw/box/", why="every season", where={"season": lambda s: s < "2026"}
    ),
)

shows(
    "an explicit list naming a season that is not there",
    lambda: iv.reads("raw/box/", why="two seasons", where={"season": ["2024", "1999"]}),
)

shows(
    "a read that selects nothing",
    lambda: iv.reads("raw/nothing_here/", why="not there"),
)


def computed_name():
    target = "processed/out/"

    @iv.step(why="names its output with a variable")
    def build():
        with iv.writes(target, why="unreadable without running the code"):
            pass


shows("a computed dataset name inside a step", computed_name)


def no_source():
    ns = {}
    exec(compile("def build():\n    pass\n", "<string>", "exec"), ns)
    iv.step(why="defined with exec, so it has no source")(ns["build"])


shows("a step whose source cannot be read", no_source)


def corrupt_index():
    (root / "processed/xpm/_index.json").write_text("{not json")
    iv.why_stale("processed/xpm/")


shows("a corrupt index, which used to read as 'never built'", corrupt_index)
