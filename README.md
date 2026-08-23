# iv

Re-run a step only when something upstream actually changed.

`iv` is a small dependency tracker for data pipelines. You declare what each stage
reads and writes; it decides what has to run. The core is stdlib-only.

There is **no index and no state database**. A derived file is named
`<part>.<key>.<fp>`:

- `key` is a digest of the upstreams it was built from,
- `fp` is a digest of its own bytes.

"Is this current?" recomputes the key from the declared upstreams and the files on
disk, then asks whether a file by that name is there. The record cannot go missing,
go stale, or disagree with the tree — it *is* the tree. A file with no key in its
name was not derived here: it is a root, and its identity is its contents.

Both hashes earn their place. The key moves when an upstream moves, which is what
makes a stage re-run. The `fp` is what downstream depends on, which is what gives
you the early cutoff: if a stage re-runs and produces identical bytes, everything
below it stays put.

## Install

Not on PyPI — the name `iv` there belongs to an unrelated package. Install from
this repo:

```bash
uv add "iv @ git+https://github.com/georgeberry/iv"
# or: pip install "git+https://github.com/georgeberry/iv"
```

Extras: `data` (polars — the default fingerprint hashes the *data*, not the bytes),
`cli` (typer), `viz` (networkx + matplotlib), `lint` (pyflakes, for `iv preflight`).

```bash
uv add "iv[data,cli] @ git+https://github.com/georgeberry/iv"
```

## The `Invalidator`

```python
from iv import Invalidator

iv = Invalidator(
    tree="data/",                       # where the DATA lives; datasets are named
                                        # relative to it, so an id survives a move
    out_tree=None,                      # where writes GO, if that is somewhere else
    project=None,                       # where the CODE lives; node names are relative
    code=("src", "scripts"),            # the modules `iv preflight` reads
)
```

## Every dataset is declared

Once, and exactly once — either as something this pipeline builds, or as something that
arrives from outside:

```python
bios = iv.source("raw/bios/", why="heights and weights, dropped in by hand once a year")

@iv.step(output="processed/features/", why="per-season box features", part="season")
def features(box=iv.same_part(box_settled, why="this season's box"),
             bio=iv.all_of(bios, as_paths=True, why="the body-shape columns")):
    ...
```

A read **names that declaration**. There is no category of dataset you refer to by writing
its path a second time, which is what a `sources=("raw/",)` prefix list used to be for.

Three things follow, and none of them is a check any more:

- **A read of something nothing declares** is a `NameError` where it is written, not a
  `READ WITH NO PRODUCER` the next time someone runs `iv check`.
- **A cycle cannot be written.** The first stage would have to name the second before the
  second exists, and Python settles that where it is written.
- **A consumer cannot be defined before its producer**, for the same reason.

And a dataset nothing reads is simply a leaf — `terminal=True` used to be an assertion
every dump had to remember to make so a check would not complain about it. The graph knows
its own consumers.

## Declaring a dataset, and producing one

Two words, because they are two statements.

`@iv.step` is the function that BUILDS. Its upstreams are **parameter defaults**, so the
whole declaration — datasets, selectors, partition key — is readable from the function
object with nothing executed and no source text needed:

```python
@iv.step(output="processed/features/", why="per-season box features", part="season")
def features(box=iv.same_part(box_raw, why="raw box for this season")):
    return box.with_columns((pl.col("pts") * 2).alias("z"))
```

Most stages write one dataset and declare it inline like that, and a read names the stage:
`iv.all_of(features, why="...")`.

`iv.data` DECLARES a dataset — a name, a format, a line on what it is for — without saying
how it is computed. Reach for it when the name has to be said somewhere other than the
`output=` that produces it, which is what a stage writing several tables needs:

```python
XPM = iv.data("processed/xpm/", why="a rating per player per season")
XPM_CAREER = iv.data("processed/xpm_career/", why="one row per player, career to date")

@iv.step(output={"ratings": XPM, "career": XPM_CAREER}, why="the joint fit")
def xpm(bf=iv.all_of(box_features, why="the box prior, every season at once")):
    return {"ratings": ..., "career": ...}

def leaderboard(x=iv.all_of(XPM, why="the headline table")):
```

One expensive fit, two tables, one run. The keys are what the body returns them under;
`XPM` is what everything else calls it. Naming the stage would not say which, so a read
names the declaration. (`xpm["ratings"]` is the same dataset by its key, if it was declared
inline in the dict rather than above.)

Omit `output=` for a stage that writes nothing into the tree — a fetch filling a download
cache, a publish copying out to a bucket. There is no artifact to be stale against, so it
runs every time.

`why=` is required everywhere. There is nowhere else for that line to live, which is what
stops it going stale.

`features.for_each(SEASONS)` builds a shard per season and rebuilds only the ones that
moved. `features("2024")` builds or loads exactly one. Whatever the function returns is
what a later call hands back.

### The selector vocabulary

Each of these is a parameter default, and each carries which shards the stage looks at:

```python
iv.all_of("config/hyperparams/", why="...")        # the whole dataset — a joint fit
iv.same_part("raw/box/", why="...")                # the partition being built
iv.before_part("processed/features/", why="...")   # strictly prior — a walk-forward bound
iv.after_part("processed/features/", why="...")    # strictly later
iv.between("raw/box/", why="...", ge="2020", lt=iv.PART)
iv.parts("raw/box/", why="...", season=["2024", "2025"])
iv.own_last_copy(why="...")                        # the copy this stage overwrites
```

The partition key is never repeated — `same_part` and `before_part` take it from the
stage's own `part=`, and `own_last_copy()` names nothing at all because it can only mean
this stage's own output. Pass `key=` to `between` when the bounds are literal and the stage
is not itself partitioned.

A read names a **dataset path, or the stage that writes it**:

```python
@iv.step(output="config/today/", why="poll once a day", ext=".json")
def today():
    return {"date": dt.date.today().isoformat()}

@iv.step(output="raw/box_live/", why="the season being played", part={"season": LIVE})
def box_live(clock=iv.all_of(today, as_paths=True, why="poll once a day")):
    ...
```

Naming the stage is an ordinary Python reference, so a typo is a `NameError` where it is
written rather than a `READ WITH NO PRODUCER` the next time someone runs `iv check` — and
the path is not spelled twice for a rename to get half of. A dataset nothing here produces
has no stage to name and stays a string, as does one several stages write.

`as_paths=True` hands the body the selected **paths** rather than their contents — what a
stage wants when it passes them to something that opens them itself, concatenates two
datasets before reading, or never looks at the value and declares the read only so the read
can make it stale. It says nothing about how staleness is decided: a key is computed from
filenames either way, and no comparison this package makes ever opens a file.

### Walk-forward, declared

```python
@iv.step(output="processed/cohorts/", why="a fit per cohort, on prior seasons only", part="season")
def cohorts(past=iv.before_part("processed/features/", why="every season before this one")):
    return past.group_by("player").agg(pl.col("z").mean())
```

The bound selects **files**, so a cohort physically cannot open a later season. Because it
is data in the signature rather than a lambda in the body, the shard's key is computable
before the body runs — which is what removes the need to write anything down.

### A root always runs

An asset with no declared upstream — a fetch, a clock, the hyperparameters someone just
edited — runs its body every time:

```python
@iv.step(output="config/hyperparams/", why="the knobs the fit answers to", ext=".json")
def hyperparams():
    return {"half_life": 4.0, "seed": 0}
```

That is deliberate. Nothing on disk can say whether the outside world moved, so `why_stale`
has no question to ask and would answer `current` forever; skipping on that seals the
pipeline shut. Running it is the safe failure — the commit is content-addressed, so a body
producing the same bytes commits the same shard and nothing downstream follows. A fetch too
expensive to repeat opts out with `once=True`.

### What comes back is what you returned

The format has to round trip. A dict written to parquet would come back a `DataFrame`, so
the same function would hand you two different types depending on whether the shard
happened to be current — parquet refuses it and names one that works:

| return | `ext=` |
| --- | --- |
| `pl.DataFrame` | `.parquet` (default) |
| dict / list | `.json` |
| anything | `.pkl` |
| str | `.html`, `.txt` |

For anything else, take an `out` parameter and write the file yourself:

```python
@iv.step(output="dump/page/", why="a rendered page", ext=".html")
def page(out):
    out.write_text(render())
```

## It refuses to guess

Anything `iv` cannot account for stops the run instead of quietly producing a wrong
number. All of these raise:

- **An undeclared read or write inside the data tree.** An undeclared read is absent
  from the recorded inputs, so its source could change forever and nothing would
  rebuild. An undeclared write makes a shard's fingerprint-name a lie. Neither is
  detectable after the fact.
- **A partition-relative read on a stage with no `part=`** — `iv.PART` only means
  something where there is a partition being built.
- **A second producer for one dataset**, unless each names a different literal `part=`.
  Otherwise they race and whichever runs last wins.
- **Building one stage from inside another** — that puts it outside the graph, so nothing
  orders the two and `iv status` cannot see the edge.
- **A parameter `iv` cannot supply** — every one is a read, the partition value, or `out`.
- **A read that selects nothing** (pass `optional=True` if that is legitimate).
- **A stray file** in a dataset directory.
- **A declared write that wrote nothing** (pass `allow_missing=True` if legitimate).

## CLI

Point the CLI at your `Invalidator` instance in `pyproject.toml`:

```toml
[tool.iv]
instance = "mypkg.pipeline:iv"
```

| command | what it does |
| --- | --- |
| `iv status` | `current`, `maybe`, or `stale` per dataset — and which shards |
| `iv plan` | what would rebuild, and what sits downstream of a rebuild |
| `iv why <dataset>` | per-shard: its key, its fp, its upstreams right now, why it is stale |
| `iv graph` | the DAG as text (`--focus <stage>`, `--full`) |
| `iv stage <name>` | one stage's card — what it reads, writes, and why |
| `iv preflight` | undefined names, missing modules, cycles — before a run starts |
| `iv check` | structural problems; `--trace <file>` also diffs against a real run |
| `iv drift` | does the code still agree with the last recorded run? |
| `iv verify` | re-fingerprint every shard, confirm it matches its name |
| `iv gc` | drop superseded shards |
| `iv viz --out dag.png` | draw the DAG — colour is status, shape is kind |

`iv status` has three answers, not two:

```
current  raw/box_settled/     3 shard(s)
stale    raw/box_live/        season=2024: its inputs moved — ...
maybe    derived/features/    4 shard(s), and reads something being rebuilt
```

**`maybe` is the useful one.** A rebuild that produces the same bytes commits the same
shard and stops there, so on an ordinary day the poll re-fetches, writes what it wrote
yesterday, and nothing below it moves. Reporting that tail as stale would be a wall of red
that is wrong by the time you read it.

A stale partitioned dataset names **which** shards, so `3/18 shards (season=2024,
season=2025, season=2026)` tells you which cohorts need refitting rather than only that
some do.

`iv viz` draws the same three answers `iv status` gives, in the same colours — green
current, cyan maybe, amber stale, grey for a source nothing here produces. Shape is a
separate question: square for a dataset that arrives from outside, diamond for one read
outside the pipeline, circle for one built and read here. A dataset several stages write is drawn as one node per shard —
`game_predictions [completed=true]` and `[completed=false]` are different things, and
collapsing them invents a cycle between the stage that reads one and the stage that writes
the other. `--plain` skips reading the tree and leaves every node grey, which is what you
want when the data is somewhere slow.

Set `IV_TRACE=<path>` on a run to record it for `iv drift`.

## A worked example

`example.py` builds one small pipeline exercising every shape `iv` has — a clock and a
config as files, a fetch-once archive, per-season shards, a read-modify-write, one
computation split into many shards, three stages sharing one dataset, a joint fit with four
outputs, two walk-forward bounds, a pickled model object and a terminal JSON dump — then
perturbs it seven ways and prints what re-ran each time:

```bash
uv run python example.py
```

Abbreviated:

```
=== nothing changed ===
    ran: ['nothing']

=== a box score is corrected upstream, but the day has not turned ===
    ran: ['nothing']                      # the fetch polls daily; a code edit is not a file

=== the day turns: the poll picks the correction up, and it flows ===
    ran: ['box:2022', 'box:2023', 'box:2024', 'schedule', 'box_features', 'ncaa', ..., 'site']

=== a knob changes: the fits move, the features do not ===
    ran: ['xpm', 'rapm_fit', 'eoy:2022', 'eoy:2023', 'eoy:2024', ..., 'site']

=== one of the fit's four outputs is deleted ===
    ran: ['xpm']                          # losing one table brings the whole fit back
```

It then prints the graph built from the declarations alone, runs `iv check` over it, and
triggers each of the errors above so you can see what they say.

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
