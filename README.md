# iv

**Re-run a stage only when the data it reads has changed.**

```python
# pipeline.py — once per project
from iv import Pipeline

pipe = Pipeline(root="gs://bucket/data", source_dirs=["stages"])
```

```python
# stages/daily_revenue.py — read, transform, write
import polars as pl
from pipeline import pipe


@pipe.step("processed/daily_revenue/",
           why="revenue per day; what the dashboard reads")
def daily_revenue(out):
    sales = pl.read_parquet(pipe.reads(
        "raw/sales/", why="one row per transaction; the only source of revenue"))

    (sales.group_by("day", maintain_order=True)
          .agg(pl.col("amount").sum().alias("revenue"))
          .write_parquet(out))


daily_revenue()
```

```
$ python stages/daily_revenue.py
  processed/daily_revenue/: not on disk (the only shard)

$ python stages/daily_revenue.py
  processed/daily_revenue/ is current — skipping

$ # ...rewrite raw/sales with the SAME rows but different bytes
$ python stages/daily_revenue.py
  processed/daily_revenue/ is current — skipping

$ # ...add a transaction
$ python stages/daily_revenue.py
  processed/daily_revenue/: input moved: raw/sales/
```

**The decorator is the output. The `pipe.reads(...)` calls in the body are the inputs.**
Nothing else to declare.

From those same call sites you also get **the DAG** (`iv graph`, `check`, `stage`) — read
straight out of your source, so it works on a fresh checkout with no data and nothing ever
run — and **documentation that cannot drift**, because `why=` is a required argument and
there is nowhere else for it to live.

iv observes; it does not run anything. Keep your bash, your Makefile, your scheduler.

## Everything is a file

A dataset is a **directory** of parquet shards. A shard is named for its partition and a
fingerprint of its own data:

```
processed/box_features/
    season=2025.a3f21c8e4f10bb92.parquet
    season=2026.7b09d4118ad10e77.parquet
    _index.json
```

**The data identifies the data.** That is the whole rule, and the discipline is in what the
name leaves out — no code hash, no version, no digest of what it was built from — because a
dependant does not care how a shard came to exist, only what is in it.

- A shard rewritten with identical rows **moves nothing**: same data, same fingerprint, same
  filename. Never mtime, never bytes.
- A rebuild that produces identical output **stops there**. Editing a builder re-runs that
  builder; if the numbers do not change, the 287-second fit downstream does not re-run.
- A staleness check is **one directory listing and zero file reads**.

Model versions, hyperparameters and today's date are files too:

```python
pipe.constants("config/hyperparams/", why="the knobs the fit shape depends on",
               half_life=4.0, epochs=300, seed=0)
```

A stage that answers to them reads them, and the ordinary machinery does the rest. What
depends on a value is therefore something you can list, diff and draw — `iv graph` shows the
edge — rather than a label on a call site that nothing can point at.

That means **there is no sledgehammer**: a version rebuilds exactly the stages that declared
they read it. More honest, and more work to wield. It is the trade this makes on purpose.

## Two questions, kept apart

| | |
|---|---|
| what a **dependant** sees | the fingerprints of the shards it read. Nothing else. |
| whether a **stage** re-runs | have those fingerprints moved, or has its own source changed |

The first is in the filename. The second needs to compare against what the last build
actually saw, and that is what `_index.json` holds: the input datasets, the partitions taken
from each, and their ids.

The index is load-bearing, and it is worth being exact about how: **losing it causes a
rebuild, never a false skip.** No record of what a shard was built from means the inputs
cannot be compared, so the shard cannot be shown current, so it is rebuilt. Corrupt, raced,
deleted — every failure lands on the safe side.

## For each season, do X

```python
def build_one(season, out):
    box = pl.read_parquet(pipe.reads(
        "raw/box/", why="raw box scores", where={"season": [season]}))
    ...

pipe.for_each(SEASONS, build_one, dataset="processed/box_features/",
              key="season", why="per-(season, player) box prior")
```

```
  partitions [processed/box_features/] by season
    reuse   (20): 2006..2025
    rebuild ( 1): 2026
```

**`where=` picks FILES, never rows.** So a walk-forward stage that must not see the future
simply never opens it:

```python
pipe.reads("processed/possessions/", why="seasons strictly before this cohort",
           where={"season": lambda s: s < cohort})
```

Past-only stops being enforced by a filter two layers down and becomes structural — visible
at the call site, and impossible to get wrong. An explicit list (`["2019", "2020"]`) is a
**coverage claim**: a missing value raises rather than silently returning a shorter read.

## The primitives

| | |
|---|---|
| `constants(dataset, *, why, **values)` | values from outside the data, as a shard |
| `reads(dataset, *, why, where=, optional=, prior=)` | inputs; returns sorted shard paths |
| `writes(dataset, *, why, part=, terminal=, allow_missing=)` | one shard, committed on clean exit |
| `step(dataset, *, why, part=, code=, terminal=, allow_missing=)` | the guard |
| `for_each(over, build_one, *, dataset, key, why)` | one shard per partition |
| `external(name, *, why)` | provenance for a source outside the pipeline |

`prior=True` reads what is on disk now and is excluded from the comparison — that is how a
stage amends its own output without being permanently stale against itself.

`code=` is the one thing that is not a file, and it earns it: a property of the *function*,
not of the data, and with no version to bump it is what catches a builder edit. The hash is
over the parsed tree, so reformatting and comments are free. It is **shallow** — this
function, not the helpers it calls.

## Guarantees worth knowing

**Reads come back sorted, by parsed partition value.** `game_id=9` before `game_id=10`, with
nobody remembering to zero-pad. Row order is an input to anything that slices or sums
floats, so this is a guarantee rather than a convenience.

**Anything unrecognised in a dataset directory is a hard error.** Skipping it would silently
drop a partition and shorten every read downstream, which is indistinguishable from thin
data after the fact. A shard is staged on local disk and moved in whole, so there is no
in-flight file to make an exception for.

**Staging is local even for a bucket.** Fingerprinting reads the rows, and reading a file
that was just uploaded pays for the data twice. Write local, hash local, upload once,
straight to the final name.

**Two shards for one partition raise.** That means an interrupted commit, and picking the
newer by mtime would be a silent, unreproducible answer. `iv gc` is the fix.

## Commands

```
iv graph  [--focus X] [--full]   # run order down, dependencies as lanes
iv stage  <name>                 # one stage's I/O, both ends, the whys
iv check  [--trace T]            # every structural check; exit 1 on error
iv drift  [--trace T]            # what the code says vs what a run did
iv status                        # current / stale, with reasons
iv why    <dataset>              # every shard, and what it was built from
iv plan                          # what would rebuild, and what might
iv export [--out m.json]         # {nodes, parent_map, datasets}
iv gc     [dataset]              # drop what an interrupted commit left behind
iv viz    [--out dag.png]        # [viz] extra
```

```
$ iv graph
╮  ○ config
╰╮ ● fetch
╮╰ ● features
╰╮ ● fit
 ╰ ● dump
```

Rows are run order, so **an edge going up is a stage reading something written later** — a
bug you can see rather than one you have to query.

The checks: read with no producer · write with no consumer (legal iff `terminal=True`) ·
**one writer per dataset**, no exceptions · ordering · cycles · and **runs once**, which is
this package's own silent-failure mode — a stage that reads no dataset has nothing that can
make it stale, so it runs once and never again. Right for a fetch-once archive; for anything
polled, the fix is to read a clock file.

## Tracing

```
IV_TRACE=.iv/trace.ndjson ./refresh.sh
iv drift
```

`recorded − declared` is an **error** (the process really did open that dataset);
`declared − recorded` is a **warning** (an absent optional input, a branch not taken).

The scan is **information, not authority**. It answers what is declared. It does not decide
staleness — "which reads will execute" is a claim about execution, not a structural fact,
and treating an approximation as authoritative means any blind spot becomes a dataset that
rebuilds forever with correct output and no error. What governs is what the last build
actually read.

## Known limits

**iv only knows about I/O routed through it.** A bare `pl.read_parquet(path)` is invisible:
it will not appear in the graph and will not enter any id. Patching the primitives as a
*detector* — so an untagged read raises rather than passing quietly — is the next thing.

**Non-parquet outputs are outside the model.** A dataset is parquet shards, because the
fingerprint is of rows. A stage that must emit JSON at a fixed path for something else to
consume can do it, but iv cannot track that file.

**`code=` is shallow.** It sees the decorated function, not the helpers it calls. Measured
on a real repo, widening it to the call closure was worse in both directions at once: 84
functions across 18 files for one stage, an edit to an import-time monkey-patch invalidating
24 of 28 datasets, and the 1,350-line model module unreachable because it is re-exported
through a computed binding.

## Install

```
pip install iv            # core is stdlib only
pip install 'iv[data]'    # polars, for fingerprints
pip install 'iv[cli]'     # typer
pip install 'iv[viz]'     # networkx + matplotlib
```

## License

MIT
