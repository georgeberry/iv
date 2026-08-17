# invalidator

**Re-run a step only when something upstream actually changed.**

```python
# pipeline.py — once per project
from invalidator import Invalidator

iv = Invalidator(data_root="data", data_version="sales-1.0", source_dirs=["stages"])
```

```python
# stages/daily_revenue.py — read, transform, write
import polars as pl
from pipeline import iv


@iv.step("processed/daily_revenue.parquet",
         why="revenue per day; what the dashboard reads")
def daily_revenue(out):
    sales = pl.read_parquet(iv.reads(
        "raw/sales.parquet", why="one row per transaction; the only source of revenue"))

    (sales.group_by("day", maintain_order=True)
          .agg(pl.col("amount").sum().alias("revenue"))
          .write_parquet(out))


daily_revenue()
```

```
$ python stages/daily_revenue.py
  processed/daily_revenue.parquet: not on disk

$ python stages/daily_revenue.py
  processed/daily_revenue.parquet is current — skipping

$ # ...rewrite raw/sales.parquet with the SAME rows but different bytes
$ python stages/daily_revenue.py
  processed/daily_revenue.parquet is current — skipping

$ # ...add a transaction
$ python stages/daily_revenue.py
  processed/daily_revenue.parquet: input moved: raw/sales.parquet  5c834461… -> 1d87d86f…
```

**The decorator is the output. The `iv.reads(...)` calls in the body are the inputs.**
Nothing else to declare.

From those same call sites you also get **the DAG** (`invalidator graph`, `check`, `stage`)
— read straight out of your source, so it works on a fresh checkout with no data and
nothing ever run — and **documentation that cannot drift**, because `why=` is a required
argument and there is nowhere else for it to live.

invalidator observes; it does not run anything. Keep your bash, your Makefile, your
scheduler.

## The rule

```
id(A) = H( fingerprint(A's data), A's metadata, the ids of A's inputs )

stale(A)  <=>  recomputed id(A) != stored id(A)
```

A **root** — a file nothing in the pipeline writes — has no inputs, so its id is simply its
data fingerprint. That is where the recursion bottoms out. Every derived artifact folds its
inputs' ids into its own, so:

- a root that moves moves the whole chain below it
- a root **rewritten with identical data moves nothing** — the fingerprint is of the rows,
  not of the bytes, so a no-op refetch invalidates nothing downstream. Never mtime.
- `data_version` sits in every id, so bumping it rebuilds everything. That is what covers
  the one thing no fingerprint of the inputs can see: a builder whose logic changed.

A staleness check is **O(number of root files)**, not O(data) and not O(depth). A derived
input's id is a dict lookup in the state file; only roots are fingerprinted.

## Configuration is constructor arguments

No TOML discovered by walking up from the working directory, no environment variables, no
global singleton. Two pipelines are two `Invalidator`s; a test is one pointed at a temp
directory.

```python
iv = Invalidator(
    data_root="gs://bucket/data",     # a path, or a URI (needs cloudpathlib)
    data_version="wnba-3.07",         # in EVERY id
    source_dirs=["scripts", "src"],   # what the static scan walks
    stages=[...],                     # or order_from="refresh.sh", for the ORDER check
    trace=".invalidator/trace.ndjson",
)
```

The CLI needs one line to find it — the only discovery in the package:

```toml
[tool.invalidator]
instance = "mypkg.pipeline:iv"
```

## Why a decorator, and why inputs are not in it

A context manager **cannot decline to run its body** — [PEP 377](https://peps.python.org/pep-0377/)
proposed exactly that and was rejected. A callable is the only thing you can choose not to
call, which is why every system in this category (Dagster's `@asset`, Prefect's `@task`,
redun's `@task`, R's `targets`) wraps the unit of work in a function.

Inputs stay in the *body* rather than the decorator because invalidator never owns your
storage — an input in the decorator could only hand you a path anyway, and then there would
be two places to look. Keeping them in the body means an optional input, a branch, a loop
over seasons, and a path that depends on a partition all work with no special syntax, and
the AST scan still sees every one of them because the path is a literal.

The body is also a natural scope, which is what makes the input set **per artifact** rather
than per file.

## Options

At a read:

| | |
|---|---|
| `why=` | **required.** What this input is for |
| `optional=` | absent degrades a feature rather than failing the stage |
| `prior=` | deliberately reads the previous run's copy |
| `fp=` | how to fingerprint it, if it is a root |
| `part=` | the partition values for a `{template}` path |

At a step or a write, plus `terminal=` (consumed outside the pipeline), `code=` (fold the
function's source into the id) and `policy=`:

| `policy=` | |
|---|---|
| `tracked` | the default |
| `manual` | never auto-deleted; a moved id is reported, not acted on |
| `settled` | fetch-once history — the question is coverage, not staleness |
| `exempt` | the input term is dropped from the id |
| `clock` | today's date joins the metadata, so the id turns over daily |

Fingerprint strategies: `data` (default, order-insensitive row hash), `data_order`, `rows`
(parquet footer), `bytes`, `present`, or your own callable. **A coarse strategy on a
derived artifact is a correctness hazard, not just imprecision** — if the id does not move,
everything downstream wrongly skips.

`code=True` folds a hash of the step function's own source into its id, so editing the
transform rebuilds it with no version bump. The hash is over the *parsed* tree, so
reformatting and comments are free. It is off by default because it is **shallow**: it sees
that function, not the helpers it calls. `data_version` is the honest blunt instrument.

## For each season, do X

Templates carry the partition, so the literal stays statically readable while the value is
runtime — and the per-partition key falls straight out of the per-partition input:

```python
def build_one(season):
    box = pl.read_parquet(iv.reads(
        "raw/box/{season}.parquet",
        why="raw box scores for one season",
        fp="rows",                          # 220 MiB a season; footer read only
        part={"season": season}))
    return ...

iv.for_each(SEASONS, build_one,
            output="processed/box_features.parquet", key="season",
            why="per-(season, player) box prior")
```

```
  partitions [processed/box_features.parquet] by season
    reuse   (20): 2006..2025
    rebuild ( 1): 2026
```

Inputs *without* the partition key in their path affect every partition, so they enter
every partition's key. `INVALIDATOR_FORCE=1` reaches both the outer guard and the inner
cache, because forcing one while the other reuses everything is a rebuild that rebuilds
nothing.

**This is only sound if the loop is causally closed per partition** — every cross-partition
term backward-looking. Test it: build incrementally and from scratch, and compare
**unsorted**.

## Commands

```
invalidator graph  [--focus X] [--artifacts] [--full]   # run order down, deps as lanes
invalidator stage  <name>                               # one stage's I/O, both ends, the whys
invalidator check                                       # every structural check; exit 1 on error
invalidator drift  [--trace T]                          # what the code says vs what a run did
invalidator status                                      # current / stale, with reasons
invalidator why    <artifact>                           # the id, its components, what moved
invalidator plan                                        # what would rebuild, and what might
invalidator export [--out m.json]                       # {nodes, parent_map} — dbt's shape
invalidator viz    [--out dag.png]                      # [viz] extra
```

```
$ invalidator graph
╮  ○ fetch
╰╮ ● totals
╮╰ ● ratings
╰─ ● publish
```

Rows are run order, so **an edge going up is a stage reading something written later** — a
bug you can see rather than one you have to query.

The checks: read with no producer · write with no consumer (legal iff `terminal=True`) ·
ordering · two writers · writers disagreeing about what an artifact is · cycles (an
`updates()` self-edge is excluded by construction) · and a **guarded fetch**, which is this
package's own silent-failure mode — an artifact built from no declared input has nothing in
its id that can move, so guarding its stage means it runs once and never again.

## Tracing

```
INVALIDATOR_TRACE=.invalidator/trace.ndjson ./refresh.sh
invalidator drift
```

`recorded − declared` is an **error** (the process really did open that file);
`declared − recorded` is a **warning** (an absent optional input, a branch not taken).

## Known limit

invalidator only knows about I/O routed through it. A bare `pl.read_parquet(path)` is
invisible: it will not appear in the graph and will not enter any id. Patching the
primitives as a *detector* — so an untagged read raises rather than passing quietly — is
the next thing.

## Install

```
pip install invalidator            # core is stdlib only
pip install 'invalidator[data]'    # polars, for the default fingerprint
pip install 'invalidator[cli]'     # typer
pip install 'invalidator[viz]'     # networkx + matplotlib
```

## License

MIT
