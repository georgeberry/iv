# dagio

**Data lineage and cache invalidation, declared at the read and write call sites.**

```python
import dagio as dg
import polars as pl

games = pl.read_parquet(dg.reads(
    "raw/games.parquet",
    why="one row per team per game; the only source of points"))

with dg.writes("processed/team_stats.parquet",
               why="season points by team; the rating denominator") as p:
    out.write_parquet(p)
```

That is the whole required surface. From those two calls you get:

- **the DAG** — `dagio graph`, `dagio stage`, plus cycle / ordering / orphan checks, read
  out of your source with nothing run and no data on disk
- **cache invalidation** — `dagio status`, `dagio why`, and a `build_if_needed` guard
- **documentation that cannot drift**, because `why=` is a required argument and there is
  nowhere else for it to live

dagio observes; it does not run anything. Keep your bash, your Makefile, your scheduler.

## Why the call site

Because code is the documentation now. A declaration block in a docstring, a manifest of
inputs, a registry of policies — each is a second description of the pipeline, and a second
description drifts. Putting the metadata in the arguments leaves nowhere for it to drift
*to*: delete the call sites and the pipeline stops describing itself, which is the point.

`why=` is required and must be a string literal. So is the path. dagio enforces both by
parsing your source — a computed path raises with the file and line, because a computed
path cannot be read without running the code, which is exactly the rot this prevents.

## The rule

```
id(A) = H( fingerprint(A's data), A's metadata, the ids of A's inputs )

stale(A)  <=>  recomputed id(A) != stored id(A)
```

A **root** — a file nothing in the pipeline writes — has no inputs and no metadata of its
own, so its id is simply its data fingerprint. That is where the recursion bottoms out.
Every derived artifact folds its inputs' ids into its own, so:

- a root that moves moves the whole chain below it
- a root **rewritten with identical data moves nothing** — the fingerprint is of the rows,
  not of the bytes, so a no-op refetch invalidates nothing downstream
- a version bump moves every id that selected that axis, which is how a change in *code*
  invalidates data

Invalidation is one forward pass in topological order, each node decided when it is
reached. A node's new fingerprint is not knowable until it rebuilds, so `dagio plan`
separates what will definitely rebuild from what is downstream of a rebuild and decided on
arrival — rather than pretending to know.

## Setup

```toml
# pyproject.toml
[tool.dagio]
data_root   = "data"                    # a path, or a gs:// / s3:// URI
source_dirs = ["stages", "src"]         # what the AST scan walks
stages      = ["stages/fetch.py", "stages/totals.py"]   # or order_from = "refresh.sh"

[tool.dagio.versions]
data  = "0.1.0"                         # bump for a data/feature pipeline change
model = "0.1.0"                         # bump for a model-form change
```

Version axes are named by you. `versions=("data", "model")` at a write site selects which
of them enter that artifact's id.

## A stage, in full

```python
"""Build the per-(season, player) box-feature matrix."""
import argparse
import polars as pl
import dagio as dg

def build():
    poss = pl.scan_parquet(dg.reads(
        "processed/possessions.parquet",
        why="lineup-level possessions; the per-player minutes denominator"))

    draft_path = dg.reads(
        "processed/draft.parquet",
        why="draft slot for the rookie prior",
        optional=True)                      # absent degrades, does not fail

    with dg.writes("processed/box_features.parquet",
                   why="per-(season, player) box prior for the xPM fit") as p:
        out.write_parquet(p)

ap = argparse.ArgumentParser()
dg.add_guard_args(ap)                       # --if-needed / --force
args = ap.parse_args()
dg.build_if_needed("processed/box_features.parquet", build,
                   if_needed=args.if_needed, force=args.force)
```

A bare invocation always rebuilds; `--if-needed` is opt-in. Forgetting the flag wastes
time, which is recoverable; the opposite default makes a forgotten flag ship stale output.

## For each season, do X

Templates carry the partition, so the literal stays statically readable while the value is
runtime — and the per-partition key falls straight out of the per-partition input:

```python
def build_one(season):
    box = pl.read_parquet(dg.reads(
        "raw/box/{season}.parquet",
        why="raw box scores for one season",
        fp="rows",                          # 220 MiB a season; footer read only
        part={"season": season}))
    return ...

dg.for_each(SEASONS, build_one,
            artifact="processed/box_features.parquet", key="season",
            why="per-(season, player) box prior")
```

```
$ python stages/box_features.py --if-needed
  partitions [processed/box_features.parquet] by season
    reuse   (20): 2006..2025
    rebuild ( 1): 2026
```

Inputs *without* the partition key in their path affect every partition, so they enter
every partition's key. `--force` reaches both the outer guard and the inner cache, because
forcing one while the other reuses everything is a rebuild that rebuilds nothing.

**This is only sound if the loop is causally closed per partition** — every cross-partition
term backward-looking. Test it: build incrementally and from scratch, and compare
**unsorted**.

## Options

At a read:

| | |
|---|---|
| `why=` | **required.** What this input is for |
| `optional=` | absent degrades a feature rather than failing the stage |
| `prior=` | deliberately reads the previous run's copy |
| `fp=` | how to fingerprint it, if it is a root |
| `part=` | the partition values for a `{template}` path |
| `scope=` | which pipeline variant this line belongs to |

At a write, plus `terminal=` (consumed outside the pipeline), `versions=` (which axes enter
the id) and `policy=`:

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

## Commands

```
dagio graph  [--focus X] [--artifacts] [--full]   # run order down, deps as lanes
dagio stage  <name>                               # one stage's I/O, both ends, the whys
dagio check                                       # every structural check; exit 1 on error
dagio drift  [--trace T]                          # what the code says vs what a run did
dagio status                                      # current / stale, with reasons
dagio why    <artifact>                           # the id, its components, and what moved
dagio plan                                        # what would rebuild, and what might
dagio export [--json out.json]                    # {nodes, parent_map} — dbt's shape
dagio viz    [--out dag.png]                      # [viz] extra
```

```
$ dagio graph
╮  ○ fetch
╰╮ ● totals
╮╰ ● ratings
╰─ ● publish
```

Rows are run order, so **an edge going up is a stage reading something written later** —
a bug you can see rather than one you have to query.

The checks: read with no producer · write with no consumer (legal iff `terminal=True`) ·
ordering · two writers · writers disagreeing about what an artifact is · cycles (an
`updates()` self-edge is excluded by construction) · and a guarded fetch, which is this
package's own silent-failure mode — an artifact built from no declared input has nothing in
its id that can move, so guarding its stage means it runs once and never again.

## Tracing

```
DAGIO_TRACE=.dagio/trace.ndjson ./refresh.sh
dagio drift
```

`recorded − declared` is an **error** (the process really did open that file);
`declared − recorded` is a **warning** (an absent optional input, a branch not taken).

## Known limit

dagio only knows about I/O routed through it. A bare `pl.read_parquet(path)` is invisible:
it will not appear in the graph and will not enter any id. Patching the primitives as a
detector — so an untagged read raises rather than passing quietly — is the next thing.

## Install

```
pip install dagio            # core is stdlib only
pip install 'dagio[data]'    # polars, for the default fingerprint
pip install 'dagio[cli]'     # typer
pip install 'dagio[viz]'     # networkx + matplotlib
```

## License

MIT
