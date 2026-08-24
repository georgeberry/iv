# iv

Re-run a pipeline stage only when something upstream actually changed.

`iv` is a small dependency tracker for data pipelines. Declare every dataset a stage
reads and writes; `iv` decides whether a shard must run. The core has no database and
uses only the Python standard library.

Derived files are their own records. Each is named:

```
<part>.<key>.<fingerprint><ext>
```

- `key` identifies the upstream shards used to build it.
- `fingerprint` identifies the resulting contents.

To decide whether a shard is current, `iv` recomputes its key from the declarations and
the files on disk, then looks for that name. If a stage reruns but produces identical
contents, its fingerprint does not change, so downstream work stops there.

## Install

`iv` is not published to PyPI—the name there belongs to a different package. Install it
from this repository:

```bash
uv add "iv @ git+https://github.com/georgeberry/iv"
# or: pip install "iv @ git+https://github.com/georgeberry/iv"
```

Install extras as needed:

```bash
uv add "iv[data,cli] @ git+https://github.com/georgeberry/iv"
```

| Extra | Provides |
| --- | --- |
| `data` | Polars and the default `.parquet` data fingerprinting |
| `cli` | The `iv` command-line interface |
| `viz` | `iv viz` (`networkx` and `matplotlib`) |
| `lint` | Undefined-name checks for `iv preflight` (`pyflakes`) |

## Quick start

Point an `Invalidator` at the directory holding datasets. Dataset names are relative to
`tree`, so they remain stable when the data directory moves.

```python
import polars as pl
from iv import Invalidator

iv = Invalidator(tree="data", project=".")

@iv.data(dataset="raw/scores/", why="official season score exports", part="season")
def scores(season):
    return pl.DataFrame({"season": [season], "points": [100]})

@iv.data(dataset="processed/features/", why="per-season score features", part="season")
def features(
    season,
    score=iv.same_part(scores, why="scores for this season"),
):
    return score.with_columns((pl.col("points") * 2).alias("points_x2"))

# Build the roots, then build only the feature partitions that need work.
scores.for_each(["2023", "2024", "2025"])
features.for_each(["2023", "2024", "2025"])
```

`iv` reads declared inputs before it calls the function. Parameter defaults are therefore
the stage declaration: they say what is read, which shards are selected, and why. A call
to a stage loads its current output or builds it if required.

## Datasets and stages

Every dataset is declared exactly once. There are three forms:

- `iv.source(...)` declares data that arrives from outside the pipeline.
- `@iv.data(...)` builds one dataset; the function returns its contents.
- `iv.dataset(...)` declares a dataset separately when it needs a name before, or apart
  from, its producer. `@iv.step(...)` then produces several declared datasets at once.

Most stages produce one output, so `@iv.data` is the usual form:

```python
@iv.data(dataset="processed/features/", why="per-season box features", part="season")
def features(
    season,
    box=iv.same_part(box_raw, why="raw box scores for this season"),
):
    return box.with_columns((pl.col("pts") * 2).alias("z"))

# A downstream read names the stage, not a second copy of its path.
@iv.data(dataset="dump/leaderboard/", why="the published leaderboard", ext=".json")
def leaderboard(features=iv.all_of(features, why="all feature shards")):
    return {"rows": features.height}
```

Use `iv.dataset` plus `@iv.step` when one expensive computation produces multiple
datasets. The mapping keys are the keys returned by the function; the declared datasets
are the names downstream stages use.

```python
RATINGS = iv.dataset("processed/ratings/", why="a rating per player")
SUMMARY = iv.dataset("processed/summary/", why="fit diagnostics")

@iv.step(output={"ratings": RATINGS, "summary": SUMMARY}, why="the joint fit")
def fit(features=iv.all_of(features, why="the feature matrix")):
    return {"ratings": make_ratings(features), "summary": make_summary(features)}

@iv.data(dataset="dump/report/", why="the report payload", ext=".json")
def report(ratings=iv.all_of(RATINGS, why="the published ratings")):
    return {"players": ratings.height}
```

`@iv.step` can also omit `output=` for an action that writes outside the data tree, such
as publishing to a bucket. With no artifact to compare, that stage runs on every call.

### Roots, polling, and `once=True`

A producing stage with no declared reads is a root. It runs whenever called: only its body
can know whether an API, a clock, or an edited configuration changed. This is safe because
identical output keeps the same fingerprint and does not invalidate downstream stages.

For a root that should only be fetched once, pass `once=True`. For example, a historical
archive can be fetched shard-by-shard once, while a live feed can read a daily clock and
be polled safely.

### Partitions and selectors

Set `part="season"` when a stage builds one partition per call; the function receives the
partition value as its `season` parameter. Set a literal partition with
`part={"source": "ncaa"}` when a stage owns exactly that shard.

Reads are parameter defaults and name a declared dataset or stage:

```python
@iv.data(dataset="processed/cohorts/", why="a fit per cohort", part="season")
def cohorts(
    season,
    past=iv.before_part(features, why="features from prior seasons"),
):
    return fit_cohort(past)
```

| Selector | Selects |
| --- | --- |
| `iv.all_of(data, why="...")` | Every shard in a dataset |
| `iv.same_part(data, why="...")` | The shard matching this stage's partition |
| `iv.before_part(data, why="...", inclusive=False)` | Earlier partitions; `inclusive=True` includes this one |
| `iv.after_part(data, why="...", inclusive=False)` | Later partitions; `inclusive=True` includes this one |
| `iv.between(data, why="...", ge="2020", lt=iv.PART)` | A bounded partition range |
| `iv.parts(data, why="...", season=["2024", "2025"])` | An explicit set of partitions |
| `iv.own_last_copy(why="...")` | The output being read, modified, and overwritten |

Partition-relative selectors take their key from the stage's `part=` declaration; use
`key="season"` with `between` when a non-partitioned stage uses literal bounds.
`optional=True` permits a selector to match no shard. `as_paths=True` gives the function
the selected file paths instead of loaded values.

### Parquet schema contracts

An important sharded dataset can declare its exact Polars schema in Python:

```python
FEATURES = iv.dataset(
    "processed/features/",
    why="model feature matrix",
    schema={"season": pl.String, "player_id": pl.Int64, "z": pl.Int64},
)
```

`schema=` is also available on `iv.source(...)` and `@iv.data(...)`. Every selected shard
must have the declared ordered columns and types; every `iv.writes(...)` commit is checked
before it reaches the data tree. The contract's digest is part of the derivation key, so a
schema change makes existing shards stale without changing the filename format.

Rebuild a partitioned schema migration with `for_each(...)`. New-schema shards commit one
at a time, but a read spanning an old shard fails until every selected shard conforms. Raw
datasets without `schema=` retain the permissive union read and `iv verify` drift report.

### Formats and manual writes

The return value must round-trip: a cached call should receive the same type a fresh call
returned.

| Return value | `ext=` |
| --- | --- |
| `polars.DataFrame` | `.parquet` (default) |
| `dict` or `list` | `.json` |
| Any picklable value | `.pkl` |
| `str` | `.html` or `.txt` |

For another format, accept an `out` parameter and write the staged file yourself:

```python
@iv.data(dataset="dump/page/", why="a rendered page", ext=".html")
def page(out):
    out.write_text(render_page())
```

## Safety guarantees

`iv` prefers an error to a silently incomplete dependency graph. It rejects undeclared I/O
within the data tree through normal stdlib operations (`open`, `pathlib`, `os`, and
`shutil`) and optional dataframe readers, undeclared function parameters, a second producer for the same shard,
partition-relative reads from an unpartitioned stage, non-optional reads that select
nothing, and stray files in dataset directories. A stage also cannot invoke another stage
from inside its body; express that relationship as a declared read instead.

Each read requires a short `why=` explanation of at most 280 characters. It is shown in
diagnostics and lives next to the dependency it describes.

## CLI

Install the `cli` extra and point the command at your instance:

```toml
[tool.iv]
instance = "mypkg.pipeline:iv"
```

| Command | What it does |
| --- | --- |
| `iv status` | Shows `current`, `maybe`, or `stale` datasets and shards |
| `iv plan` | Shows work that would rebuild and work that may follow |
| `iv why <dataset>` | Explains each shard's key, fingerprint, inputs, and status |
| `iv graph` | Prints the DAG (`--focus <stage>`, `--full`) |
| `iv stage <name>` | Shows one stage's reads, writes, and explanations |
| `iv preflight` | Checks undefined names, missing modules, and cycles |
| `iv check` | Checks the declared graph; `--trace <file>` also compares a run trace |
| `iv drift` | Compares code with the most recent recorded run |
| `iv verify [dataset]` | Re-fingerprints shards and checks their filenames |
| `iv gc [dataset]` | Removes superseded shards after an interrupted commit |
| `iv viz --out dag.png` | Draws the DAG (requires the `viz` extra) |

`maybe` is deliberate: an upstream shard will rebuild, but if it produces identical
contents, the downstream shard remains current. Set `IV_TRACE=<path>` while running a
pipeline to enable `iv drift`.

## Example and development

[`example.py`](example.py) exercises sources, roots, partitions, polling, shared datasets,
multi-output stages, walk-forward selectors, read-modify-write, and CLI checks:

```bash
uv run python example.py
```

For development:

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
