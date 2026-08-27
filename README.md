# tyke

`tyke` is a small, database-free dependency tracker for Python data pipelines. Stages
declare what they read and write; `tyke` runs only the shards whose declared inputs moved.

Each derived filename contains both its derivation key and content fingerprint:

```text
<partition>.<key>.<fingerprint><extension>
```

If a stage reruns but produces identical content, its fingerprint stays unchanged and the
rebuild stops before downstream stages.

## Install

Install `tyke` from PyPI:

```bash
uv add "tyke[data,cli]"
# or
pip install "tyke[data,cli]"
```

Extras are `data` for Polars/Parquet, `cli` for the `tyke` command, `viz` for graph images,
`lint` for preflight source checks, and `dev` for development dependencies.

## Quick start

```python
import polars as pl
from tyke import Pipeline

tyke = Pipeline(tree="data", project=".", code=["pipeline.py"])

@tyke.data(
    dataset="raw/scores/",
    part="season",
    universe=["2024", "2025"],
    why="official season exports",
)
def scores(season):
    return pl.DataFrame({"season": [season], "points": [100]})

@tyke.data(
    dataset="processed/features/",
    part="season",
    universe=["2024", "2025"],
    why="score features by season",
)
def features(
    season,
    scores=tyke.same_part(scores, why="scores for this season"),
):
    return scores.with_columns((pl.col("points") * 2).alias("points_x2"))
```

Configure the CLI in `pyproject.toml`:

```toml
[tool.tyke]
instance = "pipeline:tyke"
```

Then run:

```bash
tyke preflight
tyke run
tyke status
```

Roots have no declared inputs and run whenever called. Add `once=True` for a root that
should only be built once. All commits are content-addressed, so unchanged root output
does not move downstream.

## Declarations

- `tyke.source(...)` declares a dataset supplied outside the pipeline.
- `@tyke.data(...)` declares a stage producing one dataset.
- `tyke.dataset(...)` gives a dataset a reusable name or shared schema.
- `@tyke.step(output={...})` declares a multi-output stage. With no output, it is an action
  and runs whenever called.

Stage parameters declare reads:

| Selector | Shards selected |
| --- | --- |
| `tyke.all_of(data, why=...)` | Every shard |
| `tyke.same_part(data, why=...)` | The current output partition |
| `tyke.before_part(data, why=..., inclusive=False)` | Earlier partitions |
| `tyke.after_part(data, why=..., inclusive=False)` | Later partitions |
| `tyke.between(data, why=..., ge=..., lt=...)` | A bounded range |
| `tyke.parts(data, why=..., season=[...])` | Explicit partition values |
| `tyke.own_last_copy(why=...)` | The stage's previous output for append/update workflows |

Use `optional=True` when no matching shard is valid and `as_paths=True` when the function
needs paths instead of loaded values. `tyke.PART` is available inside range bounds.

Set `part="season"` for one shard per call, a tuple for composite partitions, or a mapping
such as `part={"source": "ncaa"}` for a literal shard. `universe=` tells `tyke run` which
dynamic partitions to enumerate. `split=True` means one call returns all partitions as a
mapping.

Every dynamic stage run through `tyke run` needs its own `universe=`. Direct calls and an
explicit `for_each([...])` already name the shards to build and do not need one.

Use `version=` to invalidate a stage deliberately. Use `schema=` on sources or datasets
to enforce an exact ordered Polars schema. Supported stored values are DataFrames
(`.parquet`), JSON-compatible dictionaries/lists (`.json`), picklable values (`.pkl`), and
strings (`.html` or `.txt`). A stage may accept `out` and write its staged file directly.

## CLI

| Command | Purpose |
| --- | --- |
| `tyke run` | Run the pipeline in dependency order |
| `tyke run --up-to STAGE` | Run a stage and its prerequisites |
| `tyke run --up-to-excluding STAGE` | Run only a stage's prerequisites |
| `tyke run --from STAGE` | Run a stage and descendants; require current upstreams |
| `tyke run --only STAGE` | Run one stage; require current upstreams |
| `tyke run --only STAGE --force` | Run despite stale upstreams; does not rebuild them |
| `tyke run --part season=2025` | Filter partitioned work; repeat for composite keys |
| `tyke run --log run.log` | Save merged stdout, stderr, and outcomes incrementally |
| `tyke determinism --only STAGE` | Run a stage twice in isolated temporary output trees and compare content |
| `tyke determinism --only STAGE --part season=2025` | Audit one partition of a stage |
| `tyke determinism --sample` | Audit every stage at its last declared partition |
| `tyke status` | Show current, maybe, and stale shards |
| `tyke plan` | Show rebuilds and conditional downstream work |
| `tyke why DATASET` | Show shard fingerprints, keys, inputs, and status |
| `tyke graph` | Print the DAG; supports `--focus` and `--full` |
| `tyke stage NAME` | Show a stage's reads and writes |
| `tyke impact STAGE --tick` | Show possible impact if all output shards change |
| `tyke impact STAGE --tick --tick-part season=2025` | Tick one existing output partition |
| `tyke preflight` | Check undefined names, missing modules, and cycles |
| `tyke check [--trace FILE]` | Validate declarations and optionally compare a trace |
| `tyke drift [--trace FILE]` | Compare code with a recorded run |
| `tyke verify [DATASET]` | Re-fingerprint shards and verify their filenames |
| `tyke gc [DATASET]` | Remove superseded shards |
| `tyke viz --out dag.png` | Render the DAG; requires the `viz` extra |

`maybe` means an upstream may change: downstream work runs only if the rebuilt content
gets a new fingerprint. Set `TYKE_TRACE=path` during a pipeline run to record a trace.
During `tyke run`, each rebuild reports its cause. If an upstream's content changed earlier
in that run, the exact dataset shard is named; older aggregate keys can only identify that
declared inputs, the version, or the schema changed.

`tyke determinism --only STAGE` forces the named output-producing stage twice, each time
into a fresh temporary output tree, and compares its output partition set and content
fingerprints. It never writes production outputs. It rejects actions because they have no
artifact to compare; use `--part key=value` to audit a particular dynamic shard.
`tyke determinism --sample` visits every output-producing stage and chooses the last
partition in TYKE's normal partition order, so its representative selection is repeatable.

## Safety

Within an active stage, reads and writes under the data tree must go through declared
`tyke` inputs and outputs. `tyke` rejects undeclared I/O, conflicting writers, invalid
partition selectors, missing required shards, schema mismatches, and nested stage calls.

## Development

```bash
uv sync --extra dev
uv run pytest
uv run python example.py
```

MIT licensed.
