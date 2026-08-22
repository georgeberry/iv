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

## Declaring a stage

```python
from iv import Pipeline

iv = Pipeline(root="data/")

@iv.step(why="what the app renders")
def dump():
    xpm = pl.read_parquet(iv.reads("processed/xpm/", why="the fit"))
    with iv.writes("dump/site/", why="what the app renders", terminal=True) as out:
        xpm.write_parquet(out)
```

Call `dump()` and it runs only if something it reads has moved. Otherwise it prints
`up to date, skipping` and returns.

`why=` is required everywhere. There is nowhere else for that line to live, which is
what stops it going stale.

### Partitions

`for_each` builds one shard per key, and rebuilds only the keys that moved:

```python
def one(season, out):
    box = pl.read_parquet(
        iv.reads("raw/box/", why="raw box for this season",
                 where={"season": [iv.PART]}))
    box.with_columns((pl.col("pts") * 2).alias("z")).write_parquet(out)

iv.for_each(SEASONS, one, dataset="processed/features/",
            key="season", why="per-season box features")
```

`iv.PART` stands for the partition being built. Because `where=` is **data** rather
than a lambda, the shard's key can be computed *before* its body runs — which is why
nothing has to be written down.

### Walk-forward bounds

`where=` selects files, so a cohort physically cannot open a later season:

```python
past = iv.reads("processed/features/",
                why="every season before this cohort",
                where={"season": {"lt": iv.PART}})
```

A season backfilled *below* the bound is picked up. One added above it is not.

### Metadata is a file

Not a version string, not a label — a shard, declared as an upstream by whoever
answers to it:

```python
iv.constants("config/hyperparams/", why="the knobs the fit depends on",
             half_life=4.0, epochs=300, seed=0)
```

Change `half_life` and exactly the stages that read it re-run. `iv graph` draws the
edge.

### Read-modify-write

```python
have = iv.reads("raw/odds_log/", why="yesterday's copy",
                update_file_on_disk=True, optional=True)
```

`update_file_on_disk=True` means: this is the copy I am about to overwrite. It is
recorded for lineage and excluded from the staleness comparison — otherwise the stage
would be permanently stale against its own last output, one step behind itself,
forever. It may only name a dataset the same stage writes.

## It refuses to guess

Anything `iv` cannot account for stops the run instead of quietly producing a wrong
number. All of these raise:

- **An undeclared read or write inside the data tree.** An undeclared read is absent
  from the recorded inputs, so its source could change forever and nothing would
  rebuild. An undeclared write makes a shard's fingerprint-name a lie. Neither is
  detectable after the fact.
- **A lambda selector**, which could not be replayed.
- **A selector built from a closure variable** the static scan cannot read without
  running the stage.
- **A computed dataset name** inside a step — unreadable without running the code.
- **A read that selects nothing** (pass `optional=True` if that is legitimate).
- **A stray file** in a dataset directory.
- **A declared write that wrote nothing** (pass `allow_missing=True` if legitimate).

## CLI

Point the CLI at your `Pipeline` instance in `pyproject.toml`:

```toml
[tool.iv]
instance = "mypkg.pipeline:iv"
```

| command | what it does |
| --- | --- |
| `iv status` | what each dataset is, and whether it is current |
| `iv plan` | what would rebuild, and what sits downstream of a rebuild |
| `iv why <dataset>` | per-shard: its key, its fp, its upstreams right now, why it is stale |
| `iv graph` | the DAG as text (`--focus <stage>`, `--full`) |
| `iv stage <name>` | one stage's card — what it reads, writes, and why |
| `iv preflight` | undefined names, missing modules, cycles — before a run starts |
| `iv check` | structural problems; `--trace <file>` also diffs against a real run |
| `iv drift` | does the code still agree with the last recorded run? |
| `iv verify` | re-fingerprint every shard, confirm it matches its name |
| `iv gc` | drop superseded shards |
| `iv viz --out dag.png` | draw the DAG |
| `iv export` | the graph as JSON |

Set `IV_TRACE=<path>` on a run to record it for `iv drift`.

## A worked example

`repro.py` builds one small pipeline exercising every shape `iv` has — partitions, a
walk-forward bound, a read-modify-write, a joint fit with two outputs, a terminal
dump — then perturbs it six ways and prints what re-ran each time:

```bash
uv run python repro.py
```

Abbreviated:

```
=== nothing changed ===
    ran: ['nothing']

=== a new day: the log appends, and only what reads it follows ===
    ran: ['append_odds', 'dump']

=== a new season lands ===
    ran: ['features:2026', 'cohort:2026', 'fit', 'dump']

=== a hyperparameter changes ===
    ran: ['fit']              # the fit's bytes did not move, so the dump did not follow

=== one of the fit's two outputs is deleted ===
    ran: ['fit']
```

It ends by triggering each of the errors above so you can see what they say.

## Development

```bash
uv sync --extra dev
uv run pytest
```

## License

MIT
