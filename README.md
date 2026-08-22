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

## Declaring a dataset

`@iv.data` names a dataset and decorates the function that builds it. Upstreams are
**parameter defaults**, so the whole declaration — dataset, selector, partition key —
is readable from the function object with nothing executed and no source text needed:

```python
@iv.data("processed/features/", why="per-season box features", part="season")
def features(box=iv.same_part("raw/box/", why="raw box for this season")):
    return box.with_columns((pl.col("pts") * 2).alias("z"))
```

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
iv.own_last_copy("raw/odds_log/", why="...")       # the copy this stage overwrites
```

The partition key is never repeated — `same_part` and `before_part` take it from the
stage's own `part=`.

### Walk-forward, declared

```python
@iv.data("processed/cohorts/", why="a fit per cohort, on prior seasons only", part="season")
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
@iv.data("config/hyperparams/", why="the knobs the fit answers to", ext=".json")
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
@iv.data("dump/page/", why="a rendered page", ext=".html", terminal=True)
def page(out):
    out.write_text(render())
```

## The `iv.reads` / `iv.writes` style

The original style still works, and `iv graph` is the **union** of what it scans and what
registers itself — so a pipeline migrates one stage at a time rather than all at once.
`example.py` is written in the declared style; a scanned stage sits alongside a declared
one in the same graph.

Here a stage is a decorated function that does its own I/O, and `iv` reads the declarations
off its source:

```python
@iv.step(why="what the app renders")
def dump():
    xpm = pl.read_parquet(iv.reads("processed/xpm/", why="the fit"))
    with iv.writes("dump/site/", why="what the app renders", terminal=True) as out:
        xpm.write_parquet(out)
```

Call `dump()` and it runs only if something it reads has moved. The catch is reach: the
scan needs the source text, so a stage defined in a REPL or a notebook cannot be read this
way, and a selector built from a variable cannot be read at all.

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
