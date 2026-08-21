# Porting a pipeline to iv

Every label became a file, and the two path-shaped concepts became one.

| was | is |
|---|---|
| `version="model"` + `versions={...}` | `constants("config/model/")` and a read of it |
| a global `data_version` | `constants("config/build/")` and a read of it, by the stages that mean it |
| hyperparameters hashed by hand | `constants("config/hyperparams/", **knobs)` |
| `policy="clock"` | `constants("config/today/", date=...)` and a read of it: "re-run daily" **is** "an input moved" |
| `policy="settled"` | a stage that does not read the clock. Nothing can move it, so it stays current — which is what fetch-once history means |
| `policy="manual"` | nothing. Artifacts are never deleted |
| `updates(path)` | `reads(dataset, prior=True)` + `writes(dataset)`, or shard by the natural key so there is no self-read at all |
| `frame(path, slice=...)` | two datasets, or two shards. Reads never filter rows |
| `collection("raw/box/{season}.parquet")` | `reads("raw/box/")` — every dataset is already many files |
| `upto=` / period bounds | `where={"season": lambda s: s <= t}` — a choice of files |
| `fp_of=` | falls out of the selection: a cohort reading `s <= C-1` keys on exactly those shards |
| `partitions(...)` + `plan`/`commit`/`building` | `for_each(...)` |
| `fp="data"` / `"rows"` / `"bytes"` / `"present"` | one fingerprint, of the rows |
| a `{season}` placeholder in a path | `part={"season": season}` — the partition is a shard name, not a path |

## Two things that change how you think

**A dependency is always whole files.** There is no way to depend on part of one. "Seasons
up to T" is a selection of shards, so a stage that must not see the future never opens it —
past-only is structural rather than enforced by a filter.

**Code and versions stop at the stage that owns them.** They decide whether that stage
re-runs. They are not in any filename, so if the numbers come out the same nothing
downstream moves. The expensive fit does not re-run because a builder two steps up was
reformatted.

## What is not covered

**Non-parquet outputs.** A dataset is parquet shards, because the fingerprint is of rows. A
stage that must emit JSON at a fixed path for something else to consume can do it, but that
file is not tracked.

**A builder's internals.** `code=` hashes the decorated function, not the helpers it calls.
If a change lives in a library module, nothing sees it — so either keep a `config/build/`
constants file the affected stages read, or accept the gap knowingly.
