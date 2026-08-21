"""iv — re-run a stage only when the data it reads has changed.

    from iv import Pipeline
    import polars as pl

    pipe = Pipeline(root="gs://bucket/data", source_dirs=["scripts"])

    @pipe.step("processed/daily_revenue/", why="revenue per day; what the dashboard reads")
    def daily_revenue(out):
        sales = pl.read_parquet(pipe.reads("raw/sales/", why="one row per transaction"))
        sales.group_by("day").agg(pl.col("amount").sum()).write_parquet(out)

    daily_revenue()      # runs, or skips because nothing upstream moved

EVERYTHING IS A FILE.

A dataset is a DIRECTORY of parquet shards, and a shard is named for its partition and a
fingerprint of its own data:

    processed/box_features/season=2026.7b09d4118ad10e77.parquet

The data identifies the data. That is the whole rule, and the discipline is in what the
name leaves out — no code hash, no version, no digest of what it was built from — because a
dependant does not care how a shard came to exist, only what is in it. Rebuild it from
different inputs and get identical rows, and nothing downstream stirs.

Model versions, hyperparameters and today's date are files too, written by `constants()`.
A stage that answers to one reads it, so what depends on a value is something you can list,
diff and draw — not a label on a call site that nothing can point at.

A staleness check is a directory listing and ZERO file reads.

iv observes; it does not run anything. Keep your bash, your Makefile, your scheduler.
"""
from __future__ import annotations

from .core import Pipeline, source_digest
from .errors import ConfigError, DeclError, IvError, StateError
from .shards import Shard, fingerprint

__version__ = "2.0.0"

__all__ = [
    "Pipeline", "Shard", "fingerprint", "source_digest",
    "IvError", "ConfigError", "DeclError", "StateError",
]
