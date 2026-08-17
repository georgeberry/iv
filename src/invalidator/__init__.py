"""invalidator — re-run a step only when something upstream actually changed.

    from invalidator import Invalidator
    import polars as pl

    iv = Invalidator(data_root="data", data_version="wnba-3.07")

    @iv.step("processed/daily_revenue.parquet",
             why="revenue per day; what the dashboard reads")
    def daily_revenue(out):
        sales = pl.read_parquet(iv.reads("raw/sales.parquet",
                                         why="one row per transaction"))
        sales.group_by("day").agg(pl.col("amount").sum()).write_parquet(out)

    daily_revenue()      # runs, or skips because nothing upstream moved

THE RULE, once:

    id(A) = H( fingerprint(A's data), A's metadata, the ids of A's inputs )
    stale(A) <=> recomputed id(A) != stored id(A)

A root — a file nothing in the pipeline writes — has no inputs, so its id IS its data
fingerprint. Rewrite it with identical rows and nothing downstream stirs; add a row and
the whole chain moves. `data_version` sits in every id, so bumping it rebuilds the world —
which is the one thing no fingerprint of the inputs can see.

Everything else falls out of the same call sites: the DAG (`invalidator graph`), the
structural checks (`invalidator check`), and documentation that cannot drift, because
`why=` is a required argument and there is nowhere else for it to live.

invalidator observes; it does not run anything. Keep your bash, your Makefile, your
scheduler.
"""
from __future__ import annotations

from .core import Invalidator, source_digest
from .errors import (ConfigError, DagioError, DeclError, FingerprintError,
                     PolicyError, StateError)
from .state import POLICIES, Spec

__version__ = "0.1.0"

__all__ = [
    "Invalidator", "Spec", "POLICIES", "source_digest",
    "DagioError", "ConfigError", "DeclError", "FingerprintError", "PolicyError",
    "StateError",
]
