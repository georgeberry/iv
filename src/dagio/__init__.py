"""dagio — data lineage and cache invalidation declared at the read and write call sites.

Two functions carry the design. `reads` resolves an input and remembers the edge; `writes`
resolves an output and, on clean exit, folds its fingerprint, its metadata, and the ids of
everything read into one id and stamps it.

    import dagio as dg

    poss = pl.scan_parquet(dg.reads("processed/possessions.parquet",
                                    why="lineup possessions; the minutes denominator"))

    with dg.writes("processed/box_features.parquet",
                   why="per-(season, player) box prior for the xPM fit") as p:
        out.write_parquet(p)

That is the entire required surface. Everything else — the graph, the cycle and ordering
checks, the staleness answer, the per-partition reuse — is derived from those call sites,
either by reading the code (`dagio graph`, `dagio check`) or by reading the state file
(`dagio status`, `dagio why`).

THE RULE, once:

    id(A) = H( fingerprint(A's data), A's metadata, the ids of A's inputs )
    stale(A) <=> recomputed id(A) != stored id(A)

dagio observes; it does not run anything. Keep your bash, your Makefile, your scheduler.
"""
from __future__ import annotations

# `get_config`, not `config` — a name that shadows the `dagio.config` submodule makes
# `from dagio import config` mean two different things depending on import order.
from .config import Config, configure, get as get_config
from .errors import (ConfigError, DagioError, DeclError, FingerprintError,
                     PolicyError, StateError)
from .guard import add_guard_args, build_if_needed, current, forced, why_stale
from .io import declared_reads, external, reads, stamp_content, updates, writes
from .partition import PartitionCache, for_each
from .state import POLICIES, Spec, compute_id, is_current, record_of

__version__ = "0.1.0"

__all__ = [
    "reads", "writes", "updates", "external", "stamp_content", "declared_reads",
    "for_each", "PartitionCache",
    "add_guard_args", "build_if_needed", "current", "forced", "why_stale",
    "is_current",
    "record_of", "configure", "get_config", "Config", "Spec", "POLICIES", "compute_id",
    "DagioError", "ConfigError", "DeclError", "FingerprintError", "PolicyError",
    "StateError",
]
