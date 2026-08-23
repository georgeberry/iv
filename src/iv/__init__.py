from __future__ import annotations

from .assets import Asset, Dataset
from .core import Invalidator
from .decl import (Read, after_part, all_of, before_part, between, own_last_copy, parts,
                   same_part)
from .errors import ConfigError, DeclError, IvError, StateError
from .shards import Shard, fingerprint

__version__ = "2.0.0"

__all__ = [
    "Invalidator", "Asset", "Dataset", "Read", "Shard", "fingerprint",
    # the selector vocabulary, also reachable as iv.all_of(...) on an instance
    "all_of", "same_part", "before_part", "after_part", "between", "parts",
    "own_last_copy",
    "IvError", "ConfigError", "DeclError", "StateError",
]
