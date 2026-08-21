"""Every way iv refuses.

One class per kind of mistake, because the recovery differs. A `ConfigError` means the
project is not set up; a `DeclError` means a call site is wrong and names file:line; a
`StateError` means the data tree says something impossible.

Nothing here has a fallback. A default in place of an error is how a dataset ends up keyed
on a constant and permanently, silently current.
"""
from __future__ import annotations


class IvError(Exception):
    """Base for everything iv raises."""


class ConfigError(IvError):
    """The project root or the data root could not be resolved."""


class DeclError(IvError):
    """A call site is malformed. Always names the file and line where possible."""


class StateError(IvError):
    """The data tree is not what it claims to be.

    A file in a dataset directory that is not a shard, two shards for one partition, a
    read that selected nothing. Deliberately fatal: each of these would otherwise shorten
    a read silently, and a short read is indistinguishable from thin data after the fact.
    """
