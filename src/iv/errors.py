"""Every way iv refuses.

One class per kind of mistake, because the recovery differs. A `ConfigError` means the
project is not set up; a `DeclError` means a call site is wrong and names file:line; a
`FingerprintError` means a strategy cannot be applied to that file and tells you which one
to pick instead.

Nothing here has a fallback. A default in place of an error is how an artifact ends up
keyed on a constant and permanently, silently current.
"""
from __future__ import annotations


class InvalidatorError(Exception):
    """Base for everything iv raises."""


class ConfigError(InvalidatorError):
    """The project root, data root, or version axes could not be resolved."""


class DeclError(InvalidatorError):
    """A call site is malformed. Always names the file and line where possible."""


class FingerprintError(InvalidatorError):
    """A fingerprint strategy cannot be applied to this file."""


class PolicyError(InvalidatorError):
    """Two writers of one artifact disagree about how it should be treated."""


class StateError(InvalidatorError):
    """A stamp is unreadable or malformed.

    Deliberately fatal rather than an empty dict: an empty state makes invalidation a
    no-op and every builder rebuild forever. A MISSING record directory is the only legal
    empty state.
    """
