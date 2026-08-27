from __future__ import annotations


class IvError(Exception):
    pass


class ConfigError(IvError):
    pass


class DeclError(IvError):
    pass


class StateError(IvError):
    pass
