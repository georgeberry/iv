from __future__ import annotations


class TykeError(Exception):
    pass


class ConfigError(TykeError):
    pass


class DeclError(TykeError):
    pass


class StateError(TykeError):
    pass
