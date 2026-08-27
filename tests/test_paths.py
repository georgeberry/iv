from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from tyke import paths
from tyke.errors import ConfigError


def test_mkpath_preserves_path_objects_and_resolves_local_strings(tmp_path):
    existing = Path("already")
    assert paths.mkpath(existing, tmp_path) is existing
    assert paths.mkpath("relative/data", tmp_path) == tmp_path / "relative/data"
    assert paths.mkpath(str(tmp_path / "absolute"), tmp_path) == tmp_path / "absolute"
    assert paths.mkpath("relative/data", None) == Path("relative/data")


def test_mkpath_explains_the_optional_cloud_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_cloudpathlib(name, *args, **kwargs):
        if name == "cloudpathlib":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cloudpathlib)
    with pytest.raises(ConfigError, match="cloudpathlib installed"):
        paths.mkpath("s3://bucket/data", None)


def test_mkpath_builds_cloud_paths_when_the_dependency_is_available(monkeypatch):
    cloudpathlib = ModuleType("cloudpathlib")
    cloudpathlib.AnyPath = lambda spec: ("cloud", spec)
    monkeypatch.setitem(sys.modules, "cloudpathlib", cloudpathlib)
    assert paths.mkpath("s3://bucket/data", None) == ("cloud", "s3://bucket/data")
