"""How a file is reduced to one short string.

There is no single right answer, which is why this is a table rather than a function.
The strategy is a fact about the artifact, so it is declared at the call site as `fp=`.

    data        sum of per-row hashes         same data -> same id, whatever the byte layout
    data_order  hash of the ordered rows      the same, but row ORDER counts as data
    rows        row count from the footer     cheap on a feed too big to hash
    bytes       sha256 of the file            any rewrite is a change, even a no-op one
    present     does it exist                 write-once archives, where the answer is coverage
    <callable>  your own                      fp=lambda p: ...

`data` is the default and the one to reach for. A fetcher rewrites its output on every
run, so the parquet's bytes move even when not one number did; compression and file
metadata are not stable either. Hashing the DATA means a no-op refetch invalidates
nothing downstream, which is the whole point of the id.

**mtime is not offered.** It moves on every rewrite and moves for reasons that are not
data at all. Nor is size, for the same reason in the other direction.

**A coarse strategy on a DERIVED artifact is a correctness hazard, not just imprecision.**
`rows` on something whose row count is stable but whose numbers changed reads as
unchanged, so the id does not move and everything downstream wrongly skips. Coarse
strategies are for large raw feeds where the trade is made deliberately.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

from .errors import FingerprintError

DIGEST_LEN = 16                     # hex chars; this is an equality check, not a signature

ABSENT = "(absent)"                 # an optional input that is not there is still an id

_TABULAR = (".parquet", ".csv", ".ndjson", ".jsonl", ".arrow", ".ipc")


def _short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:DIGEST_LEN]


def _uri(path) -> str:
    """A path as something polars and pyarrow will accept.

    A `cloudpathlib.CloudPath` is not a `PathLike` as far as they are concerned — polars
    raises `Object does not have a .read() method` — but both read a `gs://` / `s3://`
    URI natively, and reading the URI directly is also FASTER than letting cloudpathlib
    download to its cache, which it re-fetches rather than reuses.
    """
    return str(path)


def read_frame(path, **kw):
    """Read a tabular file into polars, whatever its format. `iv.frame` is the caller."""
    return _frame(path, **kw)


def _frame(path, **kw):
    """Read a tabular file into polars, whatever its format."""
    try:
        import polars as pl
    except ImportError as e:
        raise FingerprintError(
            f"fp='data' needs polars (pip install 'iv[data]'); "
            f"use fp='bytes' for {path} to stay dependency-free"
        ) from e
    src = _uri(path)
    suffix = Path(src).suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(src, **kw)
    if suffix == ".csv":
        return pl.read_csv(src, **kw)
    if suffix in (".ndjson", ".jsonl"):
        return pl.read_ndjson(src, **kw)
    if suffix in (".arrow", ".ipc"):
        return pl.read_ipc(src, **kw)
    raise FingerprintError(
        f"fp='data' does not know how to read {path} — it is not one of {_TABULAR}. "
        f"Use fp='bytes' for a document, or pass fp=<callable>."
    )


def _data(path) -> str:
    """Order-INSENSITIVE. Sum of per-row hashes, so a reordered rewrite is not a change."""
    df = _frame(path)
    if df.height == 0:
        # An empty frame still has a schema, and a schema change IS a change.
        return _short("empty|" + ",".join(f"{c}:{t}" for c, t in df.schema.items()))
    # polars sums UInt64 with wraparound, which is what we want: the sum is a commutative
    # digest mod 2**64, so a reordered rewrite lands on the same value. Casting to a
    # signed int first would raise on any hash above 2**63.
    total = df.hash_rows(seed=0).sum()
    return _short(f"{df.height}|{len(df.columns)}|{total}")


def frame_digest(frame) -> str:
    """The same digest as `_data`, from a frame already in memory.

    `stamp_content()` uses this to skip re-reading a file it just wrote. It MUST stay
    identical to `_data` or the two paths would give one artifact two fingerprints.
    """
    if frame.height == 0:
        return _short("empty|" + ",".join(f"{c}:{t}" for c, t in frame.schema.items()))
    return _short(f"{frame.height}|{len(frame.columns)}|{frame.hash_rows(seed=0).sum()}")


def _data_order(path) -> str:
    """Order-SENSITIVE. For an artifact whose row order is itself an input downstream."""
    df = _frame(path)
    h = hashlib.sha256()
    h.update(f"{df.height}|{','.join(df.columns)}|".encode())
    if df.height:
        rows = df.hash_rows(seed=0)
        try:
            h.update(rows.to_numpy().tobytes())
        except (ImportError, ModuleNotFoundError):
            for v in rows.to_list():
                h.update(v.to_bytes(8, "little"))
    return h.hexdigest()[:DIGEST_LEN]


def _rows(path) -> str:
    """Footer only. One NBA play-by-play season is 220 MiB; hashing it costs more than
    the rebuild it would avoid."""
    src = _uri(path)
    if "://" not in src:
        try:
            import pyarrow.parquet as pq
            md = pq.ParquetFile(src).metadata
            return _short(f"rows|{md.num_rows}|{md.num_columns}")
        except ImportError:
            pass
    try:
        import polars as pl
    except ImportError as e:
        raise FingerprintError(
            f"fp='rows' needs pyarrow or polars to read {path}'s footer") from e
    n = pl.scan_parquet(src).select(pl.len()).collect().item()
    return _short(f"rows|{n}")


def _bytes(path) -> str:
    h = hashlib.sha256()
    # `path.open` rather than `open(str(path))`: a CloudPath URI is not a filename.
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:DIGEST_LEN]


def _present(path) -> str:
    return _short("present")


STRATEGIES: dict[str, Callable[[object], str]] = {
    "data": _data,
    "data_order": _data_order,
    "rows": _rows,
    "bytes": _bytes,
    "present": _present,
}


def compute(path, how: str | Callable = "data") -> str:
    """Fingerprint one file. `path` must exist — absence is `ABSENT`, decided by the caller.

    An unknown strategy name is an error with no fallback. A default here would be a
    constant, and an artifact keyed on a constant is permanently, silently current.
    """
    if callable(how):
        out = how(path)
        if not isinstance(out, str) or not out:
            raise FingerprintError(
                f"custom fp callable for {path} returned {out!r}; expected a non-empty str")
        return out
    fn = STRATEGIES.get(how)
    if fn is None:
        raise FingerprintError(
            f"unknown fingerprint strategy {how!r} for {path}. "
            f"Known: {sorted(STRATEGIES)}, or pass a callable.")
    return fn(path)
