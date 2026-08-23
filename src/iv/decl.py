"""What a stage declares, as DATA — the selector without the source text.

A stage is skipped by asking what it reads and whether those bytes moved, and the answer
has to arrive before the body runs. Today that comes from parsing the body's source for
`iv.reads(...)` calls. The catch is reach: a selector written as a call ARGUMENT
(`features(before=season)`) exists only once the call happens, so the one piece that has to
be known first is the one piece a running program cannot be asked for.

So it moves to the signature, where it is a value like any other:

    @iv.data("processed/cohorts/", why="a fit per cohort", part="season")
    def cohorts(past=iv.before_part("processed/features/", why="prior seasons")):
        ...

`inspect.signature` reads that with no source text and nothing executed — which is what a
REPL, a notebook and an `exec`'d module have in common, and what the source scan cannot do.

The tuples below are byte-identical to what `static._lit_sel` reads off the old form, so
`key_of` and `why_stale` consume them unchanged. Only where they come from is new.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import DeclError

#: Stands for the partition being built, inside a selector. It is what makes a
#: partition-relative read readable without running the closure that would otherwise
#: supply it — and that is what lets a shard's key be computed before its body runs, so
#: nothing has to be written down.
PART = "\x00PART"


def _canon(dataset: str) -> str:
    if not isinstance(dataset, str) or not dataset.strip():
        raise DeclError(f"a dataset is a relative directory path, got {dataset!r}")
    d = dataset.strip().strip("/")
    if not d or d.startswith(".") or ":" in d:
        raise DeclError(
            f"{dataset!r} is not a relative dataset path. Datasets are named relative to "
            f"the root — 'processed/box_features/' — never absolutely and never as a URI, "
            f"which is what lets an id survive the data moving.")
    return d + "/"


def _target(dataset) -> str:
    """What a read names: the DECLARATION of the dataset it reads.

    An ordinary Python reference — `iv.all_of(today, ...)` — so a typo is a NameError where
    it is written rather than a READ WITH NO PRODUCER the next time someone runs `iv check`.
    Everything is declared, sources included, so there is nothing a read can name that the
    graph does not already know about, and the checks that used to look for a path with
    nothing behind it have nothing left to find.

    Duck-typed rather than imported: `assets` imports this module, so it cannot be imported
    back.
    """
    named = getattr(dataset, "dataset", None)
    if not isinstance(named, str):
        raise DeclError(
            f"a read names a DECLARED dataset, not a path: {dataset!r}. Every dataset is "
            f"declared exactly once — one this pipeline builds with @iv.data or @iv.step, "
            f"one that arrives from outside with iv.source(...) — and read by naming that "
            f"declaration. A path written a second time is a path a rename can get half "
            f"of, and one the graph cannot tell from a typo.")
    return _canon(named)


def _why(why: object, dataset: str) -> str:
    if not isinstance(why, str) or not why.strip():
        raise DeclError(
            f"{dataset} needs why= — one line on what it is for. It is required because "
            f"there is nowhere else for it to live, which is what stops it going stale.")
    return why


@dataclass(frozen=True)
class Read:
    """One declared upstream: a dataset, and which of its shards this stage looks at.

    `kind` is the selector's shape and matches what the source scan produces:
      all    the whole dataset       -> sel ()
      in     an explicit set         -> sel ((key, ("in", (v, ...))),)
      range  a comparison            -> sel ((key, ("range", ((op, bound), ...))),)
      own    the copy on disk this stage is about to overwrite; never a trigger

    `key` is the partition key the selector applies to. It is left unset by the helpers and
    filled in by the decorator from its `part=`, so a selector never repeats what the stage
    already said about itself.
    """
    dataset: str
    kind: str
    body: tuple = ()
    optional: bool = False
    key: str | None = None
    why: str = ""
    #: True hands the body the selected PATHS instead of their contents — what `iv.reads`
    #: has always returned. It says nothing about how staleness is decided: a key is
    #: computed from FILENAMES either way, and no comparison this package makes ever opens
    #: a file. It is only about the argument. Wanted by a stage that passes the paths to
    #: something which opens them itself, that concatenates two datasets before reading, or
    #: that never looks at the value at all and declares the read so the read can make it
    #: stale — a clock being the usual one.
    as_paths: bool = False

    @property
    def is_own(self) -> bool:
        return self.kind == "own"

    def bound_to(self, part_key: str | None) -> "Read":
        """The same read, with the stage's partition key filled in."""
        if self.key is not None or self.kind in ("all", "own"):
            return self
        if part_key is None:
            raise DeclError(
                f"{self.dataset} is read relative to the partition being built, but the "
                f"stage declares no part=. A partition-relative selector only means "
                f"something where there is a partition: @iv.data(..., part='season').")
        return Read(self.dataset, self.kind, self.body, self.optional, part_key, self.why,
                    self.as_paths)

    def against(self, own: str | None) -> "Read":
        """An `own_last_copy()` that named nothing, pointed at the stage's own output."""
        if self.dataset is not None:
            return self
        if own is None:
            raise DeclError(
                "own_last_copy() means the copy of its OWN output this stage is about to "
                "overwrite, and this stage writes several — name which: "
                "own_last_copy('raw/odds_log/', why='...').")
        return Read(own, self.kind, self.body, self.optional, self.key, self.why,
                    self.as_paths)

    def sel(self) -> tuple:
        """The selector, in the shape `key_of` and `_resolve_sel` already consume."""
        if self.kind in ("all", "own"):
            return ()
        return ((self.key, (self.kind, self.body)),)

    def where(self) -> tuple:
        """The subset that names partition values OUTRIGHT, for the DAG's edge test.

        Only the explicit-set form, and only when no value is `PART`: a comparison, or a
        bound that is not known until the shard is chosen, cannot rule an edge out, and a
        missing edge is a wrong DAG.
        """
        if self.kind != "in" or any(v == PART for v in self.body):
            return ()
        return ((self.key, tuple(sorted(self.body))),)

    def triple(self) -> tuple:
        """`(dataset, sel, optional)` — what `reads_in` returns off the source today."""
        return (self.dataset, self.sel(), self.optional)


# ── the vocabulary ────────────────────────────────────────────────────────────

def all_of(dataset, *, why: str, optional: bool = False, as_paths: bool = False) -> Read:
    """Every shard. A joint fit reads this way, and that is visible here rather than
    buried in the builder."""
    d = _target(dataset)
    return Read(d, "all", (), optional, None, _why(why, d), as_paths)


def same_part(dataset, *, why: str, optional: bool = False,
              as_paths: bool = False) -> Read:
    """The one shard matching the partition being built."""
    d = _target(dataset)
    return Read(d, "in", (PART,), optional, None, _why(why, d), as_paths)


def before_part(dataset, *, why: str, inclusive: bool = False,
                optional: bool = False, as_paths: bool = False) -> Read:
    """Everything ordered before this partition — a walk-forward bound.

    The selector picks FILES, so a cohort physically cannot open a later season. One
    backfilled BELOW the bound is picked up; one added above it is not.
    """
    d = _target(dataset)
    return Read(d, "range", (("le" if inclusive else "lt", PART),), optional, None,
                _why(why, d), as_paths)


def after_part(dataset, *, why: str, inclusive: bool = False,
               optional: bool = False, as_paths: bool = False) -> Read:
    d = _target(dataset)
    return Read(d, "range", (("ge" if inclusive else "gt", PART),), optional, None,
                _why(why, d), as_paths)


def between(dataset, *, why: str, optional: bool = False, as_paths: bool = False,
            key: str | None = None, **bounds) -> Read:
    """A window: between('raw/box/', why='...', ge='2020', lt=iv.PART).

    `key=` names the partition the bounds apply to. It is only needed when the bounds are
    literal and the stage is not itself partitioned — a stage with `part=` lends its own
    key, and `iv.PART` only means anything where there is one.
    """
    d = _target(dataset)
    ops = {"lt", "le", "gt", "ge"}
    bad = sorted(set(bounds) - ops)
    if bad:
        raise DeclError(f"{d}: unknown bound(s) {bad}; expected {sorted(ops)}.")
    if not bounds:
        raise DeclError(f"{d}: between() needs at least one of {sorted(ops)}.")
    body = tuple(sorted((op, v if v == PART else str(v)) for op, v in bounds.items()))
    return Read(d, "range", body, optional, key, _why(why, d), as_paths)


def parts(dataset, *, why: str, optional: bool = False, as_paths: bool = False,
          **values) -> Read:
    """An explicit set, which is a COVERAGE CLAIM: a value that is not there is an error
    rather than a quietly shorter read."""
    d = _target(dataset)
    if len(values) != 1:
        raise DeclError(
            f"{d}: parts() names exactly one partition key — "
            f"parts('raw/box/', why='...', season=['2024', '2025']).")
    (key, vals), = values.items()
    if isinstance(vals, (str, bytes)) or not hasattr(vals, "__iter__"):
        vals = [vals]
    body = tuple(sorted(v if v == PART else str(v) for v in vals))
    if not body:
        raise DeclError(f"{d}: parts() was given no values for {key!r}.")
    return Read(d, "in", body, optional, key, _why(why, d), as_paths)


def own_last_copy(dataset=None, *, why: str, as_paths: bool = False) -> Read:
    """The copy of its own output this stage is about to overwrite.

    Recorded for lineage and EXCLUDED from the comparison — otherwise the stage would be
    permanently stale against its own last output, one step behind itself, forever. Being
    absent is normal on the first run, so it is always optional.

    It is the one read that cannot name a stage: it means THIS stage, and the `Asset` does
    not exist yet when parameter defaults are evaluated. So it names nothing at all by
    default and is filled in at decoration from the stage's own output. A stage writing
    several has to say which.
    """
    d = _target(dataset) if dataset is not None else None
    return Read(d, "own", (), True, None, _why(why, d or "own_last_copy()"), as_paths)
