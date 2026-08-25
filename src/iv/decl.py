

from __future__ import annotations

from dataclasses import dataclass

from .errors import DeclError


PART = "\x00PART"
WHY_MAX_LEN = 280


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


    named = getattr(dataset, "dataset", None)
    if not isinstance(named, str):
        raise DeclError(
            f"a read names a DECLARED dataset, not a path: {dataset!r}. Every dataset is "
            f"declared exactly once — one this pipeline writes, named in a @iv.step's "
            f"output= or declared on its own with iv.data(...), one that arrives from "
            f"outside with iv.source(...) — and read by naming that "
            f"declaration. A path written a second time is a path a rename can get half "
            f"of, and one the graph cannot tell from a typo.")
    return _canon(named)


def _why(why: object, dataset: str) -> str:
    if not isinstance(why, str) or not why.strip():
        raise DeclError(
            f"{dataset} needs why= — one line on what it is for. It is required because "
            f"there is nowhere else for it to live, which is what stops it going stale.")
    if len(why) > WHY_MAX_LEN:
        raise DeclError(
            f"{dataset} has a why= of {len(why)} characters; keep it to "
            f"{WHY_MAX_LEN} or fewer. A why is the one-line reason for a dependency, "
            f"not its whole history.")
    return why


@dataclass(frozen=True)
class Read:


    dataset: str
    kind: str
    body: tuple = ()
    optional: bool = False
    key: str | tuple[str, ...] | None = None
    why: str = ""


    as_paths: bool = False

    preserve: tuple[str, ...] = ()

    @property
    def is_own(self) -> bool:
        return self.kind == "own"

    def bound_to(self, part_keys: tuple[str, ...] | None) -> "Read":

        if isinstance(part_keys, str):
            part_keys = (part_keys,)
        if self.kind in ("all", "own", "multi"):
            return self
        if self.key is not None and not (self.kind == "range" and part_keys):
            return self
        if not part_keys:
            raise DeclError(
                f"{self.dataset} is read relative to the partition being built, but the "
                f"stage declares no part=. A partition-relative selector only means "
                f"something where there is a partition: @iv.step(..., part='season').")
        if self.kind == "range" and self.key is None and len(part_keys) != 1:
            raise DeclError(
                f"{self.dataset} is a range relative to a multi-partition stage. Name "
                "the dimension: before_part(..., key='season').")
        if self.kind == "range":
            key = self.key or part_keys[0]
            if key not in part_keys:
                raise DeclError(f"{self.dataset}: range key {key!r} is not one of "
                                f"this stage's partitions {part_keys}.")
            return Read(self.dataset, self.kind, self.body, self.optional, key, self.why,
                        self.as_paths, tuple(k for k in part_keys if k != key))
        if self.key is not None:
            return self
        return Read(self.dataset, self.kind, self.body, self.optional, part_keys, self.why,
                    self.as_paths)

    def against(self, own: str | None) -> "Read":

        if self.dataset is not None:
            return self
        if own is None:
            raise DeclError(
                "own_last_copy() means the copy of its OWN output this stage is about to "
                "overwrite, and this stage writes several — name which. It cannot name "
                "the stage, which does not exist yet where the default is evaluated, so "
                "the output has to be declared above and named here:\n"
                "    BOX = iv.data('raw/box/', why='...')\n"
                "    @iv.step(output={'box': BOX, 'pbp': PBP}, why='...')\n"
                "    def patch(was=iv.own_last_copy(BOX, why='the copy this amends')):")
        return Read(own, self.kind, self.body, self.optional, self.key, self.why,
                    self.as_paths, self.preserve)

    def sel(self) -> tuple:

        if self.kind in ("all", "own"):
            return ()
        if self.kind == "multi":
            return tuple((k, ("in", vals)) for k, vals in self.body)
        if isinstance(self.key, tuple):
            return tuple((k, (self.kind, self.body)) for k in self.key)
        return ((self.key, (self.kind, self.body)),) + tuple(
            (k, ("in", (PART,))) for k in self.preserve)

    def where(self) -> tuple:


        if self.kind == "multi":
            if any(v == PART for _, vals in self.body for v in vals):
                return ()
            return tuple((k, tuple(sorted(vals))) for k, vals in self.body)
        if self.kind != "in" or any(v == PART for v in self.body):
            return ()
        if isinstance(self.key, tuple):
            return tuple((k, tuple(sorted(self.body))) for k in self.key)
        return ((self.key, tuple(sorted(self.body))),)

    def triple(self) -> tuple:

        return (self.dataset, self.sel(), self.optional)


def all_of(dataset, *, why: str, optional: bool = False, as_paths: bool = False) -> Read:


    d = _target(dataset)
    return Read(d, "all", (), optional, None, _why(why, d), as_paths)


def same_part(dataset, *, why: str, optional: bool = False,
              as_paths: bool = False) -> Read:

    d = _target(dataset)
    return Read(d, "in", (PART,), optional, None, _why(why, d), as_paths)


def before_part(dataset, *, why: str, key: str | None = None, inclusive: bool = False,
                optional: bool = False, as_paths: bool = False) -> Read:


    d = _target(dataset)
    return Read(d, "range", (("le" if inclusive else "lt", PART),), optional, key,
                _why(why, d), as_paths)


def after_part(dataset, *, why: str, key: str | None = None, inclusive: bool = False,
               optional: bool = False, as_paths: bool = False) -> Read:
    d = _target(dataset)
    return Read(d, "range", (("ge" if inclusive else "gt", PART),), optional, key,
                _why(why, d), as_paths)


def between(dataset, *, why: str, optional: bool = False, as_paths: bool = False,
            key: str | None = None, **bounds) -> Read:


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


    d = _target(dataset)
    if not values:
        raise DeclError(f"{d}: parts() needs at least one partition key.")
    body = []
    for key, vals in values.items():
        if isinstance(vals, (str, bytes)) or not hasattr(vals, "__iter__"):
            vals = [vals]
        vals = tuple(sorted(v if v == PART else str(v) for v in vals))
        if not vals:
            raise DeclError(f"{d}: parts() was given no values for {key!r}.")
        body.append((str(key), vals))
    if len(body) == 1:
        key, vals = body[0]
        return Read(d, "in", vals, optional, key, _why(why, d), as_paths)
    return Read(d, "multi", tuple(body), optional, None, _why(why, d), as_paths)


def own_last_copy(dataset=None, *, why: str, as_paths: bool = False) -> Read:


    d = _target(dataset) if dataset is not None else None
    return Read(d, "own", (), True, None, _why(why, d or "own_last_copy()"), as_paths)
