"""Should this stage run at all.

    def main():
        ap = argparse.ArgumentParser(description=__doc__)
        dagio.add_guard_args(ap)
        args = ap.parse_args()
        dagio.build_if_needed("processed/box_features.parquet", build,
                              if_needed=args.if_needed, force=args.force)

Three properties are load-bearing, all of them learned from the shell wrapper this
replaces.

**A bare invocation always rebuilds; `--if-needed` is opt-in.** Forgetting the flag in a
pipeline script wastes time, which is recoverable. The opposite default makes a forgotten
flag silently ship stale output, which is not.

**Every output is guarded, not just the first.** A stage that writes four artifacts and is
guarded on one leaves the other three to rot.

**The stamp depends on the build succeeding.** That is `writes()`' job, not this one's —
but it is why this function does not stamp anything itself. A `|| echo` in a shell wrapper
once swallowed failures into "current".
"""
from __future__ import annotations

import argparse
from typing import Callable, Sequence

from . import state as _state


# Set by build_if_needed(force=True) and read by the partition cache. `--force` has to
# reach BOTH layers: forcing the outer guard while the inner cache reuses every partition
# is a rebuild that rebuilds nothing, and it looks exactly like it worked. Threading the
# flag by hand is the kind of thing that is right the day it is written and wrong later.
_forced = False


def forced() -> bool:
    """Did this process ask for an unconditional rebuild?"""
    return _forced


def add_guard_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--if-needed", action="store_true",
                    help="skip if every output is already current")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if current; overrides --if-needed")


def why_stale(path: str) -> str | None:
    """One line saying why `path` needs rebuilding, or None if it does not."""
    return _state.why_stale(path, _declared_inputs(path))


def current(paths: str | Sequence[str]) -> bool:
    rels = [paths] if isinstance(paths, str) else list(paths)
    return all(why_stale(r) is None for r in rels)


def _declared_inputs(rel: str) -> dict[str, object] | None:
    """What the code reads NOW, from the static scan, if the scan is available.

    Without it an input that was ADDED to a stage is invisible: the stored record has no
    entry for a path the last build never read, so there is nothing to compare against.
    Returns None when the scan cannot run, in which case the check falls back to the
    recorded input set and says so.
    """
    try:
        from .static import inputs_for_artifact
        return inputs_for_artifact(rel)
    except Exception:
        return None


def build_if_needed(paths: str | Sequence[str], build: Callable[[], None], *,
                    if_needed: bool, force: bool = False) -> bool:
    """Run `build` unless every path in `paths` is current. Returns True if it ran."""
    global _forced
    rels = [paths] if isinstance(paths, str) else list(paths)
    _forced = _forced or force

    if if_needed and not force:
        reasons = [(r, why_stale(r)) for r in rels]
        if all(reason is None for _, reason in reasons):
            for r in rels:
                print(f"  {r} is current — skipping")
            return False
        for r, reason in reasons:
            if reason is not None:
                print(f"  {r}: {reason}")
                break

    build()
    return True
