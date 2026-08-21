"""The front door.

Two halves, and they are independent. `graph`, `stage`, `check`, `export` read your
SOURCE — they work on a fresh checkout with no data and nothing ever run. `status`, `why`
and `plan` read the STATE FILE — they need a run to have happened but not the code.
`drift` is the one command that needs both, which is the point of it.

The CLI has to find your `Invalidator`, since that is where the configuration lives. One
line in pyproject.toml says where:

    [tool.iv]
    instance = "mypkg.pipeline:iv"

or pass it per-invocation with `-i mypkg.pipeline:iv`. Nothing else is read from TOML —
the instance itself is the configuration.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import tomllib
from pathlib import Path

import typer

from . import graph as _graph
from . import record as _rec
from . import render as _render
from . import static as _static
from .errors import ConfigError, InvalidatorError

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Cache invalidation and lineage declared at the call sites.")

_INSTANCE: str | None = None


@app.callback()
def main(instance: str = typer.Option(
        None, "--instance", "-i",
        help="module:attr naming your Invalidator; overrides pyproject.toml")) -> None:
    global _INSTANCE
    _INSTANCE = instance


def _die(e: Exception) -> None:
    typer.secho(str(e), fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _find_root(start: Path) -> Path | None:
    for d in (start, *start.parents):
        if (d / "pyproject.toml").exists():
            return d
    return None


def _load_instance():
    """Import the project's Invalidator. The one piece of discovery in the package."""
    target = _INSTANCE or os.environ.get("IV_INSTANCE")
    root = _find_root(Path.cwd().resolve())
    if not target and root:
        raw = tomllib.loads((root / "pyproject.toml").read_text())
        target = raw.get("tool", {}).get("iv", {}).get("instance")
    if not target:
        raise ConfigError(
            "no Invalidator to load. Add\n\n"
            "    [tool.iv]\n"
            '    instance = "mypkg.pipeline:iv"\n\n'
            "to pyproject.toml, or pass -i mypkg.pipeline:iv")
    if ":" not in target:
        raise ConfigError(f"instance must be 'module:attribute', got {target!r}")

    if root and str(root) not in sys.path:
        sys.path.insert(0, str(root))       # so `mypkg` imports from a plain checkout
    modname, attr = target.split(":", 1)
    try:
        mod = importlib.import_module(modname)
    except ImportError as e:
        raise ConfigError(f"cannot import {modname} (from instance = {target!r}): {e}") from e
    try:
        return getattr(mod, attr)
    except AttributeError as e:
        raise ConfigError(f"{modname} has no attribute {attr!r}") from e


def _graph_of():
    iv = _load_instance()
    return iv, _graph.build(iv)


# ── the graph half: reads your source ─────────────────────────────────────────

@app.command()
def graph(
    focus: str = typer.Option(None, help="only this node's ancestors and descendants"),
    artifacts: bool = typer.Option(False, help="name the artifact on each edge"),
    full: bool = typer.Option(False, help="every edge, not the transitive reduction"),
    stages_only: bool = typer.Option(True, "--stages-only/--bipartite",
                                     help="stage-to-stage, or artifacts as nodes too"),
) -> None:
    """Draw the DAG: run order downward, dependencies as lanes.

    An edge going UP is a stage reading something written later.
    """
    try:
        _, g = _graph_of()
        parents = g.stage_parents() if stages_only else g.parent_map()
        order = sorted(parents, key=lambda n: (g.order.get(n, 10 ** 9), n))
        if focus:
            order, parents = _render.focus(order, parents, focus)
        if not full:
            parents = _render.transitive_reduction(order, parents)

        edge_art = None
        if artifacts and stages_only:
            edge_art = {}
            for node, stage in g.stages.items():
                for s in stage.inputs():
                    for p in g.producers_of(s.path):
                        edge_art.setdefault(f"{p}->{node}", []).append(s.path)
        typer.echo(_render.render(order, parents, edge_art))
    except (InvalidatorError, KeyError) as e:
        _die(e)


@app.command()
def stage(name: str) -> None:
    """One stage: what it reads, what it writes, and who is on each end."""
    try:
        _, g = _graph_of()
        hits = [n for n in sorted(g.stages) if name in n]
        if not hits:
            _die(InvalidatorError(f"no stage matching {name!r}. Known: {sorted(g.stages)}"))
        for n in hits:
            typer.echo(_render.stage_card(n, g))
    except InvalidatorError as e:
        _die(e)


@app.command()
def preflight() -> None:
    """The checks that make a RUN meaningless if they fail. Exit 1 to stop a pipeline.

    Only a cycle, today. A cycle means the stage order is not a fact — what depends on
    what, what breaks if this changes, what runs first all have no answer — so a run
    built on it produces artifacts whose lineage is a guess. Everything else `iv check`
    reports is worth fixing and does not make tonight's numbers wrong.
    """
    try:
        iv, g = _graph_of()
        _graph.require_acyclic(g)
    except InvalidatorError as e:
        _die(e)
    names = _static.undefined_names(iv)
    bad = (names or []) + _static.missing_imports(iv)
    if bad:
        for line in bad:
            typer.secho(f"  {line}", fg=typer.colors.RED)
        _die(InvalidatorError(
            f"{len(bad)} name(s) or module(s) a stage uses and cannot reach. Each is an "
            f"error waiting for the stage that gets there — a refactor that renamed a "
            f"parameter and left a use behind, or an import of something retired."))
    if names is None:
        # Not an error, but not silence either: the green line below would otherwise
        # claim a check that never ran. `undefined_names` is the one that found seven
        # real breakages the day it was added.
        typer.secho("  note — pyflakes is not installed, so undefined names were NOT "
                    "checked. `uv pip install pyflakes`.", fg=typer.colors.YELLOW)
    typer.secho("  ok — no cycle, no unreachable module"
                + ("" if names is None else ", no unreachable name")
                + "; the order is defined", fg=typer.colors.GREEN)


@app.command()
def check(
    trace: Path = typer.Option(None, help="also diff the code against a recorded run"),
) -> None:
    """Every structural check. Exit 1 on any error."""
    try:
        iv, g = _graph_of()
        errors, warns = _graph.check(g)
        path = trace or iv.trace_path
        if trace or (path and path.exists()):
            de, dw = _graph.drift(g, _rec.load(path))
            errors += de
            warns += dw
    except InvalidatorError as e:
        _die(e)

    for w in warns:
        typer.secho(f"warn   {w}", fg=typer.colors.YELLOW)
    for e in errors:
        typer.secho(f"ERROR  {e}", fg=typer.colors.RED)
    if not errors and not warns:
        typer.secho(f"ok — {len(g.stages)} stages, {len(g.artifacts)} artifacts",
                    fg=typer.colors.GREEN)
    raise typer.Exit(1 if errors else 0)


# A trace older than this is refused rather than believed. Generous, because a long
# pipeline legitimately takes hours; the failure it guards against is a trace from DAYS
# ago describing code that has since been deleted.
_STALE_TRACE_S = 12 * 3600


@app.command()
def drift(
    trace: Path = typer.Option(None, help="the recorded run; defaults to the configured one"),
) -> None:
    """What the code declares, against what a run actually did.

    recorded but not declared is an ERROR — the process really did open that file.
    declared but not recorded is a WARN — an absent optional input, a branch not taken.
    """
    try:
        iv, g = _graph_of()
        path = trace or iv.trace_path
        if path is None:
            _die(InvalidatorError(
                "no trace. Construct your Invalidator with trace=..., set "
                "$IV_TRACE, or pass --trace."))
        events = _rec.load(path)
        if not events:
            _die(InvalidatorError(
                f"{path} is empty — nothing recorded this run, so there is nothing to "
                f"compare. The usual cause is the pipeline exporting a different "
                f"variable from the one `iv` reads ($IV_TRACE)."))
        age = _rec.age_of(events)
        if age is not None and age > _STALE_TRACE_S:
            _die(InvalidatorError(
                f"{path} was last written {age / 3600:.1f}h ago, which is almost "
                f"certainly not this run. A trace that outlives the code reports reads "
                f"from stages that no longer exist — every line would be fiction, and "
                f"confidently so. Re-run the pipeline, or pass a fresher --trace."))
        errors, warns = _graph.drift(g, events)
    except InvalidatorError as e:
        _die(e)
    for w in warns:
        typer.secho(f"warn   {w}", fg=typer.colors.YELLOW)
    for e in errors:
        typer.secho(f"ERROR  {e}", fg=typer.colors.RED)
    if not errors and not warns:
        typer.secho("no drift — the code and the run agree", fg=typer.colors.GREEN)
    raise typer.Exit(1 if errors else 0)


@app.command()
def export(out: Path = typer.Option(None, help="write here instead of stdout")) -> None:
    """`{nodes, parent_map}` JSON — dbt's manifest shape."""
    try:
        _, g = _graph_of()
        body = json.dumps(_graph.export(g), indent=2, sort_keys=True)
    except InvalidatorError as e:
        _die(e)
    if out:
        out.write_text(body)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(body)


# ── the state half: reads the stamps ──────────────────────────────────────────

def _facts(iv, rel: str) -> str:
    """The one-line shape of an artifact, for a human reading a wall of `current`.

    Whether something is up to date is only half of what you want to know; the other half
    is what KIND of thing it is, because that is what tells you whether "current" is a
    strong claim or a weak one. A settled archive being current says almost nothing. A
    20-partition walk-forward whose inputs are period-bounded says a great deal.

    Everything here is read off the stored record, so it describes the artifact as it was
    actually built rather than as the code currently declares it.
    """
    from .paths import fields

    rec = iv.state.record_of(rel)
    if rec is None and fields(rel):
        # A partitioned FEED: one template, many files, a record each. Summarised, because
        # "current" over a 21-season feed is a different claim from "current" over one
        # file and the path alone does not say which.
        inst = iv.state.instances_of(rel)
        stamped = [r for r in (iv.state.record_of(i) for i in inst) if r]
        out = [f"{len(inst)} file" + ("s" if len(inst) != 1 else "")]
        if len(stamped) < len(inst):
            out.append(f"{len(inst) - len(stamped)} unstamped")
        out += sorted({r.get("policy", "tracked") for r in stamped} - {"tracked"})
        if ages := [r["at"] for r in stamped if r.get("at")]:
            out.append(_ago(max(ages)))
        return " · ".join(out)
    if rec is None:
        return ""
    bits = []
    if rec.get("parts"):
        bits.append(f"{len(rec['parts'])} parts")
    policy = rec.get("policy", "tracked")
    if policy != "tracked":
        bits.append(policy)
    if rec.get("version"):
        bits.append(rec["version"])
    ins = rec.get("in") or {}
    if n := sum(1 for v in ins.values() if v.get("prior")):
        bits.append(f"prior x{n}" if n > 1 else "prior")
    if n := sum(1 for v in ins.values() if v.get("upto")):
        bits.append(f"bounded x{n}" if n > 1 else "bounded")
    if n := sum(1 for v in ins.values() if v.get("slice")):
        bits.append(f"sliced x{n}" if n > 1 else "sliced")
    if rec.get("terminal"):
        bits.append("terminal")
    if at := rec.get("at"):
        bits.append(_ago(at))
    return " · ".join(bits)


def _ago(at: str) -> str:
    """How long since it was built — the first thing you look for after a run."""
    import datetime as _dt

    try:
        sec = (_dt.datetime.now(_dt.timezone.utc)
               - _dt.datetime.fromisoformat(at)).total_seconds()
    except ValueError:
        return ""
    if sec < 90:
        return "just now"
    if sec < 5400:
        return f"{int(sec // 60)}m ago"
    if sec < 86400:
        return f"{int(sec // 3600)}h ago"
    return f"{int(sec // 86400)}d ago"


def _staleness(iv, g, fingerprint: bool) -> tuple[dict[str, str | None], set[str]]:
    """`({artifact: why_stale or None}, downstream-of-something-stale)`.

    ONE WALK, BOTH COMMANDS. `status` and `plan` answer the same question and differed
    only in how expensive an answer you asked for — but they used to compute it twice, and
    the copies disagreed: `plan` propagated staleness down the graph and `status` did not,
    so on one tree in one second `plan` called `rookie_projections` a rebuild while
    `status` called it current.

    A node's own verdict is computed against what is ON DISK NOW, so a node whose parent
    has not rebuilt YET reads as current — it really does match the parent it was built
    from. The propagation is what turns that into the useful answer: an artifact's
    identity, as its dependants see it, includes its tag, so a version bump upstream
    necessarily reaches everything below it.

    The two sets stay separate because they are different claims. A rebuild that happens
    to produce identical bytes leaves its dependants alone, so `downstream` is a statement
    about the run you are about to do, not about the tree as it stands.
    """
    parents = g.parent_map()
    reasons = {p: iv.why_stale(p, fingerprint=fingerprint) for p in g.produced}
    definite = {p for p, why in reasons.items() if why is not None}
    downstream: set[str] = set()
    frontier = set(definite)
    while frontier:
        nxt = {p for p in g.produced
               if p not in definite and p not in downstream
               and any(a in frontier for a in _render.ancestors_of(p, parents))}
        downstream |= nxt
        frontier = nxt
    return reasons, downstream


@app.command()
def status(
    fingerprint: bool = typer.Option(
        False, "--fingerprint/--no-fingerprint",
        help="also read the raw feeds; off by default because that is the only I/O"),
    ancestors: bool = typer.Option(
        True, "--ancestors/--no-ancestors",
        help="under each artifact, the inputs it was built from"),
) -> None:
    """Every declared artifact: current, stale, or missing.

    By default this answers WITHOUT touching the data: roots are taken on trust, so what
    you see is what your code, your versions and your derived artifacts imply. That is the
    cheap question and usually the one you meant. `--fingerprint` reads the raw feeds too,
    which is what a live run does.
    """
    try:
        iv, g = _graph_of()
    except InvalidatorError as e:
        _die(e)
    typer.secho(f"  {iv!r}\n", fg=typer.colors.BRIGHT_BLACK)
    parents = g.parent_map()
    reasons, downstream = _staleness(iv, g, fingerprint)
    rows = [(p, reasons[p], _facts(iv, p)) for p in g.produced]
    stale = sum(r is not None for _, r, _ in rows)
    stale_set = {p for p, r, _ in rows if r is not None}
    width = max((len(p) for p, _, f in rows if f), default=0)

    # IN RUN ORDER, not alphabetical. `g.order` ranks the stages by where the refresh
    # script actually names them, so an artifact sorts by the stage that writes it: raw
    # feeds first, then what is built from them, then the dumps. Alphabetical put
    # `dump/wvorp.json` above `processed/panel/…` and made a chain impossible to read as a
    # chain — which matters most here, where the question is usually "how far down did
    # this reach".
    def _rank(path: str) -> tuple[int, str]:
        stages = g.writers.get(path) or g.updaters.get(path) or []
        return (min((g.order.get(st, 10 ** 9) for st in stages), default=10 ** 9), path)

    for path, reason, facts in sorted(rows, key=lambda r: _rank(r[0])):
        if reason is not None:
            label, colour = "stale  ", typer.colors.YELLOW
        elif path in downstream:
            label, colour = "downstr", typer.colors.CYAN
        else:
            label, colour = "current", typer.colors.GREEN
        if facts:
            typer.secho(f"  {label}  {path.ljust(width)}", fg=colour, nl=False)
            typer.secho(f"  {facts}", fg=typer.colors.BRIGHT_BLACK)
        else:
            typer.secho(f"  {label}  {path}", fg=colour)
        if ancestors:
            # `parent_map` is BIPARTITE — an artifact's parent is the stage that writes
            # it, and that stage's parents are the artifacts it reads. One hop through
            # the stage turns "which file builds this" into "what was this built FROM",
            # which is the question being asked.
            # An artifact can legitimately be its own parent — a stage that reads one
            # declared SLICE and writes another (played vs upcoming) is a real edge, not
            # a cycle. It is still not an ANCESTOR of itself, so it is dropped here
            # rather than printed as one.
            for parent in sorted({a for st in parents.get(path, ())
                                  for a in parents.get(st, ())} - {path}):
                bad = parent in stale_set
                typer.secho(f"             {'stale' if bad else '     '}  <- {parent}",
                            fg=(typer.colors.YELLOW if bad
                                else typer.colors.BRIGHT_BLACK))
        if reason is not None:
            typer.secho(f"             {reason}", fg=typer.colors.YELLOW)
    typer.echo(f"\n{len(rows) - stale}/{len(rows)} current")
    if not fingerprint:
        typer.secho("  (raw feeds not read — pass --fingerprint for the full answer)",
                    fg=typer.colors.BRIGHT_BLACK)
    raise typer.Exit(1 if stale else 0)


@app.command()
def why(
    artifact: str,
    fingerprint: bool = typer.Option(
        True, "--fingerprint/--no-fingerprint",
        help="read the raw feeds; on here because you asked about ONE artifact"),
) -> None:
    """Why one artifact would rebuild — or that it would not."""
    try:
        iv = _load_instance()
        reason = iv.why_stale(artifact, fingerprint=fingerprint)
    except InvalidatorError as e:
        _die(e)
    entry = iv.record_of(artifact)
    if entry:
        typer.echo(f"  id   {entry['id']}")
        typer.echo(f"  fp   {entry['fp']}   ({entry.get('fp_how')})")
        typer.echo(f"  meta {entry['meta']}")
        for k, v in sorted((entry.get("in") or {}).items()):
            typer.echo(f"  in   {k}  {v['id']}")
    typer.echo("")
    if reason is None:
        typer.secho("  current", fg=typer.colors.GREEN)
    else:
        typer.secho(f"  stale: {reason}", fg=typer.colors.YELLOW)


@app.command()
def plan(
    fingerprint: bool = typer.Option(
        False, "--fingerprint/--no-fingerprint",
        help="also read the raw feeds; off by default"),
) -> None:
    """What a run would rebuild, and what only might.

    A node's fingerprint is not knowable until it rebuilds, so anything downstream of a
    rebuild is decided on arrival rather than predicted. That is said here rather than
    papered over.
    """
    try:
        iv, g = _graph_of()
    except InvalidatorError as e:
        _die(e)
    reasons, maybe = _staleness(iv, g, fingerprint)
    definite = {p for p, why in reasons.items() if why is not None}
    for p in sorted(definite):
        typer.secho(f"  rebuild  {p}", fg=typer.colors.YELLOW)
    for p in sorted(maybe):
        typer.secho(f"  maybe    {p}  (downstream of a rebuild)", fg=typer.colors.CYAN)
    if not definite and not maybe:
        typer.secho("  nothing to do", fg=typer.colors.GREEN)


@app.command()
def viz(out: Path = typer.Option(Path("dag.png"), help="where to write the image")) -> None:
    """Draw the artifact graph to a PNG. Needs the [viz] extra."""
    try:
        from .viz import draw
    except ImportError:
        _die(InvalidatorError("viz needs networkx and matplotlib: pip install 'iv[viz]'"))
    try:
        _, g = _graph_of()
        _graph.require_acyclic(g)
        draw(g, out)
    except InvalidatorError as e:
        _die(e)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
