"""The front door.

Two halves, and they are independent. `graph`, `stage`, `check`, `export` read your
SOURCE — they work on a fresh checkout with no data and nothing ever run. `status`, `why`
and `plan` read the STATE FILE — they need a run to have happened but not the code.
`drift` is the one command that needs both, which is the point of it.

Nothing here reimplements anything: every command is a thin call into the module that
already does the work, so the CLI and the library cannot drift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import typer

from . import graph as _graph
from . import record as _rec
from . import render as _render
from . import state as _state
from . import static as _static
from .config import get as _cfg
from .errors import DagioError

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Data lineage and cache invalidation declared at the call sites.")


def _die(e: Exception) -> None:
    typer.secho(str(e), fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


def _build():
    return _graph.build()


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
        g = _build()
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
                for s in stage.inputs(g.scope):
                    for p in g.producers_of(s.path):
                        edge_art.setdefault(f"{p}->{node}", []).append(s.path)
        typer.echo(_render.render(order, parents, edge_art))
    except (DagioError, KeyError) as e:
        _die(e)


@app.command()
def stage(name: str) -> None:
    """One stage: what it reads, what it writes, and who is on each end."""
    try:
        g = _build()
        hits = [n for n in sorted(g.stages) if name in n]
        if not hits:
            _die(DagioError(f"no stage matching {name!r}. Known: {sorted(g.stages)}"))
        for n in hits:
            typer.echo(_render.stage_card(n, g))
    except DagioError as e:
        _die(e)


@app.command()
def check(
    trace: Path = typer.Option(None, help="also diff the code against a recorded run"),
) -> None:
    """Every structural check. Exit 1 on any error."""
    try:
        g = _build()
        errors, warns = _graph.check(g)
        if trace:
            de, dw = _graph.drift(g, _rec.load(trace))
            errors += de
            warns += dw
    except DagioError as e:
        _die(e)

    for w in warns:
        typer.secho(f"warn   {w}", fg=typer.colors.YELLOW)
    for e in errors:
        typer.secho(f"ERROR  {e}", fg=typer.colors.RED)
    if not errors and not warns:
        typer.secho(f"ok — {len(g.stages)} stages, {len(g.artifacts)} artifacts",
                    fg=typer.colors.GREEN)
    raise typer.Exit(1 if errors else 0)


@app.command()
def drift(
    trace: Path = typer.Option(None, help="the recorded run; defaults to the configured one"),
) -> None:
    """What the code declares, against what a run actually did.

    recorded but not declared is an ERROR — the process really did open that file.
    declared but not recorded is a WARN — an absent optional input, or a branch not taken.
    """
    path = trace or _cfg().trace_path
    if path is None:
        _die(DagioError(
            "no trace. Run your pipeline with DAGIO_TRACE=.dagio/trace.ndjson, "
            "or pass --trace."))
    try:
        errors, warns = _graph.drift(_build(), _rec.load(path))
    except DagioError as e:
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
        body = json.dumps(_graph.export(_build()), indent=2, sort_keys=True)
    except DagioError as e:
        _die(e)
    if out:
        out.write_text(body)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(body)


# ── the state half: reads the state file ──────────────────────────────────────

@app.command()
def status() -> None:
    """Every declared artifact: current, stale, or missing."""
    try:
        g = _build()
    except DagioError as e:
        _die(e)
    rows, stale = [], 0
    for path in g.produced:
        reason = _state.why_stale(path, _static.inputs_for_artifact(path))
        stale += reason is not None
        rows.append((path, reason))
    for path, reason in sorted(rows):
        if reason is None:
            typer.secho(f"  current  {path}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"  stale    {path}\n             {reason}",
                        fg=typer.colors.YELLOW)
    typer.echo(f"\n{len(rows) - stale}/{len(rows)} current")
    raise typer.Exit(1 if stale else 0)


@app.command()
def why(artifact: str) -> None:
    """Why one artifact would rebuild — or that it would not."""
    try:
        reason = _state.why_stale(artifact, _static.inputs_for_artifact(artifact))
    except DagioError as e:
        _die(e)
    entry = _state.record_of(artifact)
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
def plan() -> None:
    """What a run would rebuild, and what only might.

    A node's fingerprint is not knowable until it rebuilds, so anything downstream of a
    rebuild is decided on arrival rather than predicted. That is said here rather than
    papered over.
    """
    try:
        g = _build()
    except DagioError as e:
        _die(e)
    definite = {p for p in g.produced
                if _state.why_stale(p, _static.inputs_for_artifact(p)) is not None}
    parents = g.parent_map()
    maybe = set()
    frontier = set(definite)
    while frontier:
        nxt = set()
        for path in g.produced:
            if path in definite or path in maybe:
                continue
            if any(a in frontier for a in _render.ancestors_of(path, parents)):
                nxt.add(path)
        maybe |= nxt
        frontier = nxt
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
        _die(DagioError("viz needs networkx and matplotlib: pip install 'dagio[viz]'"))
    try:
        draw(_build(), out)
    except DagioError as e:
        _die(e)
    typer.echo(f"wrote {out}")


def main() -> None:
    try:
        app()
    except DagioError as e:
        typer.secho(str(e), fg=typer.colors.RED, err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
