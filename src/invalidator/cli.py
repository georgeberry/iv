"""The front door.

Two halves, and they are independent. `graph`, `stage`, `check`, `export` read your
SOURCE — they work on a fresh checkout with no data and nothing ever run. `status`, `why`
and `plan` read the STATE FILE — they need a run to have happened but not the code.
`drift` is the one command that needs both, which is the point of it.

The CLI has to find your `Invalidator`, since that is where the configuration lives. One
line in pyproject.toml says where:

    [tool.invalidator]
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
from .errors import ConfigError, DagioError

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
    target = _INSTANCE or os.environ.get("INVALIDATOR_INSTANCE")
    root = _find_root(Path.cwd().resolve())
    if not target and root:
        raw = tomllib.loads((root / "pyproject.toml").read_text())
        target = raw.get("tool", {}).get("invalidator", {}).get("instance")
    if not target:
        raise ConfigError(
            "no Invalidator to load. Add\n\n"
            "    [tool.invalidator]\n"
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
    except (DagioError, KeyError) as e:
        _die(e)


@app.command()
def stage(name: str) -> None:
    """One stage: what it reads, what it writes, and who is on each end."""
    try:
        _, g = _graph_of()
        hits = [n for n in sorted(g.stages) if name in n]
        if not hits:
            _die(DagioError(f"no stage matching {name!r}. Known: {sorted(g.stages)}"))
        for n in hits:
            typer.echo(_render.stage_card(n, g))
    except DagioError as e:
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
    except DagioError as e:
        _die(e)
    bad = _static.undefined_names(iv) + _static.missing_imports(iv)
    if bad:
        for line in bad:
            typer.secho(f"  {line}", fg=typer.colors.RED)
        _die(DagioError(
            f"{len(bad)} name(s) or module(s) a stage uses and cannot reach. Each is an "
            f"error waiting for the stage that gets there — a refactor that renamed a "
            f"parameter and left a use behind, or an import of something retired."))
    typer.secho("  ok — no cycle, no unreachable name or module; the order is defined",
                fg=typer.colors.GREEN)


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
    declared but not recorded is a WARN — an absent optional input, a branch not taken.
    """
    try:
        iv, g = _graph_of()
        path = trace or iv.trace_path
        if path is None:
            _die(DagioError(
                "no trace. Construct your Invalidator with trace=..., set "
                "$INVALIDATOR_TRACE, or pass --trace."))
        errors, warns = _graph.drift(g, _rec.load(path))
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
        _, g = _graph_of()
        body = json.dumps(_graph.export(g), indent=2, sort_keys=True)
    except DagioError as e:
        _die(e)
    if out:
        out.write_text(body)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(body)


# ── the state half: reads the stamps ──────────────────────────────────────────

@app.command()
def status(
    fingerprint: bool = typer.Option(
        False, "--fingerprint/--no-fingerprint",
        help="also read the raw feeds; off by default because that is the only I/O"),
) -> None:
    """Every declared artifact: current, stale, or missing.

    By default this answers WITHOUT touching the data: roots are taken on trust, so what
    you see is what your code, your versions and your derived artifacts imply. That is the
    cheap question and usually the one you meant. `--fingerprint` reads the raw feeds too,
    which is what a live run does.
    """
    try:
        iv, g = _graph_of()
    except DagioError as e:
        _die(e)
    typer.secho(f"  {iv!r}\n", fg=typer.colors.BRIGHT_BLACK)
    rows, stale = [], 0
    for path in g.produced:
        reason = iv.why_stale(path, fingerprint=fingerprint)
        stale += reason is not None
        rows.append((path, reason))
    for path, reason in sorted(rows):
        if reason is None:
            typer.secho(f"  current  {path}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"  stale    {path}\n             {reason}",
                        fg=typer.colors.YELLOW)
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
    except DagioError as e:
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
    except DagioError as e:
        _die(e)
    definite = {p for p in g.produced
                if iv.why_stale(p, fingerprint=fingerprint) is not None}
    parents = g.parent_map()
    maybe, frontier = set(), set(definite)
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
        _die(DagioError("viz needs networkx and matplotlib: pip install 'iv[viz]'"))
    try:
        _, g = _graph_of()
        _graph.require_acyclic(g)
        draw(g, out)
    except DagioError as e:
        _die(e)
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
