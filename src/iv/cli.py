"""The commands.

Split by what they need. `graph`, `stage`, `check` and `export` read SOURCE, so they work
on a fresh checkout with no data and nothing ever run. `status`, `why` and `plan` read the
DATA TREE, so they describe what is actually on disk. `drift` needs both.

None of them run anything. iv observes.
"""
from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import typer

from . import graph as _graph
from . import record as _rec
from . import render as _render
from . import shards as _sh
from .errors import ConfigError, IvError

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Re-run a stage only when the data it reads has changed.")

_INSTANCE: str | None = None
_STALE_TRACE_S = 12 * 3600


@app.callback()
def main(instance: str = typer.Option(
        None, "--instance", "-i",
        help="module:attr of the Pipeline. Default: [tool.iv] instance in pyproject.toml")):
    global _INSTANCE
    _INSTANCE = instance


def _die(e: Exception) -> None:
    typer.secho(str(e), fg="red", err=True)
    raise typer.Exit(1)


def _find_root(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return None


def _load():
    """The one piece of discovery in the package: which Pipeline to talk about."""
    spec = _INSTANCE
    root = _find_root(Path.cwd())
    if not spec:
        if root is None:
            _die(ConfigError("no pyproject.toml found, and no --instance given."))
        cfg = tomllib.loads((root / "pyproject.toml").read_text())
        spec = (cfg.get("tool", {}).get("iv", {}) or {}).get("instance")
        if not spec:
            _die(ConfigError(
                'no [tool.iv] instance in pyproject.toml. Add:\n\n'
                '    [tool.iv]\n    instance = "mypkg.pipeline:iv"\n'))
    if root and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    mod, _, attr = spec.partition(":")
    try:
        return getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as e:
        _die(ConfigError(f"could not load {spec!r}: {e}"))


def _graph_of():
    try:
        return _graph.build(_load())
    except IvError as e:
        _die(e)


# ── source ────────────────────────────────────────────────────────────────────

@app.command()
def graph(focus: str = typer.Option(None, "--focus", help="only this stage and its cone"),
          full: bool = typer.Option(False, "--full", help="every edge, not the reduction")):
    """The DAG. Run order downward, dependencies as lanes.

    An edge going UP is a stage reading something written later — a bug you can see rather
    than one you have to query.
    """
    g = _graph_of()
    order, parents = g.order(), g.parent_map()
    if not full:
        parents = _render.transitive_reduction(order, parents)
    if focus:
        order, parents = _render.focus(order, parents, focus)
    typer.echo(_render.render(order, parents))


@app.command()
def stage(name: str):
    """One stage: what it reads, what it writes, who is on each end, and the whys."""
    g = _graph_of()
    hits = [n for n in g.stages if name in n]
    if not hits:
        _die(IvError(f"no stage matching {name!r}. Try `iv graph`."))
    for n in hits:
        typer.echo(_render.stage_card(n, g))


@app.command()
def check(trace: Path = typer.Option(None, "--trace", help="also diff against a run")):
    """Every structural check. Exit 1 on any error."""
    g = _graph_of()
    errors, warns = _graph.check(g)
    if trace:
        e2, w2 = _graph.drift(g, _rec.load(trace))
        errors += e2
        warns += w2
    for w in warns:
        typer.secho(f"warn  {w}", fg="yellow")
    for e in errors:
        typer.secho(f"ERROR {e}", fg="red")
    if errors:
        raise typer.Exit(1)
    typer.secho(f"ok — {len(g.stages)} stages, {len(g.produced)} datasets", fg="green")


@app.command()
def drift(trace: Path = typer.Option(None, "--trace")):
    """What the code declares against what a run actually did."""
    iv = _load()
    path = trace or iv.trace_path
    if not path:
        _die(ConfigError("no trace. Set IV_TRACE=... on the run, or pass --trace."))
    events = _rec.load(Path(path))
    if not events:
        _die(ConfigError(f"{path} has no events this recorder can read."))
    age = _rec.age_of(events)
    if age and age > _STALE_TRACE_S:
        _die(ConfigError(
            f"{path} is {age / 3600:.0f}h old. Every line would be fiction about code "
            f"that has since changed — re-run with IV_TRACE set."))
    errors, warns = _graph.drift(_graph.build(iv), events)
    for w in warns:
        typer.secho(f"warn  {w}", fg="yellow")
    for e in errors:
        typer.secho(f"ERROR {e}", fg="red")
    if errors:
        raise typer.Exit(1)
    typer.secho("ok — the code and the run agree", fg="green")


@app.command()
def export(out: Path = typer.Option(None, "--out", help="write here instead of stdout")):
    """`{nodes, parent_map, datasets}` as JSON."""
    import json
    body = json.dumps(_graph_of().export(), indent=1, sort_keys=True)
    if out:
        out.write_text(body)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(body)


@app.command()
def viz(out: Path = typer.Option(Path("dag.png"), "--out")):
    """Draw the dataset graph. Needs the [viz] extra."""
    from . import viz as _viz
    typer.echo(f"wrote {_viz.draw(_graph_of(), out)}")


# ── the data tree ─────────────────────────────────────────────────────────────

def _staleness(iv, g):
    """`{dataset: reason or None}` in run order, plus the set that will rebuild."""
    out, stale = {}, set()
    for node in g.order():
        for site in g.stages[node].outputs:
            name = site.dataset
            if name in out:
                continue
            d = iv.resolve_out(name)
            parts = sorted(_sh.current_shards(d)) or [""]
            reasons = [(p, iv.why_stale(name, _sh.decode_part(p) or None))
                       for p in parts]
            bad = [(p, r) for p, r in reasons if r]
            out[name] = None if not bad else (
                f"{bad[0][1]}" if len(bad) == 1 and not bad[0][0]
                else f"{len(bad)}/{len(reasons)} shards: {bad[0][1]}")
            if bad:
                stale.add(name)
    return out, stale


@app.command()
def status():
    """Current or stale, per dataset, in run order. Exit 1 if anything is stale."""
    iv = _load()
    g = _graph_of()
    reasons, stale = _staleness(iv, g)
    for name, why in reasons.items():
        n = len(_sh.current_shards(iv.resolve_out(name))) if not why else 0
        if why:
            typer.secho(f"  stale    {name:<44} {why}", fg="yellow")
        else:
            typer.secho(f"  current  {name:<44} {n} shard(s)", fg="green")
    typer.echo(f"\n{len(reasons) - len(stale)}/{len(reasons)} current")
    if stale:
        raise typer.Exit(1)


@app.command()
def why(dataset: str):
    """One dataset: every shard, what it holds, and what it was built from."""
    iv = _load()
    d = iv.resolve_out(dataset)
    try:
        present = _sh.current_shards(d)
    except IvError as e:
        _die(e)
    if not present:
        typer.echo(f"{dataset}: nothing on disk")
        raise typer.Exit(1)
    index = (_sh.read_index(d).get("shards") or {})
    for part in sorted(present, key=_sh.sort_key):
        sh = present[part]
        reason = iv.why_stale(dataset, _sh.decode_part(part) or None)
        typer.echo(f"\n{dataset}{part or '(one shard)'}")
        typer.echo(f"  fp      {sh.fp}")
        entry = index.get(part, {})
        for k in ("by", "at", "seconds", "why"):
            if entry.get(k) not in (None, ""):
                typer.echo(f"  {k:<7} {entry[k]}")
        for name, was in sorted((entry.get("inputs") or {}).items()):
            typer.echo(f"  in      {name:<40} {was.get('id')} ({len(was.get('parts', []))})")
        for name in entry.get("prior") or ():
            typer.echo(f"  prior   {name}")
        for ext in entry.get("external") or ():
            typer.echo(f"  from    {ext}")
        if reason:
            typer.secho(f"  stale: {reason}", fg="yellow")
        else:
            typer.secho("  current", fg="green")


@app.command()
def plan():
    """What would rebuild, and what might because it sits downstream of one."""
    iv = _load()
    g = _graph_of()
    reasons, stale = _staleness(iv, g)
    if not stale:
        typer.echo("nothing to do")
        return
    parents = g.parent_map()
    writers = {s.dataset: n for n, st in g.stages.items() for s in st.outputs}
    downstream = set()
    for node in g.order():
        ps = parents.get(node, [])
        if any(writers.get(name) in ps for name in stale if writers.get(name)):
            for s in g.stages[node].outputs:
                if s.dataset not in stale:
                    downstream.add(s.dataset)
    for name, r in reasons.items():
        if r:
            typer.secho(f"  rebuild  {name:<44} {r}", fg="yellow")
    for name in sorted(downstream):
        typer.secho(f"  maybe    {name:<44} (downstream of a rebuild)", fg="cyan")


@app.command()
def gc(dataset: str = typer.Argument(None, help="one dataset, or all of them")):
    """Drop shards no longer current — what an interrupted commit leaves behind."""
    iv = _load()
    g = _graph_of()
    targets = [dataset] if dataset else g.produced
    total = 0
    for name in targets:
        d = iv.resolve_out(name)
        found = _sh.list_shards(d) if d.exists() else {}
        keep = {sorted(v, key=lambda s: s.name)[0].name for v in found.values()}
        removed = _sh.gc(d, keep=keep) if any(len(v) > 1 for v in found.values()) else []
        for name in removed:
            typer.echo(f"  dropped {name}{name}")
        total += len(removed)
    typer.echo(f"{total} shard(s) dropped")
