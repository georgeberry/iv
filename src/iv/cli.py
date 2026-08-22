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
from . import static as _static
from .core import _canon, _resolve_sel
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


def reports(fn):
    """A command that touches the data tree reports what is wrong with it.

    `IvError` says something true and actionable about the tree — a file that is not a shard,
    two shards for one partition, an index from another version. A traceback buries that
    under twenty lines of this package's own frames, which is nobody's problem but ours.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **kw):
        try:
            return fn(*a, **kw)
        except IvError as e:
            _die(e)
    return wrapper


def _die(e: Exception) -> None:
    typer.secho(str(e), fg="red", err=True)
    raise typer.Exit(1)


def _find_root(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / "pyproject.toml").exists():
            return d
    return None


def _load():
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


@app.command()
def graph(focus: str = typer.Option(None, "--focus", help="only this stage and its cone"),
          full: bool = typer.Option(False, "--full", help="every edge, not the reduction")):
    g = _graph_of()
    order, parents = g.order(), g.parent_map()
    if not full:
        parents = _render.transitive_reduction(order, parents)
    if focus:
        order, parents = _render.focus(order, parents, focus)
    typer.echo(_render.render(order, parents))


@app.command()
def stage(name: str):
    g = _graph_of()
    hits = [n for n in g.stages if name in n]
    if not hits:
        _die(IvError(f"no stage matching {name!r}. Try `iv graph`."))
    for n in hits:
        typer.echo(_render.stage_card(n, g))


@app.command()
def preflight():
    """Stop a run before it starts: undefined names, missing modules, cycles."""
    iv = _load()
    g = _graph.build(iv)
    bad = []
    names = _static.undefined_names(iv)
    if names is None:
        typer.secho("  pyflakes is not installed, so undefined names were NOT checked",
                    fg="yellow")
    for line in names or ():
        bad.append(f"UNDEFINED NAME  {line}")
    for line in _static.missing_imports(iv):
        bad.append(f"MISSING MODULE  {line}")
    cyc = _graph.find_cycle(g)
    if cyc:
        bad.append(f"CYCLE  {cyc}")
    for b in bad:
        typer.secho(b, fg="red")
    if bad:
        raise typer.Exit(1)
    typer.secho(f"ok — {len(g.stages)} stages, no cycle, no unreachable name", fg="green")


@app.command()
def check(trace: Path = typer.Option(None, "--trace", help="also diff against a run")):
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
    import json
    body = json.dumps(_graph_of().export(), indent=1, sort_keys=True)
    if out:
        out.write_text(body)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(body)


@app.command()
def viz(out: Path = typer.Option(Path("dag.png"), "--out")):
    from . import viz as _viz
    typer.echo(f"wrote {_viz.draw(_graph_of(), out)}")


def _staleness(iv, g):
    """The CLI has no running step to ask, so it reads each stage's declared upstreams off
    the static scan — the same `(dataset, selector, optional)` triples `reads_in` takes off
    the source at runtime, arrived at the other way round.

    It must NOT fall back to the runtime registry: `_load()` imports the module holding the
    Pipeline, not the module holding the steps, so nothing would be registered and every
    dataset would look like a root — which reads as `current` for anything with a file on
    disk. A green that means "I could not find the question" is worse than a red."""
    out, stale = {}, set()
    for node in g.order():
        inputs = tuple((s.dataset, s.sel, s.optional) for s in g.stages[node].triggers)
        for site in g.stages[node].outputs:
            name = site.dataset
            if name in out:
                continue
            d = iv.resolve_out(name)
            parts = sorted(_sh.current_shards(d)) or [""]
            reasons = [(p, iv.why_stale(name, _sh.decode_part(p) or None, inputs=inputs))
                       for p in parts]
            bad = [(p, r) for p, r in reasons if r]
            out[name] = None if not bad else (
                f"{bad[0][1]}" if len(bad) == 1 and not bad[0][0]
                else f"{len(bad)}/{len(reasons)} shards: {bad[0][1]}")
            if bad:
                stale.add(name)
    return out, stale


@app.command()
@reports
def status():
    iv = _load()
    g = _graph_of()
    # Nothing here writes, so one view of the tree answers every question — without it the
    # same input directory is re-listed once per partition of every dataset that reads it.
    with _sh.snapshot():
        reasons, stale = _staleness(iv, g)
        counts = {name: (len(_sh.current_shards(iv.resolve_out(name))) if not why else 0)
                  for name, why in reasons.items()}
    for name, why in reasons.items():
        n = counts[name]
        if why:
            typer.secho(f"  stale    {name:<44} {why}", fg="yellow")
        else:
            typer.secho(f"  current  {name:<44} {n} shard(s)", fg="green")
    typer.echo(f"\n{len(reasons) - len(stale)}/{len(reasons)} current")
    if stale:
        raise typer.Exit(1)


@app.command()
@reports
def why(dataset: str):
    iv = _load()
    g = _graph_of()
    d = iv.resolve_out(dataset)
    try:
        present = _sh.current_shards(d)
    except IvError as e:
        _die(e)
    if not present:
        typer.echo(f"{dataset}: nothing on disk")
        raise typer.Exit(1)
    node = next((n for n, st in g.stages.items()
                 if any(s.dataset == _canon(dataset) for s in st.outputs)), None)
    inputs = tuple((s.dataset, s.sel, s.optional)
                   for s in g.stages[node].triggers) if node else ()
    # One view of the tree for the whole report: every partition asks the same upstream
    # directories the same question, and nothing here writes.
    with _sh.snapshot():
        for part in sorted(present, key=_sh.sort_key):
            sh = present[part]
            reason = iv.why_stale(dataset, _sh.decode_part(part) or None, inputs=inputs)
            typer.echo(f"\n{_canon(dataset)}{part or '(one shard)'}")
            typer.echo(f"  fp      {sh.fp}")
            typer.echo(f"  key     {sh.key or ('(no key in the name — not written by the '
                                           'pipeline as it stands)' if node else
                                           '(a root — nothing here derives it)')}")
            if node:
                typer.echo(f"  by      {node}")
            # A key does not invert, so this cannot say which input moved. It says what the
            # upstreams are RIGHT NOW, which is the same question asked forwards.
            for name, sel, _ in inputs:
                live = _sh.current_shards(iv.resolve(name))
                try:
                    got = _sh.select(live, _resolve_sel(sel, _sh.decode_part(part) or None,
                                                        name), dataset=name)
                except IvError:
                    got = []
                typer.echo(f"  in      {name:<40} {_sh.dataset_id(got)} ({len(got)})")
            if reason:
                typer.secho(f"  stale: {reason}", fg="yellow")
            else:
                typer.secho("  current", fg="green")


@app.command()
@reports
def plan():
    iv = _load()
    g = _graph_of()
    with _sh.snapshot():
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
@reports
def verify(dataset: str = typer.Argument(None, help="one dataset, or all of them")):
    """Re-fingerprint every shard and check it still matches the name it is filed under."""
    iv = _load()
    g = _graph_of()
    bad = []
    for name in ([dataset] if dataset else g.produced):
        if iv.resolve_out(name).exists():
            bad += [f"{name}{line}" for line in iv.verify(name)]
    for b in bad:
        typer.secho(b, fg="red")
    if bad:
        raise typer.Exit(1)
    typer.secho("ok — every shard matches its name", fg="green")


@app.command()
@reports
def gc(dataset: str = typer.Argument(None, help="one dataset, or all of them")):
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
