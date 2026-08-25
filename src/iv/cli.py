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
        help="module:attr of the Invalidator. Default: [tool.iv] instance in pyproject.toml")):
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


def _stage_name(g, query: str) -> str:
    hits = [n for n in g.stages if query == n or query in n]
    if not hits:
        raise IvError(f"no stage matching {query!r}. Try `iv graph`.")
    if len(hits) > 1:
        raise IvError(f"{query!r} matches more than one stage: {', '.join(hits)}.")
    return hits[0]


def _cone(parents: dict[str, list[str]], start: str, *, reverse=False) -> set[str]:
    edges = parents
    if reverse:
        edges = {n: [] for n in parents}
        for child, ps in parents.items():
            for parent in ps:
                edges[parent].append(child)
    out, todo = set(), [start]
    while todo:
        node = todo.pop()
        if node in out:
            continue
        out.add(node)
        todo += edges[node]
    return out


def _part_flags(raw: list[str]) -> dict[str, str]:
    out = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise IvError(f"--part is key=value, got {item!r}.")
        if key in out and out[key] != value:
            raise IvError(f"--part names {key!r} twice with different values.")
        out[key] = value
    return out


def _asset_parts(iv, asset, filters: dict[str, str]) -> list[dict | None]:
    if not asset.part_keys or asset.split:
        return [None]
    parts = asset.universe_parts()
    if parts is None:
        parts = iv.partitions_for(asset.part_keys)
    return [p for p in parts if all(p.get(k) == v for k, v in filters.items() if k in p)]


def _stale_asset_parts(iv, asset, filters) -> list[dict | None]:
    out = []
    for p in _asset_parts(iv, asset, filters):
        if asset.why_stale(**p) if p is not None else asset.why_stale():
            out.append(p)
    return out


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
def viz(out: Path = typer.Option(Path("dag.png"), "--out"),
        full: bool = typer.Option(False, "--full", help="every edge, not the reduction"),
        plain: bool = typer.Option(False, "--plain",
                                   help="do not read the tree; leave every node grey"),
        html: bool = typer.Option(False, "--html",
                                  help="an interactive page instead of a picture"),
        reduce: bool = typer.Option(False, "--reduce",
                                    help="--html: hide edges a longer path implies")):
    """Draw the DAG: colour is `iv status`, shape is what kind of dataset it is.

    `--html` writes a page instead: click a node for what it is for, what is stale about
    it, and which nodes are upstream and downstream of it.
    """
    from . import viz as _viz
    iv = _load()
    g = _graph.build(iv)
    status, state, maybe = {}, {}, set()
    if not plain:
        with _sh.snapshot():
            state = _staleness(iv, g)
            maybe = _downstream_of(g, state)
        status = _viz.states(state, maybe)
    if html:
        from . import web as _web
        if out.suffix == ".png":
            out = out.with_suffix(".html")
        got = _web.write(g, out, status=status, state=state, maybe=maybe,
                         title=iv.tree.name, reduce=reduce)
        typer.echo(f"wrote {got} — open it")
        return
    typer.echo(f"wrote {_viz.draw(g, out, full=full, status=status)}")


def _staleness(iv, g):
    """Per SHARD: {dataset: {part_str: reason or None}}.

    The CLI has no running stage to ask, so it reads each stage's declared upstreams off
    the graph — the same triples the run takes off the decorated function, arrived at the
    other way round.

    WHICH STAGE OWNS WHICH SHARD. Two stages may share a dataset by writing different
    partitions of it — the played and the unplayed half of a prediction table, three blocks
    of a college feature table — and each has its own upstreams. This used to take the first
    writer in the order and judge every shard against it, so most of them were answered with
    the wrong question.
    """
    writers: dict[str, list[tuple]] = {}
    for node in g.order():
        inputs = tuple((s.dataset, s.sel, s.optional) for s in g.stages[node].triggers)
        for site in g.stages[node].outputs:
            writers.setdefault(site.dataset, []).append((tuple(site.part), inputs))

    out: dict[str, dict] = {}
    for name, owners in writers.items():
        shards = {}
        for p in (sorted(_sh.current_shards(iv.resolve_out(name))) or [""]):
            part = _sh.decode_part(p) or None
            shards[p] = iv.why_stale(name, part, inputs=_owner(owners, part))
        out[name] = shards
    return out


def _stale_shards(state: dict) -> set:
    return {(ds, p) for ds, shards in state.items()
            for p, why in shards.items() if why}


def _which(parts: list[str], total: int) -> str:
    """Name the shards, compactly: a list while it is readable, a range when it is not."""
    named = sorted(parts, key=_sh.sort_key)
    if total == 1:
        return named[0] or "(one shard)"
    if len(named) == total:
        return f"all {total}"
    shown = ", ".join(named[:3]) if len(named) <= 3 else f"{named[0]}..{named[-1]}"
    return f"{len(named)}/{total} ({shown})"


def _line(shards: dict, maybe: set, dataset: str) -> tuple:
    """One dataset's line: its worst state, and the shard counts behind it.

    A dataset is rarely all one thing. One season of a panel is stale because its own feed
    moved and the other twenty may follow because they share a crosswalk — reporting only
    the worst of those loses the number that says how much work is coming.
    """
    total = len(shards)
    bad = {p: why for p, why in shards.items() if why}
    soft = [p for p in shards if not shards[p] and (dataset, p) in maybe]
    if not bad and not soft:
        return "current", f"{total} shard(s)"

    parts = []
    if bad:
        groups: dict[str, list[str]] = {}
        for p, why in bad.items():
            groups.setdefault(why, []).append(p)
        parts += [f"{_which(ps, total)} stale: {why}" for why, ps in groups.items()]
    if soft:
        parts.append(f"{_which(soft, total)} may follow")
    return ("stale" if bad else "maybe"), "; ".join(parts)


def _downstream_of(g, state: dict, iv=None) -> set:
    """Shards that read something being rebuilt: current now, and possibly not after.

    POSSIBLY, not certainly, which is the whole reason this is a third state rather than
    more red. A rebuild that produces the same bytes commits the same shard and stops
    there, so on an ordinary day the poll re-fetches, writes what it wrote yesterday, and
    nothing below it moves.

    PER SHARD, because that is the question worth answering. A cohort fit on seasons
    before 2010 cannot see a shard of 2026, and saying so is the difference between "this
    dataset may move" and "these three of its twenty-one may". The selector is data, so it
    can be resolved for each of a stage's own partitions and asked whether it reaches a
    shard that is moving — which is exactly what `select` asks of a directory.
    """
    from .core import _resolve_sel
    moving = _stale_shards(state)
    for node in g.order():                      # topological: parents decided first
        st = g.stages[node]
        for site in st.outputs:
            fixed = dict(site.part) or None
            for p in state.get(site.dataset, {""}):
                if (site.dataset, p) in moving:
                    continue
                part = _sh.decode_part(p) or fixed
                for t in st.triggers:
                    where = _resolve_sel(t.sel, part, t.dataset)
                    if any(_sh.covers(where, _sh.decode_part(q) or {})
                           for (d, q) in moving if d == t.dataset):
                        moving.add((site.dataset, p))
                        break
    return moving - _stale_shards(state)


def _owner(owners, part) -> tuple:
    """The upstreams of the stage that writes THIS shard.

    A writer naming a literal part= owns exactly that shard. One naming none owns whatever
    is left — a `for_each` does not know its keys until it runs, so it cannot name them.
    """
    if part:
        for fixed, inputs in owners:
            if fixed and all(str(part.get(k)) == v for k, v in fixed):
                return inputs
    for fixed, inputs in owners:
        if not fixed:
            return inputs
    return owners[0][1]


@app.command()
@reports
def status():
    iv = _load()
    g = _graph_of()
    # Nothing here writes, so one view of the tree answers every question — without it the
    # same input directory is re-listed once per partition of every dataset that reads it.
    with _sh.snapshot():
        state = _staleness(iv, g)
        maybe = _downstream_of(g, state)

    tally = {"current": 0, "maybe": 0, "stale": 0}
    for name, shards in state.items():
        tier, note = _line(shards, maybe, name)
        typer.secho(f"  {tier:<8} {name:<44} {note}",
                    fg={"current": "green", "maybe": "cyan", "stale": "yellow"}[tier])
        for p, why in shards.items():
            tally["stale" if why else "maybe" if (name, p) in maybe else "current"] += 1

    total = sum(tally.values())
    typer.echo(f"\n{total} shard(s): {tally['stale']} stale, "
               f"{tally['maybe']} may follow, {tally['current']} current")
    if tally["stale"]:
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
        state = _staleness(iv, g)
        maybe = _downstream_of(g, state)
    if not _stale_shards(state):
        typer.echo("nothing to do")
        return
    for name, shards in state.items():
        tier, note = _line(shards, maybe, name)
        if tier == "stale":
            typer.secho(f"  rebuild  {name:<44} {note}", fg="yellow")
    for name, shards in state.items():
        tier, note = _line(shards, maybe, name)
        if tier == "maybe":
            typer.secho(f"  maybe    {name:<44} {note}", fg="cyan")


@app.command()
@reports
def run(
    up_to: str = typer.Option(None, "--up-to", help="run this stage and everything it needs"),
    up_to_excluding: str = typer.Option(None, "--up-to-excluding", help="run prerequisites, not this stage"),
    from_: str = typer.Option(None, "--from", help="run this stage and descendants; upstream must be current"),
    only: str = typer.Option(None, "--only", help="run exactly this stage; upstream must be current"),
    part: list[str] = typer.Option([], "--part", help="partition filter, repeat as key=value"),
):
    """Execute pipeline stages in dependency order.

    Partitioned stages are expanded from Invalidator(partitions=...). `--from` and
    `--only` are deliberately conservative: they never silently leave stale inputs behind.
    """
    choices = [x for x in (up_to, up_to_excluding, from_, only) if x]
    if len(choices) > 1:
        raise IvError("choose only one of --up-to, --up-to-excluding, --from, or --only.")
    iv = _load()
    g = _graph.build(iv)
    parents = g.parent_map()
    filters = _part_flags(part)
    selected = set(g.stages)
    safe = False
    if up_to:
        selected = _cone(parents, _stage_name(g, up_to))
    elif up_to_excluding:
        target = _stage_name(g, up_to_excluding)
        selected = _cone(parents, target) - {target}
    elif from_:
        target = _stage_name(g, from_)
        selected = _cone(parents, target, reverse=True)
        safe = True
    elif only:
        selected = {_stage_name(g, only)}
        safe = True

    if safe:
        required = set()
        for node in selected:
            required.update(_cone(parents, node) - selected)
        stale = []
        for node in g.order():
            if node not in required:
                continue
            asset = iv._assets[node]
            for p in _stale_asset_parts(iv, asset, filters):
                stale.append(f"{node} {p or '(one shard)'}")
        if stale:
            raise IvError("refusing to run with stale upstream shard(s): " + "; ".join(stale)
                          + ". Run `iv run --up-to " + choices[0] + "` first.")

    ran = 0
    for node in g.order():
        if node not in selected:
            continue
        asset = iv._assets[node]
        for p in _asset_parts(iv, asset, filters):
            changed = asset(**p) if p is not None else asset()
            if changed is not False:
                ran += 1
                typer.echo(f"  ran      {node}" + (f" {p}" if p else ""))
    typer.echo(f"{ran} stage shard(s) ran")


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
