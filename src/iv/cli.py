from __future__ import annotations

import importlib
import io
import sys
import tempfile
import tomllib
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

import typer

from . import graph as _graph
from . import paths as _paths
from . import record as _rec
from . import render as _render
from . import shards as _sh
from . import static as _static
from .core import _canon, _resolve_sel
from .errors import ConfigError, DeclError, IvError

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


def _part_flags(raw: list[str], option: str = "--part") -> dict[str, str]:
    out = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise IvError(f"{option} is key=value, got {item!r}.")
        if key in out and out[key] != value:
            raise IvError(f"{option} names {key!r} twice with different values.")
        out[key] = value
    return out


def _asset_parts(iv, asset, filters: dict[str, str]) -> list[dict | None]:
    if not asset.part_keys or asset.split:
        return [None]
    parts = asset.universe_parts()
    if parts is None:
        raise DeclError(
            f"{asset.primary}: iv run needs universe= to enumerate this dynamic stage. "
            "Use universe=[...] (or a callable), or build an explicit shard directly.")
    return [p for p in parts if all(p.get(k) == v for k, v in filters.items() if k in p)]


def _stale_asset_parts(iv, asset, filters) -> list[dict | None]:
    out = []
    for p in _asset_parts(iv, asset, filters):
        if asset.why_stale(**p) if p is not None else asset.why_stale():
            out.append(p)
    return out


def _part_label(part: dict | None) -> str:
    if not part:
        return ""
    return " [" + ", ".join(f"{k}={v}" for k, v in sorted(part.items())) + "]"


class _LiveOutput(io.TextIOBase):

    def __init__(self, console, sink=None) -> None:
        self.console = console
        self.sink = sink
        self.pending = ""
        self.started = False

    def write(self, text) -> int:
        if self.sink:
            self.sink.write(text)
            self.sink.flush()
        self.pending += text
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self._line(line)
        self.console.flush()
        return len(text)

    def _line(self, line: str) -> None:
        if not self.started:
            self.console.write("  output\n")
            self.started = True
        self.console.write(f"  │ {line}\n")

    def finish(self) -> None:
        if self.pending:
            self._line(self.pending)
            self.pending = ""
        self.console.flush()

    def flush(self) -> None:
        if self.sink:
            self.sink.flush()
        self.console.flush()


def _open_run_log(path: Path | None, total: int):
    if path is None:
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = path.open("w", encoding="utf-8", buffering=1)
    except OSError as e:
        raise IvError(f"cannot open --log {path}: {e}") from e
    out.write(f"iv run · {total} stage shard(s)\n")
    out.write(f"started {time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
    out.flush()
    return out


def _run_causes(asset, part: dict | None,
                changed: set[tuple[str, str]]) -> list[str]:
    out = []
    for read in asset.triggers:
        where = _resolve_sel(read.sel(), part, read.dataset)
        for dataset, part_str in sorted(changed):
            if dataset != read.dataset:
                continue
            if _sh.covers(where, _sh.decode_part(part_str)):
                out.append(f"{dataset}{part_str or '(one shard)'}")
    return sorted(set(out))


def _rebuild_reason(iv, asset, part: dict | None, stale: str | None,
                    changed: set[tuple[str, str]]) -> str:
    if iv.force:
        return "forced by IV_FORCE"
    if asset.acts_only:
        return "action has no output to mark current"
    causes = _run_causes(asset, part, changed)
    if causes:
        shown = ", ".join(causes[:3])
        return f"upstream changed: {shown}" + (
            f" (+{len(causes) - 3} more)" if len(causes) > 3 else "")
    if stale:
        return stale
    if not asset.may_skip:
        return "root has no declared inputs"
    return "declared inputs, version, or schema changed"


def _execute_work(iv, work, log: Path | None) -> None:
    run_log = _open_run_log(log, len(work))
    typer.secho(f"iv run · {len(work)} stage shard(s)", bold=True)
    for i, (node, _, p) in enumerate(work, 1):
        typer.echo(f"  {i:>3}. {node}{_part_label(p)}")
    typer.echo()
    ran = skipped = 0
    changed: set[tuple[str, str]] = set()
    started = time.perf_counter()
    for i, (node, asset, p) in enumerate(work, 1):
        heading = f"[{i}/{len(work)}] {node}{_part_label(p)}"
        typer.secho(heading, bold=True)
        if run_log:
            run_log.write(f"\n{heading}\n"); run_log.flush()
        output = _LiveOutput(sys.stdout, run_log)
        step_started = time.perf_counter()
        try:
            args = p or {}
            stale = asset.why_stale(**args)
            will_run = iv.force or not asset.may_skip or bool(stale)
            reason = _rebuild_reason(iv, asset, p, stale, changed) if will_run else None
            if reason:
                typer.secho(f"  rebuild — {reason}", fg="yellow")
                if run_log:
                    run_log.write(f"[iv] rebuild — {reason}\n"); run_log.flush()
            iv._changes.clear()
            with redirect_stdout(output), redirect_stderr(output):
                asset._invoke(p, stale)
            changed.update(iv._changes)
            checkpoint = getattr(iv, "_remote_checkpoint", None)
            if checkpoint is not None:
                checkpoint()
        except BaseException:
            output.finish()
            if run_log:
                run_log.write(f"\n[iv] failed ({time.perf_counter() - step_started:.2f}s)\n")
                run_log.close()
            raise
        output.finish()
        elapsed = time.perf_counter() - step_started
        if will_run:
            ran += 1; outcome = f"reran ({elapsed:.2f}s)"
            typer.secho(f"  {outcome}", fg="bright_green")
        else:
            skipped += 1; outcome = f"current — skipped ({elapsed:.2f}s)"
            typer.secho(f"  {outcome}", fg="bright_green")
        if run_log:
            run_log.write(f"\n[iv] {outcome}\n"); run_log.flush()
    summary = f"complete in {time.perf_counter() - started:.2f}s · {ran} reran, {skipped} current — skipped"
    typer.secho(f"\n{summary}", bold=True)
    if run_log:
        run_log.write(f"\n{summary}\n"); run_log.close()


def _trial_outputs(iv, asset, part: dict | None, out_tree: Path) -> dict[str, dict[str, str]]:
    """Force one stage into an isolated output tree and return content fingerprints."""
    original_out, original_force = iv.out_tree, iv.force
    try:
        iv.out_tree, iv.force = out_tree, True
        asset(**part) if part is not None else asset()
        return {
            output.dataset: {
                key: shard.fp for key, shard in _sh.current_shards(iv.resolve_out(output.dataset)).items()
            }
            for output in asset.outputs.values()
        }
    finally:
        iv.out_tree, iv.force = original_out, original_force


def _determinism_differences(first, second) -> list[str]:
    differences = []
    for dataset in sorted(set(first) | set(second)):
        left, right = first.get(dataset, {}), second.get(dataset, {})
        for part in sorted(set(left) | set(right), key=_sh.sort_key):
            if left.get(part) != right.get(part):
                label = part or "(one shard)"
                differences.append(
                    f"{dataset}{label}: {left.get(part, '(absent)')} != {right.get(part, '(absent)')}")
    return differences


def _audit_determinism(iv, node: str, asset, parts: list[dict | None]) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="iv-determinism-") as root:
        first = [_trial_outputs(iv, asset, p, Path(root) / "first") for p in parts]
        second = [_trial_outputs(iv, asset, p, Path(root) / "second") for p in parts]
    differences = []
    for p, left, right in zip(parts, first, second):
        prefix = f"{node}{_part_label(p)} — "
        differences += [prefix + line for line in _determinism_differences(left, right)]
    return differences


def _last_part(parts: list[dict | None]) -> dict | None:
    return max(parts, key=lambda p: _sh.sort_key(_sh.encode_part(p)))


@app.command()
def graph(focus: str = typer.Option(None, "--focus", help="only this stage and its cone")):
    g = _graph_of()
    order, parents = g.order(), g.parent_map()
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
@reports
def impact(stage: str, tick: bool = typer.Option(False, "--tick",
                                                   help="assume this stage's output changes"),
           tick_part: list[str] = typer.Option(
               [], "--tick-part", help="output partition to tick, repeat as key=value")):

    if tick_part and not tick:
        raise IvError("--tick-part requires --tick.")
    iv = _load()
    g = _graph.build(iv)
    node = _stage_name(g, stage)
    filters = _part_flags(tick_part, "--tick-part")
    parents = g.parent_map()
    upstream = _render.ancestors_of(node, parents)
    downstream = _render.descendants_of(node, parents)
    order = g.order()
    with _sh.snapshot():
        state = _staleness(iv, g)
        maybe = _downstream_of(g, state)

        typer.secho(node, bold=True)
        _print_impact_list("upstream", [n for n in order if n in upstream], g, state, maybe)
        _print_impact_list("this stage", [node], g, state, maybe)
        _print_impact_list("downstream", [n for n in order if n in downstream], g, state, maybe)

        if not tick:
            return

        moving = _stage_output_shards(g, node, state, filters)
        if not moving:
            available = sorted(
                p or "(one shard)"
                for _, p in _stage_output_shards(g, node, state)
            )
            wanted = ", ".join(f"{k}={v}" for k, v in sorted(filters.items()))
            detail = f" matching {wanted}" if wanted else ""
            have = f" Available: {', '.join(available)}." if available else ""
            raise IvError(f"{node} has no output shards on disk{detail} to tick.{have}")
        possible = _downstream_of(g, state, moving=moving)
        affected: dict[str, list[tuple[str, str]]] = {}
        for other in order:
            if other not in downstream:
                continue
            outputs = {s.dataset for s in g.stages[other].outputs}
            shards = sorted((ds, p) for ds, p in possible if ds in outputs)
            if shards:
                affected[other] = shards
        typer.echo()
        label = _part_label(filters)
        typer.secho(f"if {node}{label} changes", bold=True)
        typer.secho(f"  will run  {node}{label}", fg="yellow")
        if affected:
            for other, shards in affected.items():
                typer.secho(f"  may rebuild {other}", fg="bright_blue")
                for dataset, part in shards:
                    typer.echo(f"    {dataset}{part or '(one shard)'}")
        else:
            typer.echo("  no downstream stage reads its current output shards")


@app.command()
def preflight():

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
    with iv.bookkeeping():
        for name in sorted(iv._sources):
            if not _sh.current_shards(iv.resolve(name)):
                bad.append(f"EMPTY SOURCE  {name} is declared but nothing is there")
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
        plain: bool = typer.Option(False, "--plain",
                                   help="do not read the tree; leave every node grey"),
        html: bool = typer.Option(False, "--html",
                                  help="an interactive page instead of a picture")):


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
                         title=iv.tree.name)
        typer.echo(f"wrote {got} — open it")
        return
    typer.echo(f"wrote {_viz.draw(g, out, status=status)}")


def _declared_part_keys(iv, g, name: str) -> set[tuple[str, ...]]:


    source = iv._sources.get(name)
    out = ({tuple(sorted(source.part_keys))}
           if source is not None and source.part_keys is not None else set())
    for node, stage in g.stages.items():
        if not any(site.dataset == name for site in stage.outputs):
            continue
        asset = iv._assets.get(node)
        if asset is None:
            continue
        out.add(tuple(sorted(asset.fixed_part)) if asset.fixed_part
                else tuple(sorted(asset.part_keys or ())))
    return out


def _staleness(iv, g):


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

    named = sorted(parts, key=_sh.sort_key)
    if total == 1:
        return named[0] or "(one shard)"
    if len(named) == total:
        return f"all {total}"
    shown = ", ".join(named[:3]) if len(named) <= 3 else f"{named[0]}..{named[-1]}"
    return f"{len(named)}/{total} ({shown})"


def _line(shards: dict, maybe: set, dataset: str) -> tuple:


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


def _downstream_of(g, state: dict, iv=None, moving=None) -> set:


    from .core import _resolve_sel
    seed = _stale_shards(state) if moving is None else set(moving)
    moving = set(seed)
    for node in g.order():
        st = g.stages[node]
        asset = getattr(g.iv, "_assets", {}).get(node)
        for site in st.outputs:
            fixed = dict(site.part) or None
            for p in state.get(site.dataset, {""}):
                shard_part = _sh.decode_part(p)
                if not _site_owns(g, node, site, shard_part):
                    continue
                if (site.dataset, p) in moving:
                    continue
                part = shard_part or fixed


                if part is None and asset is not None and asset.part_keys:
                    continue
                for t in st.triggers:
                    where = _resolve_sel(t.sel, part, t.dataset)
                    if any(_sh.covers(where, _sh.decode_part(q) or {})
                           for (d, q) in moving if d == t.dataset):
                        moving.add((site.dataset, p))
                        break
    return moving - seed


def _part_matches(part: dict[str, str], filters: dict[str, str]) -> bool:

    return all(part.get(k) == v for k, v in filters.items())


def _site_owns(g, node: str, site, part: dict[str, str]) -> bool:

    fixed = dict(site.part)
    if fixed:
        return _part_matches(part, fixed)
    return not any(
        other != node and candidate.dataset == site.dataset and candidate.part
        and _part_matches(part, dict(candidate.part))
        for other, stage in g.stages.items()
        for candidate in stage.outputs
    )


def _stage_output_shards(g, node: str, state: dict,
                         filters: dict[str, str] | None = None) -> set[tuple[str, str]]:

    filters = filters or {}
    out = set()
    for site in g.stages[node].outputs:
        for part_str in state.get(site.dataset, {}):
            part = _sh.decode_part(part_str)
            if not _site_owns(g, node, site, part):
                continue
            if _part_matches(part, filters):
                out.add((site.dataset, part_str))
    return out


def _owner(owners, part) -> tuple:


    if part:
        for fixed, inputs in owners:
            if fixed and all(str(part.get(k)) == v for k, v in fixed):
                return inputs
    for fixed, inputs in owners:
        if not fixed:
            return inputs
    return owners[0][1]


def _stage_state(node: str, g, state: dict, maybe: set) -> str:
    datasets = {s.dataset for s in g.stages[node].outputs}
    shards = [(d, p, why) for d in datasets for p, why in state.get(d, {}).items()]
    if any(why for _, _, why in shards):
        return "stale"
    if any((d, p) in maybe for d, p, _ in shards):
        return "maybe"
    return "current" if shards else "action"


def _print_impact_list(title: str, nodes: list[str], g, state, maybe, *, target=None) -> None:
    typer.secho(title, bold=True)
    if not nodes:
        typer.echo("  (none)")
        return
    colors = {"current": "bright_green", "maybe": "bright_blue", "stale": "yellow", "action": "bright_black"}
    for node in nodes:
        status = "will run" if node == target else _stage_state(node, g, state, maybe)
        typer.secho(f"  {status:<9} {node}", fg=colors.get(status, "yellow"))


@app.command()
@reports
def status():
    iv = _load()
    g = _graph_of()


    with _sh.snapshot():
        state = _staleness(iv, g)
        maybe = _downstream_of(g, state)

    tally = {"current": 0, "maybe": 0, "stale": 0}
    for name, shards in state.items():
        tier, note = _line(shards, maybe, name)
        typer.secho(f"  {tier:<8} {name:<44} {note}",
                    fg={"current": "bright_green", "maybe": "bright_blue", "stale": "yellow"}[tier])
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
                typer.secho("  current", fg="bright_green")


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
            typer.secho(f"  maybe    {name:<44} {note}", fg="bright_blue")


@app.command()
@reports
def determinism(
    only: str = typer.Option(None, "--only", help="stage to run twice in isolated output trees"),
    sample: bool = typer.Option(False, "--sample", help="audit every stage at its last declared partition"),
    part: list[str] = typer.Option([], "--part", help="partition filter, repeat as key=value"),
):
    """Check that selected stages produce identical content from identical inputs."""
    if bool(only) == sample:
        raise IvError("choose exactly one of --only STAGE or --sample.")
    if sample and part:
        raise IvError("--part is only meaningful with `iv determinism --only STAGE`.")
    iv = _load()
    g = _graph.build(iv)
    if only:
        node = _stage_name(g, only)
        asset = iv._assets[node]
        if asset.acts_only:
            raise IvError(f"{node} is an action with no output, so determinism cannot be measured.")
        parts = _asset_parts(iv, asset, _part_flags(part))
        differences = _audit_determinism(iv, node, asset, parts)
        if differences:
            typer.secho("not deterministic", fg="red", bold=True)
            for line in differences:
                typer.echo(f"  {line}")
            raise typer.Exit(1)
        typer.secho(f"deterministic · {node} · {len(parts)} stage shard(s) matched", fg="green")
        return

    failed = checked = skipped = 0
    typer.secho("determinism sample", bold=True)
    for node in g.order():
        asset = iv._assets[node]
        if asset.acts_only:
            skipped += 1
            typer.secho(f"  skipped  {node} — action", fg="bright_black")
            continue
        chosen = _last_part(_asset_parts(iv, asset, {}))
        differences = _audit_determinism(iv, node, asset, [chosen])
        checked += 1
        if differences:
            failed += 1
            typer.secho(f"  failed   {node}{_part_label(chosen)}", fg="red")
            for line in differences:
                typer.echo(f"    {line}")
        else:
            typer.secho(f"  ok       {node}{_part_label(chosen)}", fg="green")
    summary = f"{checked} checked, {failed} failed, {skipped} action(s) skipped"
    if failed:
        typer.secho(f"not deterministic · {summary}", fg="red", bold=True)
        raise typer.Exit(1)
    typer.secho(f"deterministic · {summary}", fg="green", bold=True)


@app.command()
@reports
def run(
    up_to: str = typer.Option(None, "--up-to", help="run this stage and everything it needs"),
    up_to_excluding: str = typer.Option(None, "--up-to-excluding", help="run prerequisites, not this stage"),
    from_: str = typer.Option(None, "--from", help="run this stage and descendants; upstream must be current"),
    only: str = typer.Option(None, "--only", help="run exactly this stage; upstream must be current"),
    part: list[str] = typer.Option([], "--part", help="partition filter, repeat as key=value"),
    force: bool = typer.Option(
        False, "--force",
        help="rebuild the selected stages even when current, and let --only/--from "
             "run over a stale upstream"),
    log: Path = typer.Option(
        None, "--log",
        help="write all merged stage output and outcomes to this file"),
    dev: Path = typer.Option(
        None, "--dev",
        help="write --only output to this local tree instead of production"),
):

    iv = _load()
    with _dev_output(iv, dev):
        with _paths.local_tree_snapshot(iv, report=typer.echo):
            _run_local(iv, up_to, up_to_excluding, from_, only, part, force, log)


@contextmanager
def _dev_output(iv, dev: Path | None):
    if dev is None:
        yield
        return
    target = dev.expanduser().resolve()
    remote_tree = _paths.is_remote(iv.tree)
    remote_out = _paths.is_remote(iv.out_tree)
    if remote_tree and remote_out and str(iv.tree) != str(iv.out_tree):
        raise IvError("--dev cannot clone separate remote input and output trees into one "
                      "directory; use a pipeline with one remote tree.")
    if not remote_out:
        production = Path(iv.out_tree).expanduser().resolve()
        if target == production:
            raise IvError("--dev must differ from the production output tree.")
    original_tree, original_out = iv.tree, iv.out_tree
    if remote_tree or remote_out:
        source = iv.tree if remote_tree else iv.out_tree
        _paths.fetch_tree(source, target, report=typer.echo, replace=True)
        iv.tree = target
    iv.out_tree = target
    typer.echo(f"development output · {target}")
    try:
        yield
    finally:
        iv.tree, iv.out_tree = original_tree, original_out


def _run_local(iv, up_to, up_to_excluding, from_, only, part, force, log):

    choices = [x for x in (up_to, up_to_excluding, from_, only) if x]
    if len(choices) > 1:
        raise IvError("choose only one of --up-to, --up-to-excluding, --from, or --only.")
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
    if force:
        iv.force = True
        safe = False

    with _sh.snapshot():
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
                raise IvError("refusing to run with stale upstream shard(s): "
                              + "; ".join(stale) + ". Run `iv run --up-to "
                              + choices[0] + "` first.")

        work = [(node, iv._assets[node], p)
                for node in g.order() if node in selected
                for p in _asset_parts(iv, iv._assets[node], filters)]
        if not work:
            typer.secho("nothing selected", fg="yellow")
            return

        _execute_work(iv, work, log)


@app.command()
@reports
def fetch(
    destination: Path = typer.Argument(..., help="local directory for remote state"),
    replace: bool = typer.Option(
        False, "--replace", help="replace an existing directory after download succeeds"),
):
    iv = _load()
    _paths.fetch_tree(iv.tree, destination, report=typer.echo, replace=replace)


@app.command()
@reports
def verify(dataset: str = typer.Argument(None, help="one dataset, or all of them")):

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


def _orphan_datasets(iv, g) -> list[tuple[str, object]]:


    known = set(g.produced) | set(iv._sources) | set(iv._datasets)
    root = iv.out_tree
    dirs = {}
    with iv.bookkeeping():
        if not root.exists():
            return []
        for path in root.rglob("*"):
            if _sh.parse_name(path) is None:
                continue
            d = path.parent
            dirs[str(d)] = d
    out = []
    for d in dirs.values():
        name = _canon(str(d)[len(str(root)):].strip("/"))
        if name not in known:
            out.append((name, d))
    return sorted(out)


@app.command()
@reports
def gc(
    dataset: str = typer.Argument(None, help="one dataset, or all of them"),
    partition_key: list[str] = typer.Option(
        [], "--partition-key", help="expected shard key; repeat for a composite partition"),
):
    if partition_key and dataset is None:
        raise IvError("--partition-key needs one DATASET so the repair scope is explicit.")
    if len(set(partition_key)) != len(partition_key):
        raise IvError("--partition-key names the same key more than once.")
    iv = _load()
    g = _graph_of()
    targets = [dataset] if dataset else g.produced
    total = 0
    stack = iv.bookkeeping()
    stack.__enter__()
    for name in targets:
        d = iv.resolve_out(name)
        found = _sh.list_shards(d) if d.exists() else {}
        if not found:
            continue
        want = ({tuple(sorted(partition_key))} if partition_key
                else _declared_part_keys(iv, g, name))
        live = {p: v for p, v in found.items()
                if not want or tuple(sorted(_sh.decode_part(p))) in want}
        orphaned = sorted(set(found) - set(live))
        if not orphaned and not any(len(v) > 1 for v in live.values()):
            continue
        for p in orphaned:
            typer.secho(f"  {name}: orphaned partition {p or '(no part)'} — the "
                        f"stage does not key on {tuple(sorted(_sh.decode_part(p)))} "
                        f"any more", fg="yellow")
        keep = {sorted(v, key=lambda s: s.name)[0].name for v in live.values()}
        for gone in _sh.gc(d, keep=keep):
            typer.echo(f"  dropped {name}{gone}")
            total += 1
    if dataset is None:
        for name, d in _orphan_datasets(iv, g):
            typer.secho(f"  {name}: no stage produces this and nothing declares it — "
                        f"dropping the whole dataset", fg="yellow")
            unknown = []
            dropped = False
            for path in sorted(d.iterdir()):
                if _sh.parse_name(path) is None:
                    unknown.append(path.name)
                    continue
                path.unlink()
                typer.echo(f"  dropped {name}{path.name}")
                total += 1
                dropped = True
            if dropped:
                _sh._cache_drop(d)
            if unknown:
                typer.secho(f"  {name}: left {len(unknown)} file(s) iv did not write "
                            f"(e.g. {unknown[0]}) — delete them yourself if they are "
                            f"dead too", fg="yellow")
            elif isinstance(d, Path) and not any(d.iterdir()):
                d.rmdir()
    stack.__exit__(None, None, None)
    typer.echo(f"{total} shard(s) dropped")
