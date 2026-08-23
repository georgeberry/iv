"""The DAG as a page you can click, in one self-contained file.

`iv viz` draws a PNG, which answers "what is the shape of this" and nothing else. The
questions that actually come up in front of a stale pipeline are about ONE node —

    what is this for, and what is stale about it
    what does it read, and what reads it
    if I rebuild it, what follows

— and a picture cannot answer them because it has no place to put the answer. On a 60-node
graph the labels alone are at the edge of legible, and the reasons never fit at all.

So this emits the same graph as DATA and lets the browser do the layout and the asking.
Python computes nothing new: the nodes are `viz.to_networkx`'s, the colours are
`viz.STATUS`, the staleness is the CLI's, per shard, with the reasons it already prints.
What is new is only that a click can select a node and something can be shown about it.

WHY THE LAYOUT IS NOT DONE HERE. Upstream and downstream are the whole question, and in a
graph library they are one call — `predecessors()` and `successors()` — against a layout
that already puts an edge's ends on either side of it. Recomputing that in matplotlib means
reimplementing layered layout, edge routing and hit-testing to arrive somewhere worse. The
page is a template with a JSON blob in it, and everything interactive is the graph library's
own.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import shards as _sh
from . import viz as _viz

#: Pulled at open time. Vendoring them would make this a 1.2MB file per graph; both are
#: pinned to an exact version so a page that worked keeps working.
CDN = (
    "https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js",
    "https://unpkg.com/dagre@0.8.5/dist/dagre.min.js",
    "https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js",
)


def payload(g, status: dict | None = None, state: dict | None = None,
            maybe: set | None = None) -> dict:
    """Everything the page shows, as JSON. No presentation, no layout — those are the
    template's, so what is asserted about this can be asserted about a dict."""
    d = _viz.to_networkx(g)
    status = status or {}
    state = state or {}
    maybe = maybe or set()
    iv = g.iv

    stages = {}
    for name, st in g.stages.items():
        a = getattr(iv, "_assets", {}).get(name)
        stages[name] = {
            "why": getattr(a, "why", "") if a else "",
            "part": getattr(a, "part_key", None) if a else None,
            "once": bool(getattr(a, "once", False)) if a else False,
            "split": bool(getattr(a, "split", False)) if a else False,
            "externals": [list(e) for e in getattr(a, "externals", ())] if a else [],
        }

    nodes = []
    for n in d.nodes:
        ds, part = n
        # A dataset several stages write is one node PER PARTITION, and each node owns only
        # the shards of its own. Handing every node the whole dataset's shards would count
        # the shared ones once per writer, so the page and `iv status` would disagree about
        # how many shards there are — and the played half would list the unplayed half's
        # staleness as its own.
        shards = {k: v for k, v in state.get(ds, {}).items() if _owns(part, k)}
        writers = g.producers_of(ds)
        src = getattr(iv, "_sources", {}).get(ds)
        decl = getattr(iv, "_datasets", {}).get(ds)
        nodes.append({
            "id": _nid(n),
            "label": _viz.short(n),
            "dataset": ds,
            "part": dict(part),
            "kind": d.nodes[n]["kind"],
            # A root has no stage to ask, so it is not "current", it is grey — the same
            # thing `iv status` does by not listing it at all.
            "status": status.get(ds, "source" if d.nodes[n]["kind"] == "root" else "current"),
            "why": (getattr(src, "why", None) or getattr(decl, "why", None)
                    or (stages[writers[0]]["why"] if writers else "")),
            "declared": "source" if src else ("dataset" if decl else "inline"),
            "writers": writers,
            "shards": [{"part": p or "(one shard)", "reason": why,
                        "maybe": (ds, p) in maybe}
                       for p, why in sorted(shards.items())],
            "externals": [e for w in writers for e in stages[w]["externals"]],
        })

    edges = [{"source": _nid(u), "target": _nid(v), "stage": d.edges[u, v].get("stage", "")}
             for u, v in d.edges]
    return {"nodes": nodes, "edges": edges, "stages": stages,
            "counts": _counts(nodes)}


def _owns(part: tuple, shard_key: str) -> bool:
    """Is this shard the one this node writes? Every shard, where the node is the dataset."""
    if not part:
        return True
    got = _sh.decode_part(shard_key) or {}
    return all(got.get(k) == v for k, v in part)


def _counts(nodes) -> dict:
    c = {"current": 0, "maybe": 0, "stale": 0, "source": 0}
    for n in nodes:
        c[n["status"]] = c.get(n["status"], 0) + 1
    return c


def _nid(node) -> str:
    ds, part = node
    return ds + ("|" + ",".join(f"{k}={v}" for k, v in part) if part else "")


def write(g, out: Path, status: dict | None = None, state: dict | None = None,
          maybe: set | None = None, title: str = "iv") -> Path:
    out = Path(out)
    out.write_text(_PAGE.replace("__DATA__", json.dumps(
        payload(g, status, state, maybe), indent=None))
        .replace("__TITLE__", title)
        .replace("__COLOURS__", json.dumps(_viz.STATUS))
        .replace("__SCRIPTS__", "\n".join(
            f'<script src="{u}"></script>' for u in CDN)))
    return out


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ — the graph</title>
__SCRIPTS__
<style>
:root {
  --bg:#fbfbfa; --panel:#fff; --ink:#1a1a1a; --dim:#6b6b6b; --line:#e3e1dd;
  --sel:#1a1a1a; --up:#5b8def; --down:#c2410c;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#16171a; --panel:#1e2024; --ink:#e8e6e3; --dim:#9a9a9a; --line:#2e3136;
  --sel:#e8e6e3; --up:#7ba3f5; --down:#f97316;}}
*{box-sizing:border-box}
body{margin:0;height:100vh;display:flex;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
#cy{flex:1;min-width:0}
#side{width:380px;border-left:1px solid var(--line);background:var(--panel);
  overflow-y:auto;padding:18px 20px}
h1{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);
  margin:0 0 14px;font-weight:600}
h2{font-size:17px;margin:0 0 2px;word-break:break-all;font-weight:600}
.why{color:var(--dim);margin:0 0 16px;font-size:13px}
.pill{display:inline-block;padding:2px 9px;border-radius:99px;font-size:11px;
  font-weight:600;letter-spacing:.03em;color:#fff}
.sec{margin:18px 0 6px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--dim);font-weight:600}
ul{margin:0;padding:0;list-style:none}
li{padding:5px 0;border-bottom:1px solid var(--line);font-size:13px}
li:last-child{border-bottom:0}
a{color:var(--up);cursor:pointer;text-decoration:none;word-break:break-all}
a:hover{text-decoration:underline}
.reason{color:var(--down);font-size:12px}
.ok{color:var(--dim);font-size:12px}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--bg);
  padding:1px 5px;border-radius:4px}
#bar{position:absolute;top:14px;left:16px;display:flex;gap:8px;align-items:center;
  flex-wrap:wrap;max-width:60%}
#q{padding:6px 11px;border:1px solid var(--line);border-radius:7px;width:210px;
  background:var(--panel);color:var(--ink);font-size:13px}
.key{font-size:11px;color:var(--dim);display:flex;align-items:center;gap:5px;
  background:var(--panel);padding:4px 9px;border-radius:99px;border:1px solid var(--line)}
.dot{width:9px;height:9px;border-radius:99px;display:inline-block}
.hint{color:var(--dim);font-size:13px}
</style></head><body>
<div id="cy"></div>
<div id="bar">
  <input id="q" placeholder="filter datasets…" autocomplete="off">
  <span class="key" id="legend"></span>
</div>
<div id="side"><h1>__TITLE__</h1><p class="hint" id="side-body">
Click a node. Its <b style="color:var(--up)">upstreams</b> and
<b style="color:var(--down)">downstreams</b> light up — everything it reads from, however
far back, and everything a rebuild would carry forward.</p></div>
<script>
const DATA = __DATA__, C = __COLOURS__;
const byId = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));

document.getElementById('legend').innerHTML = ['stale','maybe','current','source']
  .map(s => `<span class="dot" style="background:${C[s]}"></span>${s} ${DATA.counts[s]||0}`)
  .join('&nbsp;&nbsp;');

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: [
    ...DATA.nodes.map(n => ({data: n})),
    ...DATA.edges.map((e,i) => ({data: {...e, id: 'e'+i}})),
  ],
  style: [
    {selector:'node', style:{
      'background-color': n => C[n.data('status')] || C.source,
      'label':'data(label)', 'font-size':11, 'color':'var(--ink)',
      'text-valign':'center','text-halign':'right','text-margin-x':6,
      'width':16,'height':16,'border-width':0,
      // SHAPE is kind, COLOUR is status — the same two channels the PNG uses, so a
      // reader of one is not learning a second vocabulary for the other.
      'shape': n => ({root:'square', terminal:'diamond'})[n.data('kind')] || 'ellipse',
    }},
    {selector:'edge', style:{
      'width':1.2,'line-color':'var(--line)','target-arrow-color':'var(--line)',
      'target-arrow-shape':'triangle','arrow-scale':.8,'curve-style':'bezier',
    }},
    {selector:'.faded', style:{'opacity':.12,'text-opacity':0}},
    {selector:'node.sel', style:{'border-width':3,'border-color':'var(--sel)',
      'width':22,'height':22,'font-weight':'bold','z-index':99}},
    {selector:'node.up', style:{'border-width':2,'border-color':'var(--up)'}},
    {selector:'node.down', style:{'border-width':2,'border-color':'var(--down)'}},
    {selector:'edge.up', style:{'line-color':'var(--up)','target-arrow-color':'var(--up)','width':2}},
    {selector:'edge.down', style:{'line-color':'var(--down)','target-arrow-color':'var(--down)','width':2}},
  ],
  layout: {name:'dagre', rankDir:'LR', nodeSep:14, rankSep:110, edgeSep:6},
  wheelSensitivity:.2,
});

function esc(s){const d=document.createElement('div');d.textContent=s==null?'':s;return d.innerHTML}

function select(node, fit){
  cy.elements().removeClass('faded up down sel');
  // The two questions, and the graph library answers both outright: everything that can
  // reach this node, and everything this node can reach.
  const up = node.predecessors(), down = node.successors();
  cy.elements().difference(up.union(down).union(node)).addClass('faded');
  up.addClass('up'); down.addClass('down'); node.addClass('sel');
  if (fit) cy.animate({fit:{eles:up.union(down).union(node), padding:60}, duration:250});
  panel(node, up.nodes(), down.nodes());
}

function panel(node, up, down){
  const d = node.data();
  const nb = dir => cy.edges().filter(e => e[dir==='in'?'target':'source']().id()===d.id)
      .map(e => e[dir==='in'?'source':'target']().id());
  const list = ids => ids.length
    ? `<ul>${[...new Set(ids)].sort().map(i =>
        `<li><a onclick="pick('${esc(i)}')">${esc(byId[i].label)}</a>
         <span class="dot" style="background:${C[byId[i].status]||C.source};
         margin-left:6px"></span></li>`).join('')}</ul>`
    : '<p class="ok">none</p>';

  const shards = d.shards.length ? `<ul>${d.shards.map(s =>
      `<li><code>${esc(s.part)}</code><br>${
        s.reason ? `<span class="reason">${esc(s.reason)}</span>`
        : s.maybe ? '<span class="ok">may follow — an upstream is stale</span>'
        : '<span class="ok">current</span>'}</li>`).join('')}</ul>`
    : '<p class="ok">nothing on disk yet</p>';

  document.getElementById('side-body').outerHTML = `<div id="side-body">
    <h2>${esc(d.dataset)}</h2>
    <p class="why">${esc(d.why)}</p>
    <span class="pill" style="background:${C[d.status]||C.source}">${esc(d.status)}</span>
    <span class="pill" style="background:var(--dim)">${esc(d.kind)}</span>
    ${d.writers.length ? `<div class="sec">built by</div><ul>${d.writers.map(w =>
        `<li><code>${esc(w.split('::').pop())}</code>
         <div class="ok">${esc((DATA.stages[w]||{}).why||'')}</div></li>`).join('')}</ul>`
      : '<div class="sec">arrives from outside</div><p class="ok">nothing here writes it</p>'}
    ${d.externals.length ? `<div class="sec">outside this tree</div><ul>${
        d.externals.map(e => `<li><code>${esc(e[0])}</code>
        <div class="ok">${esc(e[1])}</div></li>`).join('')}</ul>` : ''}
    <div class="sec">shards — ${d.shards.filter(s=>s.reason).length} stale of ${d.shards.length}</div>
    ${shards}
    <div class="sec">reads directly (${nb('in').length})</div>${list(nb('in'))}
    <div class="sec">read directly by (${nb('out').length})</div>${list(nb('out'))}
    <div class="sec">upstream in all (${up.length}) · downstream in all (${down.length})</div>
    <p class="ok">a rebuild of this carries into ${down.length} node(s).</p>
  </div>`;
}

window.pick = id => select(cy.$id(id), true);
cy.on('tap','node', e => select(e.target, false));
cy.on('tap', e => { if (e.target === cy) {
  cy.elements().removeClass('faded up down sel');
  document.getElementById('side-body').outerHTML =
    '<p class="hint" id="side-body">Click a node.</p>';
}});

document.getElementById('q').addEventListener('input', e => {
  const q = e.target.value.trim().toLowerCase();
  cy.nodes().forEach(n => n.toggleClass('faded',
    !!q && !n.data('dataset').toLowerCase().includes(q)));
});
</script></body></html>
"""
