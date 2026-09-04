

from __future__ import annotations

import json
from pathlib import Path

from . import shards as _sh
from . import viz as _viz


CDN = ("https://unpkg.com/cytoscape@3.28.1/dist/cytoscape.min.js",)


FONT, MARKER = 14, 26.0
ROW, CHAR, GAP = 52.0, FONT * 0.56, 30.0


def payload(g, status: dict | None = None, state: dict | None = None,
            maybe: set | None = None) -> dict:


    whole = _viz.to_networkx(g)


    d = whole
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


        shards = {k: v for k, v in state.get(ds, {}).items() if _owns(part, k)}
        writers = g.producers_of(ds)
        src = getattr(iv, "_sources", {}).get(ds)
        decl = getattr(iv, "_datasets", {}).get(ds)
        partition_keys = sorted({key for shape in iv._expected_part_keys(ds) for key in shape})
        nodes.append({
            "id": _nid(n),
            "label": _viz.short(n),
            "dataset": ds,
            "part": dict(part),
            "kind": d.nodes[n]["kind"],


            "status": status.get(ds, "source" if d.nodes[n]["kind"] == "root" else "current"),
            "why": (getattr(src, "why", None) or getattr(decl, "why", None)
                    or (stages[writers[0]]["why"] if writers else "")),
            "declared": "source" if src else ("dataset" if decl else "inline"),
            "partitionKeys": partition_keys,
            "writers": writers,
            "shards": [{"part": p or "(one shard)", "reason": why,
                        "maybe": (ds, p) in maybe}
                       for p, why in sorted(shards.items())],
            "externals": [e for w in writers for e in stages[w]["externals"]],


            "reads": sorted({_nid(m) for m in whole.predecessors(n)}),
            "readBy": sorted({_nid(m) for m in whole.successors(n)}),
        })

    edges = [{"source": _nid(u), "target": _nid(v), "stage": d.edges[u, v].get("stage", ""),
              "rule": d.edges[u, v].get("rule", "all_of"),
              "optional": d.edges[u, v].get("optional", False)}
             for u, v in d.edges]
    at = _positions(d)
    for n in nodes:
        n["position"] = at[n["id"]]
    partition_keys = sorted({key for node in nodes for key in node["partitionKeys"]})
    return {"nodes": nodes, "edges": edges, "stages": stages,
            "counts": _counts(nodes), "partitionKeys": partition_keys}


def _positions(d) -> dict:


    depth = _viz._layers(d)
    cols: dict[int, list] = {}
    for n, k in sorted(depth.items(), key=lambda kv: (kv[1], _viz.short(kv[0]))):
        cols.setdefault(k, []).append(n)

    order = {n: i for c in cols.values() for i, n in enumerate(c)}
    for sweep in range(8):
        keys = sorted(cols)
        for k in (keys if sweep % 2 == 0 else keys[::-1]):
            near = (d.predecessors if sweep % 2 == 0 else d.successors)
            def bary(n):
                ns = [order[m] for m in near(n) if m in order]
                return sum(ns) / len(ns) if ns else order[n]
            cols[k] = sorted(cols[k], key=bary)
            order.update({n: i for i, n in enumerate(cols[k])})

    pos, x = {}, 0.0
    for k in sorted(cols):


        width = MARKER + CHAR * max(len(_viz.short(n)) for n in cols[k]) + GAP
        for i, n in enumerate(cols[k]):
            pos[_nid(n)] = {"x": round(x, 1),
                            "y": round((i - (len(cols[k]) - 1) / 2) * ROW, 1)}
        x += width
    return pos


def _owns(part: tuple, shard_key: str) -> bool:

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
  --sel:#1a1a1a; --up:#5b8def; --down:#c2410c; --edge:#7a7a7a;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#16171a; --panel:#1e2024; --ink:#e8e6e3; --dim:#9a9a9a; --line:#2e3136;
  --sel:#e8e6e3; --up:#7ba3f5; --down:#f97316; --edge:#8f9298;}}
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
#partition-key{font-size:11px;color:var(--dim);background:var(--panel);padding:5px 9px;
  border-radius:7px;border:1px solid var(--line);cursor:pointer}
.dot{width:9px;height:9px;border-radius:99px;display:inline-block}
.edge-sample{display:inline-block;width:22px;border-top:2px solid var(--edge);vertical-align:middle}
.edge-sample.same_part{border-top-style:dashed}
.edge-sample.before_part,.edge-sample.before_part_inclusive{border-top-style:dotted}
.edge-sample.after_part,.edge-sample.after_part_inclusive{border-top-style:dashed;border-top-width:3px}
.edge-sample.parts,.edge-sample.between{border-top-style:dashed;border-top-width:1px}
.edge-rule{border:0;background:none;color:var(--dim);font:inherit;font-size:11px;padding:0;cursor:pointer}
.edge-rule:hover{color:var(--ink)}
.edge-rule[aria-pressed="true"]{color:var(--ink);font-weight:700}
.hint{color:var(--dim);font-size:13px}
</style></head><body>
<div id="cy"></div>
<div id="bar">
  <input id="q" placeholder="filter datasets…" autocomplete="off">
  <select id="partition-key" aria-label="Select a partition key"></select>
  <span class="key" id="edge-legend"></span>
  <span class="key" id="legend"></span>
  <span class="key" id="reset" style="cursor:pointer">reset view</span>
</div>
<div id="side"><h1>__TITLE__</h1><p class="hint" id="side-body">
Click a node. Its <b style="color:var(--up)">upstreams</b> and
<b style="color:var(--down)">downstreams</b> light up — everything it reads from, however
far back, and everything a rebuild would carry forward.</p></div>
<script>
const DATA = __DATA__, C = __COLOURS__;
// Every node arrives with a position, so there is nothing to lay out — see `_positions`.
const LAYOUT = {name:'preset', fit:false};

// A pipeline is WIDE — nine columns deep and five rows tall here, far more there — so
// fitting the whole thing into the viewport sets the zoom from the width and the type ends
// up small however large it was drawn. Making the markers and the font bigger did nothing
// for exactly this reason: fit cancelled it.
//
// So: fit, but not below the point where a label stops being readable. Under that floor
// the graph starts at its ROOTS, which is the end you read from, and you pan right.
const FLOOR = 0.8;
function land(){
  cy.fit(undefined, 36);
  if (cy.zoom() >= FLOOR) return;
  cy.zoom(FLOOR);
  const b = cy.elements().boundingBox(), h = cy.height();
  cy.pan({x: 36 - b.x1 * FLOOR, y: h / 2 - (b.y1 + b.h / 2) * FLOOR});
}
const byId = Object.fromEntries(DATA.nodes.map(n => [n.id, n]));

const partitionSelect = document.getElementById('partition-key');
partitionSelect.innerHTML = DATA.partitionKeys.length
  ? `<option value="">partition key: all</option><option value="__none__">partition key: none</option>${DATA.partitionKeys.map(k =>
      `<option value="${esc(k)}">partition key: ${esc(k)}</option>`).join('')}`
  : '<option value="">partition key: all</option><option value="__none__">partition key: none</option>';

document.getElementById('legend').innerHTML = ['stale','maybe','current','source']
  .map(s => `<span class="dot" style="background:${C[s]}"></span>${s} ${DATA.counts[s]||0}`)
  .join('&nbsp;&nbsp;');

const RULES = [
  ['all_of', 'all of'], ['same_part', 'same part'],
  ['before_part', 'before'], ['before_part_inclusive', 'through'],
  ['after_part', 'after'], ['after_part_inclusive', 'from'],
  ['parts', 'selected'], ['between', 'range'],
];
document.getElementById('edge-legend').innerHTML = RULES.map(([rule, label]) =>
  `<button class="edge-rule" data-rule="${rule}" aria-pressed="false">
     <span class="edge-sample ${rule}"></span> ${label}</button>`).join('&nbsp;&nbsp;');

const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: [
    ...DATA.nodes.map(n => ({data: n, position: {...n.position}})),
    ...DATA.edges.map((e,i) => ({data: {...e, id: 'e'+i}})),
  ],
  style: [
    {selector:'node', style:{
      'background-color': n => C[n.data('status')] || C.source,
      'label':'data(label)', 'font-size':14, 'color':'var(--ink)',
      'text-valign':'center','text-halign':'right','text-margin-x':9,
      'width':26,'height':26,'border-width':0,
      // SHAPE is kind, COLOUR is status — the same two channels the PNG uses, so a
      // reader of one is not learning a second vocabulary for the other.
      'shape': n => ({root:'diamond', terminal:'square'})[n.data('kind')] || 'ellipse',
    }},
    {selector:'edge', style:{
      'width':1.2,'line-color':'var(--edge)','target-arrow-color':'var(--edge)',
      // Translucent, so a node reads as a node and fifty edges read as texture behind it.
      // Opacity rather than a paler colour: it stays right in both themes, and a bundle of
      // edges over the same path darkens, which is information.
      'opacity':.35,
      'line-style': e => ({before_part:'dotted', before_part_inclusive:'dotted',
        same_part:'dashed', after_part:'dashed', after_part_inclusive:'dashed',
        parts:'dashed', between:'dashed'}[e.data('rule')] || 'solid'),
      'line-dash-pattern': e => ({same_part:[8,4], after_part:[12,4],
        after_part_inclusive:[12,4], parts:[3,3], between:[3,3]}[e.data('rule')] || [1,0]),
      'target-arrow-shape':'triangle','arrow-scale':.9,
      // A long edge spanning several columns is drawn as an arc rather than a chord, so it
      // reads as going AROUND the columns it passes rather than through them.
      'curve-style':'unbundled-bezier','control-point-distances':[-18],
      'control-point-weights':[.5],
    }},
    {selector:'.faded', style:{'opacity':.07,'text-opacity':0}},
    {selector:'node.partition-match', style:{'border-width':3,
      'border-color':'var(--sel)','font-weight':'bold','z-index':50}},
    {selector:'node.sel', style:{'border-width':4,'border-color':'var(--sel)',
      'width':34,'height':34,'font-weight':'bold','font-size':15,'z-index':99}},
    {selector:'node.up', style:{'border-width':3,'border-color':'var(--up)'}},
    {selector:'node.down', style:{'border-width':3,'border-color':'var(--down)'}},
    {selector:'node.up-far', style:{'border-width':2,'border-color':'var(--up)',
      'opacity':.32,'text-opacity':.42}},
    {selector:'node.down-far', style:{'border-width':2,'border-color':'var(--down)',
      'opacity':.32,'text-opacity':.42}},
    // A highlighted edge is the answer to the question, so it comes back to full strength.
    {selector:'edge.up', style:{'line-color':'var(--up)','target-arrow-color':'var(--up)',
      'width':2.2,'opacity':.9,'z-index':10}},
    {selector:'edge.down', style:{'line-color':'var(--down)','target-arrow-color':'var(--down)',
      'width':2.2,'opacity':.9,'z-index':10}},
    {selector:'edge.up-far', style:{'line-color':'var(--up)',
      'target-arrow-color':'var(--up)','width':1.5,'opacity':.28,'z-index':5}},
    {selector:'edge.down-far', style:{'line-color':'var(--down)',
      'target-arrow-color':'var(--down)','width':1.5,'opacity':.28,'z-index':5}},
    {selector:'edge.sel', style:{'line-color':'var(--sel)','target-arrow-color':'var(--sel)',
      'width':3,'opacity':1,'z-index':20}},
    {selector:'edge.rule-match', style:{'width':2.8,'opacity':.95,'z-index':15}},
  ],
  layout: LAYOUT,
  wheelSensitivity:.2,
});
cy.ready(land);

function esc(s){const d=document.createElement('div');d.textContent=s==null?'':s;return d.innerHTML}

function clearFilters(){
  selectedRule = null;
  partitionSelect.value = '';
  document.getElementById('q').value = '';
  document.querySelectorAll('.edge-rule').forEach(b => b.setAttribute('aria-pressed', 'false'));
}

function applyFilters(){
  const q = document.getElementById('q').value.trim().toLowerCase();
  const selectedPartition = partitionSelect.value;
  cy.elements().removeClass('faded partition-match rule-match up down up-far down-far sel');
  const matches = cy.nodes().filter(n =>
    (!q || n.data('dataset').toLowerCase().includes(q)) &&
    (!selectedPartition || (selectedPartition === '__none__'
      ? n.data('partitionKeys').length === 0
      : n.data('partitionKeys').includes(selectedPartition))));
  cy.nodes().difference(matches).addClass('faded');
  if (selectedRule) {
    cy.edges().forEach(e => e.toggleClass('rule-match', e.data('rule') === selectedRule));
    cy.edges().difference(cy.edges('.rule-match')).addClass('faded');
  } else if (selectedPartition) {
    cy.edges().addClass('faded');
    matches.addClass('partition-match');
  }
}

partitionSelect.addEventListener('change', applyFilters);
let selectedRule = null;
document.getElementById('edge-legend').addEventListener('click', e => {
  const button = e.target.closest('.edge-rule');
  if (!button) return;
  selectedRule = selectedRule === button.dataset.rule ? null : button.dataset.rule;
  document.querySelectorAll('.edge-rule').forEach(b =>
    b.setAttribute('aria-pressed', b.dataset.rule === selectedRule ? 'true' : 'false'));
  applyFilters();
});

function select(node, fit){
  clearFilters();
  cy.elements().removeClass('faded partition-match rule-match up down up-far down-far sel');
  // The two questions, and the graph library answers both outright: everything that can
  // reach this node, and everything this node can reach.
  const up = node.predecessors(), down = node.successors();
  const upDirect = node.incomers(), downDirect = node.outgoers();
  cy.elements().difference(up.union(down).union(node)).addClass('faded');
  up.difference(upDirect).addClass('up-far');
  down.difference(downDirect).addClass('down-far');
  upDirect.addClass('up'); downDirect.addClass('down'); node.addClass('sel');
  if (fit) cy.animate({fit:{eles:up.union(down).union(node), padding:60}, duration:250});
  panel(node, up.nodes(), down.nodes());
}

function edgePanel(edge){
  clearFilters();
  cy.elements().removeClass('faded partition-match rule-match up down up-far down-far sel');
  edge.addClass('sel');
  edge.connectedNodes().addClass('sel');
  const d = edge.data(), source = byId[d.source], target = byId[d.target];
  const stage = DATA.stages[d.stage] || {};
  document.getElementById('side-body').outerHTML = `<div id="side-body">
    <h2>dependency</h2>
    <p class="why">${esc(source ? source.dataset : d.source)} → ${esc(target ? target.dataset : d.target)}</p>
    <span class="pill" style="background:var(--dim)">${esc(d.rule)}</span>
    ${d.optional ? '<span class="pill" style="background:var(--dim)">optional</span>' : ''}
    <div class="sec">declared by</div>
    <p><code>${esc(d.stage)}</code></p>
    <p class="ok">${esc(stage.why || 'no stage rationale recorded')}</p>
    <div class="sec">reads</div>
    <p>${esc(source ? source.dataset : d.source)} into ${esc(target ? target.dataset : d.target)}</p>
    <div class="sec">rule</div>
    <p class="ok">${esc(d.rule)}${d.optional ? ' — missing input is allowed' : ''}</p>
  </div>`;
}

function panel(node, up, down){
  const d = node.data();
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
  const partitioned = d.partitionKeys.length
    ? `partitioned by ${d.partitionKeys.map(esc).join(' · ')}`
    : 'unpartitioned';

  document.getElementById('side-body').outerHTML = `<div id="side-body">
    <h2>${esc(d.dataset)}</h2>
    <p class="why">${esc(d.why)}</p>
    <span class="pill" style="background:${C[d.status]||C.source}">${esc(d.status)}</span>
    <span class="pill" style="background:var(--dim)">${esc(d.kind)}</span>
    <div class="sec">${partitioned}</div>
    <div class="sec">immediate upstream (${d.reads.length})</div>${list(d.reads)}
    <div class="sec">immediate downstream (${d.readBy.length})</div>${list(d.readBy)}
    <div class="sec">upstream in all (${up.length}) · downstream in all (${down.length})</div>
    <p class="ok">a rebuild of this carries into ${down.length} node(s).</p>
    ${d.writers.length ? `<div class="sec">built by</div><ul>${d.writers.map(w =>
        `<li><code>${esc(w.split('::').pop())}</code>
         <div class="ok">${esc((DATA.stages[w]||{}).why||'')}</div></li>`).join('')}</ul>`
      : '<div class="sec">arrives from outside</div><p class="ok">nothing here writes it</p>'}
    ${d.externals.length ? `<div class="sec">outside this tree</div><ul>${
        d.externals.map(e => `<li><code>${esc(e[0])}</code>
        <div class="ok">${esc(e[1])}</div></li>`).join('')}</ul>` : ''}
    <div class="sec">shards — ${d.shards.filter(s=>s.reason).length} stale of ${d.shards.length}</div>
    ${shards}
  </div>`;
}

window.pick = id => select(cy.$id(id), true);
cy.on('tap','node', e => select(e.target, false));
cy.on('tap','edge', e => edgePanel(e.target));
cy.on('tap', e => { if (e.target === cy) {
  clearFilters();
  cy.elements().removeClass('faded partition-match up down up-far down-far sel');
  document.getElementById('side-body').outerHTML =
    '<p class="hint" id="side-body">Click a node.</p>';
}});

document.getElementById('reset').onclick = () => {
  clearFilters();
  cy.elements().removeClass('faded partition-match up down up-far down-far sel');
  land();
};

document.getElementById('q').addEventListener('input', applyFilters);
</script></body></html>
"""
