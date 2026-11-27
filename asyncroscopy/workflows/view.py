"""One page for a person that shows every workflow with its graph, its purpose, who asked for it and its status.

    python -m asyncroscopy.workflows view            # writes outputs/workflows/workflows.html
    http://127.0.0.1:8003/workflows                         # the same page served live by analysis_host

The page is one HTML file that loads nothing from the network. Each graph is drawn as SVG from the workflow's
blueprint. Workflows waiting for approval come first, each with the request it was made for and the command that
approves it. Anything a model wrote is escaped and shown as text.
"""

from __future__ import annotations

import html
import time

from .library import APPROVED, DRAFT, SHIPPED, Library
from .checks import END, START, edges_of

NODE_W, NODE_H, GAP, SIDE = 230, 46, 34, 150
ORDER = {DRAFT: 0, APPROVED: 1, SHIPPED: 2}
HEADINGS = {DRAFT: "Waiting for a person's approval", APPROVED: "Approved", SHIPPED: "Shipped with the software"}


def _ordered(blueprint: dict) -> list[str]:
    """START, then the nodes in the order the graph reaches them, then END."""
    following: dict[str, list[str]] = {}
    for edge in edges_of(blueprint):
        for target in (edge.get("to"), edge.get("otherwise")):
            if edge.get("from") and target:
                following.setdefault(str(edge["from"]), []).append(str(target))
    seen, queue = [START], [START]
    while queue:
        for nxt in following.get(queue.pop(0), []):
            if nxt not in seen and nxt != END:
                seen.append(nxt)
                queue.append(nxt)
    rest = [n for n in blueprint.get("nodes") or {} if n not in seen]
    return [*seen, *rest, END]


def graph_svg(blueprint: dict) -> str:
    """Draw the workflow's graph as SVG.

    Nodes go top to bottom in the order they run. An edge that skips ahead or goes back is drawn as an arc at the
    side with its condition.
    """
    order = _ordered(blueprint)
    row = {name: k for k, name in enumerate(order)}
    nodes = blueprint.get("nodes") or {}
    height = len(order) * (NODE_H + GAP) - GAP + 20
    width = NODE_W + 2 * SIDE
    left, mid = SIDE, SIDE + NODE_W / 2

    def top(name: str) -> float:
        return 10 + row[name] * (NODE_H + GAP)

    parts = [f'<svg class="graph" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="workflow graph">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0,0 L10,5 L0,10 z" class="head"/></marker></defs>']
    arcs = 0
    for edge in edges_of(blueprint):
        source = str(edge.get("from"))
        joined = f"after {' and '.join(str(n) for n in edge['after'])}" if isinstance(edge.get("after"), list) else None
        for target, label in ((edge.get("to"), edge.get("when") or joined), (edge.get("otherwise"), "otherwise" if "when" in edge else None)):
            if target is None or source not in row or str(target) not in row:
                continue
            target = str(target)
            if row[target] == row[source] + 1:
                y1, y2 = top(source) + NODE_H, top(target)
                parts.append(f'<line class="edge" x1="{mid}" y1="{y1}" x2="{mid}" y2="{y2 - 2}" marker-end="url(#arrow)"/>')
                if label:
                    parts.append(f'<text class="label" x="{mid + 8}" y="{(y1 + y2) / 2 + 4}">{html.escape(str(label))}</text>')
                continue
            arcs += 1
            right = arcs % 2 == 1
            x = left + NODE_W if right else left
            reach = (40 + 22 * ((arcs - 1) // 2)) * (1 if right else -1)
            y1, y2 = top(source) + NODE_H / 2 + (6 if target == source else 0), top(target) + NODE_H / 2 - (6 if target == source else 0)
            if target == source:
                y1, y2 = top(source) + NODE_H - 10, top(source) + 10
            parts.append(f'<path class="edge" d="M{x},{y1} C{x + reach},{y1} {x + reach},{y2} {x + (2 if right else -2)},{y2}" marker-end="url(#arrow)"/>')
            if label:
                bound = f"at most {edge['max_visits']} times" if target == edge.get("to") and edge.get("max_visits") else ""
                anchor, at = "start" if right else "end", x + reach * 0.82 + (6 if right else -6)
                middle = (y1 + y2) / 2 + (-3 if bound else 4)
                parts.append(f'<text class="label" text-anchor="{anchor}" x="{at}" y="{middle}">{html.escape(str(label))}'
                             + (f'<tspan x="{at}" dy="14">{html.escape(bound)}</tspan>' if bound else "") + "</text>")
    for name in order:
        y = top(name)
        if name in (START, END):
            parts.append(f'<rect class="terminal" x="{mid - 45}" y="{y + 8}" width="90" height="{NODE_H - 16}" rx="15"/>'
                         f'<text class="name" text-anchor="middle" x="{mid}" y="{y + NODE_H / 2 + 5}">{name.lower()}</text>')
            continue
        block = (nodes.get(name) or {}).get("block")
        parts.append(f'<rect class="node" x="{left}" y="{y}" width="{NODE_W}" height="{NODE_H}" rx="8"/>'
                     f'<text class="name" text-anchor="middle" x="{mid}" y="{y + 19}">{html.escape(name)}</text>'
                     f'<text class="block" text-anchor="middle" x="{mid}" y="{y + 36}">{html.escape(str(block) if block else "not a block")}</text>')
    parts.append("</svg>")
    return "".join(parts)


def _when(seconds) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(seconds)))
    except (TypeError, ValueError):
        return ""


def _card(library: Library, entry: dict) -> str:
    name, status = str(entry["name"]), str(entry["status"])
    esc = html.escape
    try:
        workflow, _ = library.find(name)
        blueprint, checked, source = workflow.blueprint(), workflow.lint(), workflow.source()
    except Exception as exc:  # one broken workflow must not hide the others
        return f'<article class="card"><h3>{esc(name)}</h3><p class="warn">This workflow could not be read: {esc(str(exc))}</p></article>'
    facts = []
    if status == DRAFT:
        asked = library.request(name)
        if asked:
            facts.append(f'<p class="request"><b>Requested</b> {esc(_when(asked.get("at")))}: “{esc(str(asked.get("request") or ""))}”</p>')
        else:
            facts.append('<p class="request"><b>Requested</b>: no request was recorded with this draft.</p>')
        if entry.get("note"):
            facts.append(f'<p class="warn">{esc(str(entry["note"]))}</p>')
        runs = library.rehearsals(name, checked.get("hash"))
        done = [r for r in runs if r.get("state") == "done"]
        facts.append(f'<p><b>Rehearsed on a digital twin</b>: {len(done)} finished run(s) of this exact code'
                     + (f', last {esc(_when(done[-1].get("at")))}' if done else "; it has never run to the end anywhere") + ".</p>")
        facts.append(f'<p><b>To approve it</b>, after reading the code below, run:</p><pre class="command">python -m asyncroscopy.workflows approve {esc(name)}</pre>')
    elif status == APPROVED:
        record = library.approval(name) or {}
        ends = f', until {esc(str(entry.get("approval_ends")))}' if entry.get("approval_ends") else ", until revoked"
        facts.append(f'<p><b>Approved</b> by {esc(str(record.get("by") or "?"))} on {esc(_when(record.get("at")))}, for '
                     f'{esc(str(record.get("instrument") or "any"))} instrument{ends}.</p>')
        if record.get("request"):
            facts.append(f'<p class="request"><b>Requested</b>: “{esc(str(record["request"]))}”</p>')
    inputs = "".join(
        f'<tr><td><code>{esc(str(k))}</code></td><td>{esc(str((v or {}).get("default", "")))}</td>'
        f'<td>{esc(str((v or {}).get("min", "")))} … {esc(str((v or {}).get("max", "")))}</td><td>{esc(str((v or {}).get("doc", "")))}</td></tr>'
        for k, v in (blueprint.get("inputs") or {}).items())
    problems = "".join(f"<li>{esc(str(p))}</li>" for p in checked.get("problems") or [])
    return (
        f'<article class="card" id="{esc(name)}"><header><h3>{esc(name)}</h3><span class="badge {esc(status)}">{esc(status)}</span>'
        + ('<span class="badge acts">moves the instrument</span>' if checked.get("acts") else '<span class="badge quiet">reads only</span>')
        + f'</header><p class="summary">{esc(str(blueprint.get("summary") or ""))}</p>{"".join(facts)}'
        + (f'<ul class="warn">{problems}</ul>' if problems else "")
        + f'<div class="body"><div class="drawing">{graph_svg(blueprint)}</div><div class="detail">'
        + (f'<table><thead><tr><th>Input</th><th>Default</th><th>Range</th><th>What it is</th></tr></thead><tbody>{inputs}</tbody></table>' if inputs else "")
        + f'<p class="small"><b>Blocks</b>: {esc(", ".join(checked.get("blocks") or []))}</p>'
        f'<p class="small"><b>Tools it can call</b>: {esc(", ".join(checked.get("tools") or []))}</p>'
        f'<p class="small"><b>Identity</b>: <code>{esc(str(checked.get("hash") or ""))}</code></p>'
        f'<details><summary>The LangGraph code</summary><pre class="code">{esc(source)}</pre></details></div></div></article>'
    )


STYLE = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#1c2330;--muted:#5d6b7e;--line:#d5dbe5;--accent:#2457c5;--warn:#a4520a;--ok:#1d7a46;--node:#eef3fd}
@media (prefers-color-scheme: dark){:root{--bg:#12161d;--card:#1a202a;--ink:#e6eaf1;--muted:#9aa7b8;--line:#2d3644;--accent:#7ba5ff;--warn:#f0a35a;--ok:#63c98f;--node:#222c3c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 10px;color:var(--muted);font-weight:600}
.lead{color:var(--muted);margin:0 0 8px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 16px}
.card header{display:flex;align-items:center;gap:8px;flex-wrap:wrap}h3{font-size:17px;margin:0;font-family:ui-monospace,Consolas,monospace}
.badge{font-size:12px;padding:2px 9px;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.badge.draft{color:var(--warn);border-color:var(--warn)}.badge.approved,.badge.shipped{color:var(--ok);border-color:var(--ok)}.badge.acts{color:var(--accent);border-color:var(--accent)}
.summary{margin:8px 0}.request{border-left:3px solid var(--accent);padding-left:10px}.warn{color:var(--warn)}.small{font-size:13px;color:var(--muted);overflow-wrap:anywhere}
.body{display:flex;gap:20px;flex-wrap:wrap;align-items:flex-start;margin-top:10px}.drawing{overflow-x:auto;max-width:100%}.detail{flex:1 1 380px;min-width:0}
table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:8px}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
pre{margin:6px 0;padding:10px 12px;border:1px solid var(--line);border-radius:8px;overflow-x:auto;font:13px/1.45 ui-monospace,Consolas,monospace;background:var(--bg)}
details summary{cursor:pointer;color:var(--accent)}code{font-family:ui-monospace,Consolas,monospace}
svg.graph{display:block}.node{fill:var(--node);stroke:var(--accent);stroke-width:1.3}.terminal{fill:none;stroke:var(--muted);stroke-width:1.2}
.name{fill:var(--ink);font:600 13px ui-monospace,Consolas,monospace}.block{fill:var(--muted);font:12px system-ui,sans-serif}
.edge{fill:none;stroke:var(--muted);stroke-width:1.4}.head{fill:var(--muted)}.label{fill:var(--muted);font:11.5px system-ui,sans-serif}
.empty{color:var(--muted);font-style:italic}
"""


def render_html(library: Library) -> str:
    """Build the whole page as one HTML string."""
    entries = sorted(library.listed(), key=lambda e: (ORDER.get(str(e.get("status")), 9), str(e.get("name"))))
    sections = []
    for status in (DRAFT, APPROVED, SHIPPED):
        cards = [_card(library, e) for e in entries if e.get("status") == status]
        body = "".join(cards) if cards else '<p class="empty">None.</p>'
        sections.append(f'<h2>{HEADINGS[status]} ({len(cards)})</h2>{body}')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Workflows</title><style>{STYLE}</style></head><body><main><h1>Workflows</h1>'
        '<p class="lead">Each workflow is a LangGraph graph built from blocks. A shipped or approved workflow may run on the instrument; '
        'one waiting for approval runs on a digital twin only. An agent writes a workflow for a task a person gave it, and a scientist approves it.</p>'
        f'<p class="small">Library: {html.escape(str(library.root))} · drawn {html.escape(_when(time.time()))}</p>'
        f'{"".join(sections)}</main></body></html>'
    )
