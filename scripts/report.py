"""Render evals/results/results.json into a self-contained HTML report."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

METRICS = [
    ("accuracy", "Overall accuracy", True),
    ("lookup", "Lookup questions", True),
    ("aggregate", "Aggregate questions", True),
    ("refusal_on_unanswerable", "Says “not found” when it should", True),
    ("hallucination_rate", "Hallucination rate", False),
    ("citation_accuracy", "Cites the correct document (of lookups answered)", True),
    ("median_seconds", "Median seconds / question", False),
]

CONFIG_NOTES = {
    "naive": "Vector search only, raw chunks — the typical tutorial RAG.",
    "+hybrid": "Adds BM25 keyword search (SQLite FTS5) fused with vector results.",
    "+headers": "Adds the document’s identity (contract no., type, counterparty) to every chunk.",
    "+scoping": "When the question names a contract or counterparty, searches only that document.",
    "full": "Adds structured extraction at ingest and routes list/count/sum questions to SQL.",
}

CSS = """
:root { --fg:#1d2733; --muted:#5b6b7c; --line:#dfe5ec; --good:#1a7f4b; --bad:#b3261e; --accent:#2f4f6f; --bg:#f7f9fb; }
* { box-sizing: border-box; }
body { font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--fg); margin: 0; background: #fff; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 36px 0 12px; border-bottom: 1px solid var(--line); padding-bottom: 6px; }
.meta { color: var(--muted); font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { font-weight: 600; font-size: 13px; color: var(--muted); }
td.num, th.num { text-align: right; }
.best { font-weight: 700; color: var(--good); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 16px; }
.tile { background: var(--bg); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }
.tile .v { font-size: 28px; font-weight: 700; }
.tile .l { color: var(--muted); font-size: 13px; }
.tile .d { font-size: 13px; color: var(--muted); }
.ok { color: var(--good); font-weight: 600; }
.hal { color: var(--bad); font-weight: 600; }
.miss { color: var(--muted); font-weight: 600; }
details { margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
.ans { color: var(--muted); font-size: 13px; max-width: 520px; }
code { font-size: 12px; background: var(--bg); padding: 1px 4px; border-radius: 3px; }
"""


def pct(v) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def metric(summary: dict, key: str):
    if key in ("lookup", "aggregate"):
        return summary["by_type"][key]["accuracy"]
    return summary.get(key)


def fmt(key: str, v) -> str:
    if v is None:
        return "–"
    return f"{v:.1f}" if key == "median_seconds" else pct(v)


def write_report(run: dict, path: Path) -> Path:
    configs = run["configs"]
    names = list(configs)
    first, last = configs[names[0]]["summary"], configs[names[-1]]["summary"]
    n = last["n"]
    bt = last["by_type"]
    e = html.escape

    parts = [f"<!doctype html><html lang='en'><meta charset='utf-8'><title>Accuracy report</title><style>{CSS}</style><main>"]
    parts.append("<h1>Document assistant — accuracy report</h1>")
    parts.append(
        f"<p class='meta'>{n} questions ({bt['lookup']['n']} lookup, {bt['aggregate']['n']} aggregate, "
        f"{bt['unanswerable']['n']} unanswerable) over 30 synthetic contracts · model <code>{e(run['model'])}</code> · "
        f"embeddings <code>{e(run['embed_model'])}</code> · {e(run['started'])}</p>"
    )

    if len(names) > 1:
        parts.append("<div class='tiles'>")
        for key, label, _ in METRICS[:5]:
            a, b = metric(first, key), metric(last, key)
            parts.append(
                f"<div class='tile'><div class='l'>{e(label)}</div><div class='v'>{fmt(key, b)}</div>"
                f"<div class='d'>{e(names[0])}: {fmt(key, a)}</div></div>"
            )
        parts.append("</div>")

    parts.append("<h2>Each fix, measured</h2><table><tr><th>Metric</th>")
    parts += [f"<th class='num'>{e(nm)}</th>" for nm in names]
    parts.append("</tr>")
    for key, label, higher in METRICS:
        vals = [metric(configs[nm]["summary"], key) for nm in names]
        present = [v for v in vals if v is not None]
        best = (max if higher else min)(present) if present else None
        parts.append(f"<tr><td>{e(label)}</td>")
        for v in vals:
            cls = " best" if v is not None and v == best and len(present) > 1 else ""
            parts.append(f"<td class='num{cls}'>{fmt(key, v)}</td>")
        parts.append("</tr>")
    parts.append("</table><table style='margin-top:12px'>")
    for nm in names:
        parts.append(f"<tr><td><code>{e(nm)}</code></td><td class='ans'>{e(CONFIG_NOTES.get(nm, ''))}</td></tr>")
    parts.append("</table>")

    parts.append("<h2>Results by question type</h2><table><tr><th>Tag</th><th class='num'>n</th>")
    parts += [f"<th class='num'>{e(nm)}</th>" for nm in names]
    parts.append("</tr>")
    tags = sorted({s["tag"] for s in configs[names[0]]["scores"]})
    for tag in tags:
        row = [f"<tr><td>{e(tag)}</td>"]
        for i, nm in enumerate(names):
            items = [s for s in configs[nm]["scores"] if s["tag"] == tag]
            if i == 0:
                row.append(f"<td class='num'>{len(items)}</td>")
            row.append(f"<td class='num'>{pct(sum(s['correct'] for s in items) / len(items))}</td>")
        parts.append("".join(row) + "</tr>")
    parts.append("</table>")

    parts.append("<h2>Every question</h2>")
    for nm in names:
        scores = configs[nm]["scores"]
        ok = sum(s["correct"] for s in scores)
        parts.append(f"<details{' open' if nm == names[-1] else ''}><summary>{e(nm)} — {ok}/{len(scores)} correct</summary><table>")
        parts.append("<tr><th>ID</th><th>Result</th><th>Question</th><th>Expected</th><th>Answer</th><th>Route</th></tr>")
        for s in scores:
            if s["correct"]:
                res = "<span class='ok'>correct</span>"
            elif s["hallucinated"]:
                res = "<span class='hal'>wrong</span>"
            else:
                res = "<span class='miss'>no answer</span>"
            exp = s["expected"]
            exp = "not in documents" if exp is None else ", ".join(map(str, exp)) if isinstance(exp, list) else str(exp)
            if s.get("detail", {}).get("missing"):
                res += f"<div class='ans'>missing {e(', '.join(s['detail']['missing']))}</div>"
            if s.get("detail", {}).get("extra"):
                res += f"<div class='ans'>extra {e(', '.join(s['detail']['extra']))}</div>"
            parts.append(
                f"<tr><td>{e(s['qid'])}</td><td>{res}</td><td>{e(s['question'])}</td><td class='ans'>{e(exp)}</td>"
                f"<td class='ans'>{e(s['answer'][:400])}</td><td>{e(s['route'])}</td></tr>"
            )
        parts.append("</table></details>")

    parts.append(
        "<h2>How scoring works</h2><p class='ans' style='max-width:none'>Scoring is deterministic — no LLM judges another LLM. "
        "Lookup answers must contain the expected value (dates and amounts are normalized across formats). Aggregate answers must list "
        "exactly the expected contracts: one missing or one extra contract makes the answer wrong. Unanswerable questions must be "
        "declined; asserting anything counts as a hallucination. A wrong answer is counted as a hallucination; saying “not found” "
        "for an answerable question is a miss, not a hallucination.</p></main></html>"
    )
    path.write_text("".join(parts), encoding="utf-8")
    return path


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "evals/results/results.json")
    print(write_report(json.loads(src.read_text(encoding="utf-8")), src.with_name("report.html")))
