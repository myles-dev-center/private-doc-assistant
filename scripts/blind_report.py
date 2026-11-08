"""The method comparison on blind CUAD sets 4-6, all sets pooled, as Markdown + JSON.

    uv run python scripts/blind_report.py      # prints the table, writes evals/cmp4b/summary.json

Pipeline rows come from evals/cmp4/<combo>/set<N>.json (scripts/run_combos.py), the other
methods from evals/cmp4b/<method>/set<N>/results.json (run_agent.py, run_fullread.py,
run_codex.py). Accuracy comes with a 95% Wilson interval; each method is compared with the
pipeline on the same model and the same questions by an exact McNemar test.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from combos_report import mcnemar, wilson  # noqa: E402

PIPE, BASE = Path("evals/cmp4"), Path("evals/cmp4b")
SETS = ("set4", "set5", "set6")
# (key, label, source, pipeline to compare with). A pipeline combo that reused another
# combo's extraction is charged that combo's extraction cost too: "ingest_from".
RUNS = [
    ("local", "Local · this pipeline", {"combo": "local"}, None),
    ("local-fullread", "Local · model reads every document", None, "local"),
    ("local-agent", "Local · agent with search tools", {"method": "local-agent"}, "local"),
    ("local-codex", "Local · Codex CLI on the document folder", None, "local"),
    ("cloud", "Cloud · this pipeline", {"combo": "cloud", "ingest_from": "cloud-chat"}, None),
    ("cloud-fullread", "Cloud · all documents in the prompt", {"method": "cloud-fullread"}, "cloud"),
    ("cloud-agent", "Cloud · agent with search tools", {"method": "cloud-agent"}, "cloud"),
    ("cloud-codex", "Cloud · Codex CLI on the document folder", {"method": "cloud-codex"}, "cloud"),
]


def load_pipeline(combo: str, ingest_from: str | None) -> dict:
    rows, q_usd, calls, ing_usd, ing_s = [], 0.0, 0, 0.0, 0.0
    for s in SETS:
        r = json.loads((PIPE / combo / f"{s}.json").read_text(encoding="utf-8"))
        rows += [dict(x, set=s) for x in r["scores"]]
        q_usd += r["usage"]["questions"].get("usd", 0)
        calls += r["usage"]["questions"].get("calls", 0)
        ing_usd += r["usage"]["ingest"].get("usd", 0)
        ing_s += r["usage"]["ingest"].get("seconds", 0)
        if ingest_from:
            src = json.loads((PIPE / ingest_from / f"{s}.json").read_text(encoding="utf-8"))["usage"]["ingest"]
            ing_usd += src.get("usd", 0)
            ing_s += src.get("seconds", 0)
    return {"rows": rows, "usd": q_usd, "calls": calls, "ingest_usd": ing_usd, "ingest_min_per_set": ing_s / 60 / len(SETS)}


def load_method(method: str) -> dict:
    rows, q_usd, calls = [], 0.0, 0
    for s in SETS:
        run = json.loads((BASE / method / s / "results.json").read_text(encoding="utf-8"))
        (cfg,) = run["configs"].values()
        rows += [dict(x, set=s) for x in cfg["scores"]]
        q = cfg["summary"]["usage"]["questions"]
        q_usd += q.get("usd", 0)
        calls += q.get("calls", 0)
    return {"rows": rows, "usd": q_usd, "calls": calls, "ingest_usd": None, "ingest_min_per_set": None}


def stats(d: dict) -> dict:
    rows = d["rows"]
    n = len(rows)
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, n)
    by = {t: [x for x in rows if x["qtype"] == t] for t in ("lookup", "aggregate", "unanswerable")}
    secs = sorted(r["seconds"] for r in rows)
    return {
        "questions": n,
        "accuracy": round(k / n, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        **{t: round(sum(x["correct"] for x in v) / len(v), 3) for t, v in by.items()},
        "hallucination": round(sum(r["hallucinated"] for r in rows) / n, 3),
        "median_s": round(statistics.median(secs), 1),
        "p90_s": round(secs[int(0.9 * (n - 1))], 1),
        "calls_per_question": round(d["calls"] / n, 1),
        "usd_per_question": round(d["usd"] / n, 4),
        "usd_questions": round(d["usd"], 2),
        "usd_ingest": None if d["ingest_usd"] is None else round(d["ingest_usd"], 2),
        "ingest_min_per_set": None if d["ingest_min_per_set"] is None else round(d["ingest_min_per_set"]),
    }


def main() -> None:
    loaded, out = {}, []
    for key, label, src, vs in RUNS:
        if src is None:
            out.append({"key": key, "label": label, "run": False})
            continue
        d = load_pipeline(src["combo"], src.get("ingest_from")) if "combo" in src else load_method(src["method"])
        loaded[key] = d
        s = {"key": key, "label": label, "run": True} | stats(d)
        if vs:
            a = {(r["set"], r["qid"]): r["correct"] for r in loaded[vs]["rows"]}
            b = {(r["set"], r["qid"]): r["correct"] for r in d["rows"]}
            only_pipe = sum(a[q] and not b[q] for q in a.keys() & b.keys())
            only_this = sum(b[q] and not a[q] for q in a.keys() & b.keys())
            s["vs_pipeline"] = {"pipeline_only": only_pipe, "method_only": only_this, "p": round(mcnemar(only_pipe, only_this), 4)}
        out.append(s)

    print(
        "| Method | Accuracy (95% CI) | One agreement | Across all | Not found | Hallucination | Median s | p90 s "
        "| Calls / question | $ / question | Ingestion, 3 sets | vs pipeline (p) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in out:
        if not s["run"]:
            print(f"| {s['label']} | not run | | | | | | | | | | |")
            continue
        ing = "—" if s["usd_ingest"] is None else f"${s['usd_ingest']:.2f}, {s['ingest_min_per_set']} min / set"
        vs = s.get("vs_pipeline")
        vs = f"{vs['method_only']} vs {vs['pipeline_only']} (p={vs['p']:.2f})" if vs else ""
        print(
            f"| {s['label']} | **{s['accuracy']:.0%}** ({s['ci95'][0]:.0%}–{s['ci95'][1]:.0%}) | {s['lookup']:.0%} | "
            f"{s['aggregate']:.0%} | {s['unanswerable']:.0%} | {s['hallucination']:.0%} | {s['median_s']:.0f} | {s['p90_s']:.0f} | "
            f"{s['calls_per_question']:g} | ${s['usd_per_question']:.3f} | {ing} | {vs} |"
        )
    (BASE / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {BASE / 'summary.json'}")


if __name__ == "__main__":
    main()
