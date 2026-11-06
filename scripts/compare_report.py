"""Summarize the method comparison (evals/cmp/*/results.json) as a Markdown table + JSON.

    uv run python scripts/compare_report.py            # prints the table, writes evals/cmp/summary.json

Cost is list price for the tokens actually used (see PRICES in docassist/llm.py);
"per question" excludes one-off ingestion, which is shown separately.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

CMP = Path("evals/cmp")
RUNS = [
    ("local-rag", "Local · our RAG"),
    ("local-fullread", "Local · read every document"),
    ("local-agent", "Local · agent with tools"),
    ("cloud-rag", "Cloud · our RAG"),
    ("cloud-fullread", "Cloud · all documents in the prompt"),
    ("cloud-agent", "Cloud · agent with tools"),
    # "calls" for Codex are the shell commands it ran; it reports tokens per turn, not per model call.
    ("cloud-codex", "Cloud · Codex CLI"),
]


def load(name: str) -> dict:
    run = json.loads((CMP / name / "results.json").read_text(encoding="utf-8"))
    (cfg,) = run["configs"].values()
    s, scores = cfg["summary"], cfg["scores"]
    usage = s.get("usage", {})
    q = usage.get("questions", {})
    setup = usage.get("ingest") or usage.get("setup") or {}
    secs = sorted(x["seconds"] for x in scores)
    n = len(scores)
    calls = q.get("calls", 0)
    return {
        "name": name,
        "model": run["model"],
        "accuracy": s["accuracy"],
        "lookup": s["by_type"]["lookup"]["accuracy"],
        "aggregate": s["by_type"]["aggregate"]["accuracy"],
        "unanswerable": s["by_type"]["unanswerable"]["accuracy"],
        "hallucination": s["hallucination_rate"],
        "aggregate_recall": s["aggregate_recall"],
        "median_s": round(statistics.median(secs), 1),
        "p90_s": round(secs[int(0.9 * (n - 1))], 1),
        "usd_per_question": round(q.get("usd", 0) / n, 4),
        "usd_questions": round(q.get("usd", 0), 2),
        "usd_ingest": round(setup.get("usd", 0), 2),
        "ingest_min": round(setup.get("seconds", 0) / 60, 1),
        "calls_per_question": round(calls / n, 1),
        "input_tokens_per_question": round(q.get("input", 0) / n),
    }


def pct(v: float) -> str:
    return f"{v * 100:.0f}%"


def main() -> None:
    rows = [load(name) | {"label": label} for name, label in RUNS if (CMP / name / "results.json").exists()]
    print(
        "| Method | Accuracy | Lookup | Aggregate | Says “not found” | Hallucination | Median s/question "
        "| Model calls / question | Cost / question | One-off ingestion |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ingest = f"${r['usd_ingest']:.2f}, {r['ingest_min']:.0f} min" if r["ingest_min"] else "—"
        print(
            f"| {r['label']} | {pct(r['accuracy'])} | {pct(r['lookup'])} | {pct(r['aggregate'])} | {pct(r['unanswerable'])} "
            f"| {pct(r['hallucination'])} | {r['median_s']:.0f} | {r['calls_per_question']:g} | ${r['usd_per_question']:.3f} | {ingest} |"
        )
    (CMP / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
