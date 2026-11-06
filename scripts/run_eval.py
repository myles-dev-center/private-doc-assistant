"""Run the golden set against several pipeline configurations and write a report.

    uv run python scripts/run_eval.py                       # all configs
    uv run python scripts/run_eval.py --configs full --limit 5

Configurations are cumulative, so the report shows what each fix contributes:

    naive       vector search only, raw chunks          (typical tutorial RAG)
    +hybrid     + BM25 keyword search fused with RRF     (exact IDs, names)
    +headers    + document identity on every chunk       (boilerplate-heavy docs)
    +scoping    + answer from the document a question names (contract no., counterparty)
    full        + structured extraction & SQL routing    (aggregate questions)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml
from compare_common import public_endpoint

from docassist import Assistant, get_settings
from docassist.evaluation import score, summarize
from docassist.llm import Usage
from docassist.pipeline import Answer

CONFIGS = {
    "naive": {"store": "plain", "hybrid": False, "context_headers": False, "doc_scoping": False, "structured": False},
    "+hybrid": {"store": "plain", "hybrid": True, "context_headers": False, "doc_scoping": False, "structured": False},
    "+headers": {"store": "enriched", "hybrid": True, "context_headers": True, "doc_scoping": False, "structured": False},
    "+scoping": {"store": "enriched", "hybrid": True, "context_headers": True, "doc_scoping": True, "structured": False},
    "full": {"store": "enriched", "hybrid": True, "context_headers": True, "doc_scoping": True, "structured": True},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    ap.add_argument("--golden", type=Path, default=Path("evals/golden.yaml"))
    ap.add_argument("--out", type=Path, default=Path("evals/results"))
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--tag", default="", help="suffix for store dirs, e.g. 'holdout' for a second corpus")
    args = ap.parse_args()

    golden = yaml.safe_load(args.golden.read_text(encoding="utf-8"))
    # A golden file is a list of questions, or {documents: {file: [aliases]}, questions: [...]}
    # for corpora whose documents are recognized by name rather than by an ID pattern.
    documents = golden.get("documents") if isinstance(golden, dict) else None
    if isinstance(golden, dict):
        golden = golden["questions"]
    if args.limit:
        golden = golden[: args.limit]
    base = get_settings()
    args.out.mkdir(parents=True, exist_ok=True)
    run = {
        "model": base.llm_model,
        "embed_model": base.embed_model,
        "corpus": str(args.corpus),
        "endpoint": public_endpoint(base.llm_base_url),
        "started": time.strftime("%Y-%m-%d %H:%M"),
        "configs": {},
    }

    for name in args.configs.split(","):
        cfg = CONFIGS[name]
        store_dir = base.data_dir.parent / f"eval-{cfg['store']}{'-' + args.tag if args.tag else ''}"
        # Configs sharing a store reuse it; ingestion skips unchanged files.
        settings = base.model_copy(update={k: v for k, v in cfg.items() if k != "store"} | {"data_dir": store_dir})
        assistant = Assistant(settings)
        print(f"\n== {name}: ingesting into {store_dir}", flush=True)
        t0 = time.time()
        assistant.ingest_dir(args.corpus, progress=lambda i, n, r: print(f"  [{i}/{n}] {r['status']} {r['filename']}", flush=True))
        # Token usage is split into one-off ingestion and per-question answering. Ingestion
        # of a store that already exists is skipped, so its usage reads as zero.
        ingest = assistant.llm.usage.snapshot() | {"seconds": round(time.time() - t0, 1)}
        assistant.llm.usage = Usage()

        scores = []
        for q in golden:
            try:
                ans = assistant.ask(q["question"])
            except Exception as e:  # keep going; a crash counts as a wrong answer
                ans = Answer(q["question"], f"ERROR: {e}", "error")
            s = score(q, ans, documents)
            scores.append(s)
            mark = "ok " if s.correct else ("HAL" if s.hallucinated else "-- ")
            print(f"  {mark} {q['id']} [{s.route}] {s.seconds:>5.1f}s  {q['question'][:70]}", flush=True)
        summary = summarize(scores)
        summary["usage"] = {"ingest": ingest, "questions": assistant.llm.usage.snapshot()}
        print(f"  => accuracy {summary['accuracy']}, hallucination {summary['hallucination_rate']}", flush=True)
        run["configs"][name] = {"settings": cfg, "summary": summary, "scores": [asdict(s) for s in scores]}
        (args.out / "results.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")

    from report import write_report  # scripts/report.py

    path = write_report(run, args.out / "report.html")
    print(f"\nreport: {path}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
