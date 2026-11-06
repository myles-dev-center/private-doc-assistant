"""Method "Codex": a production coding agent (OpenAI Codex CLI) answers from a folder of documents.

    uv run python scripts/run_codex.py --corpus data/corpus-cuad-holdout2 \
        --golden evals/cuad-golden-holdout2.yaml --out evals/cmp/cloud-codex \
        --codex-config ~/.codex-bench/config.toml --max-cost 30

Each document is converted to a text file with the same parser the other methods
use (one "--- page N ---" marker per page, so the agent can cite pages), and the
folder is mounted read-only into a Docker container running `codex exec`. Every
question starts a fresh session, so no question sees another's work. Codex runs
with its own sandbox off because the container is the sandbox; on Windows the
built-in sandbox blocks every shell command in `codex exec`.

--codex-config is a config.toml with only the model and provider (no MCP servers,
memories or skills), so the run measures Codex itself, not a personal setup.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from compare_common import ANSWER_RULES, Run, doc_text, load_corpus, load_golden, to_answer

from docassist.llm import call_cost

PROMPT = """The current directory holds a collection of documents, one text file per document
(file name = original document name + ".txt"; pages are marked "--- page N ---").
Answer the question below from these documents only. Do not modify any files.
{rules}
Cite documents by their original file name (without ".txt").

Question: {question}"""


def export_text(docs, folder: Path) -> None:
    for d in docs:
        (folder / f"{d.filename}.txt").write_text(doc_text(d), encoding="utf-8")


def ask(args, folder: Path, question: str) -> tuple[str, dict]:
    cmd = [
        "docker", "run", "--rm", "-i", "--network", "bridge",
        "-v", f"{args.codex_config.resolve()}:/codex-home/config.toml:ro",
        "-v", f"{folder.resolve()}:/docs:ro",
        "-e", "CODEX_HOME=/codex-home",
        args.image,
        "codex", "exec", "--json", "--skip-git-repo-check", "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox", "-C", "/docs",
        "-c", f"model_reasoning_effort={args.reasoning_effort}",
        "-m", args.model,
        PROMPT.format(rules=ANSWER_RULES, question=question),
    ]  # fmt: skip
    try:
        out = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, timeout=args.timeout, encoding="utf-8")
        lines = out.stdout.splitlines()
    except subprocess.TimeoutExpired as e:
        lines = (e.stdout or b"").decode("utf-8", "replace").splitlines() if isinstance(e.stdout, bytes) else (e.stdout or "").splitlines()
        lines.append(json.dumps({"type": "timeout"}))
    text, usage, commands, turns = "", {"input": 0, "cached": 0, "output": 0}, 0, 0
    events = []
    for line in lines:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        events.append(ev)
        item = ev.get("item") or {}
        if ev.get("type") == "item.completed" and item.get("type") == "agent_message" and item.get("text", "").strip():
            text = item["text"]
        if ev.get("type") == "item.completed" and item.get("type") == "command_execution":
            commands += 1
        if ev.get("type") == "turn.completed":
            turns += 1
            u = ev.get("usage") or {}
            usage["input"] += u.get("input_tokens", 0)
            usage["cached"] += u.get("cached_input_tokens", 0)
            usage["output"] += u.get("output_tokens", 0)
    timed_out = any(e.get("type") == "timeout" for e in events)
    # Codex reports one usage total per turn, not per model call, so the long-context
    # surcharge (per call, >272k prompt) cannot be applied; list price without it.
    usd = call_cost(args.model, usage["input"], usage["cached"], usage["output"])
    return text or ("ERROR: timeout" if timed_out else "ERROR: no answer"), {
        **usage,
        "calls": commands,
        "embed_tokens": 0,
        "usd": round(usd, 4),
        "_events": events,
        "timed_out": timed_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--codex-config", type=Path, required=True)
    ap.add_argument("--image", default="codex-bench:0.160.0")
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per question")
    ap.add_argument("--max-cost", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    documents, questions = load_golden(args.golden, args.limit)
    docs = load_corpus(args.corpus)
    files = [d.filename for d in docs]
    run = Run(
        "codex",
        args.out,
        documents,
        {
            "model": args.model,
            "embed_model": "-",
            "corpus": str(args.corpus),
            "endpoint": "OpenAI-compatible API",
            "agent": f"Codex CLI ({args.image.split(':')[-1]}), reasoning effort {args.reasoning_effort}",
        },
    )
    traces = args.out / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        export_text(docs, folder)
        for q in questions:
            t0 = time.time()
            text, usage = ask(args, folder, q["question"])
            events = usage.pop("_events")
            (traces / f"{q['id']}.jsonl").write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            ans = to_answer(q["question"], text, files, round(time.time() - t0, 1), "codex")
            timed_out = usage.pop("timed_out")
            run.add(q, ans, usage, {"steps": usage["calls"], "timed_out": timed_out})
            if run.spent() > args.max_cost:
                print(f"STOP: spend ${run.spent():.2f} passed --max-cost {args.max_cost}", flush=True)
                break
    run.save()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
