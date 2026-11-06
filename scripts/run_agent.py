"""Method "agent": the model searches the documents itself with tools, then answers.

    uv run python scripts/run_agent.py --corpus data/corpus-cuad-holdout2 \
        --golden evals/cuad-golden-holdout2.yaml --out evals/cmp/cloud-agent --max-cost 15

The tools are the generic ones a coding or research agent has (list files, grep,
read a page) - no index, no embeddings, nothing tuned to the question set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from compare_common import ANSWER_RULES, Run, delta, load_corpus, load_golden, public_endpoint, to_answer

from docassist import get_settings
from docassist.llm import LLM

SYSTEM = (
    "You answer questions about a collection of documents. You cannot see them directly: use the tools "
    "to list, search and read them, then answer. Search before you read; read the pages that matter.\n" + ANSWER_RULES
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": "List every document: file name, number of pages and its first line.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Case-insensitive search of all documents (or one). Plain words or a regular expression. "
            "Returns up to 40 matching lines with file name and page.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "file": {"type": "string", "description": "optional: one file name"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": "Read one page of one document.",
            "parameters": {
                "type": "object",
                "properties": {"file": {"type": "string"}, "page": {"type": "integer"}},
                "required": ["file", "page"],
            },
        },
    },
]


class Library:
    def __init__(self, docs):
        self.docs = {d.filename: d for d in docs}

    def call(self, name: str, args: dict) -> str:
        try:
            if name == "list_documents":
                return "\n".join(
                    f"{f} | {len(d.pages)} pages | {(d.pages[0].text.strip().splitlines() or [''])[0][:100]}" for f, d in self.docs.items()
                )
            if name == "search":
                try:
                    rx = re.compile(args["pattern"], re.I)
                except re.error:
                    rx = re.compile(re.escape(args["pattern"]), re.I)
                hits = []
                for f, d in self.docs.items():
                    if args.get("file") and f != args["file"]:
                        continue
                    for p in d.pages:
                        for line in p.text.splitlines():
                            if rx.search(line):
                                hits.append(f"{f} p.{p.number}: {line.strip()[:200]}")
                return "\n".join(hits[:40]) + (f"\n... {len(hits) - 40} more matches" if len(hits) > 40 else "") or "no matches"
            if name == "read_page":
                d = self.docs.get(args["file"])
                if not d:
                    return f"no such file: {args['file']}"
                page = next((p for p in d.pages if p.number == int(args["page"])), None)
                return page.text if page else f"{args['file']} has pages 1-{len(d.pages)}"
        except Exception as e:
            return f"tool error: {e}"
        return f"unknown tool {name}"


def ask(llm: LLM, lib: Library, question: str, max_steps: int, max_usd: float) -> tuple[str, int]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    kwargs = {"reasoning_effort": llm.s.llm_reasoning_effort} if llm.s.llm_reasoning_effort else {}
    start = llm.usage.usd
    for step in range(max_steps + 1):
        last = step == max_steps or llm.usage.usd - start > max_usd
        if last:
            msgs.append({"role": "user", "content": "Stop searching now and answer with what you have found."})
        resp = llm.chat_client.chat.completions.create(
            model=llm.s.llm_model,
            messages=msgs,
            tools=None if last else TOOLS,
            temperature=0,
            max_tokens=1500,
            **kwargs,
        )
        llm.usage.add(resp.usage, llm.s.llm_model)
        m = resp.choices[0].message
        if last or not m.tool_calls:
            return m.content or "", step
        msgs.append({"role": "assistant", "content": m.content, "tool_calls": [t.model_dump() for t in m.tool_calls]})
        for t in m.tool_calls:
            try:
                a = json.loads(t.function.arguments or "{}")
            except json.JSONDecodeError:
                a = {}
            msgs.append({"role": "tool", "tool_call_id": t.id, "content": lib.call(t.function.name, a)[:12_000]})
    return "", max_steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-cost", type=float, default=15.0)
    ap.add_argument("--max-cost-per-question", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    documents, questions = load_golden(args.golden, args.limit)
    settings = get_settings()
    llm = LLM(settings)
    docs = load_corpus(args.corpus)
    lib, files = Library(docs), [d.filename for d in docs]
    run = Run(
        "agent",
        args.out,
        documents,
        {"model": settings.llm_model, "embed_model": "-", "corpus": str(args.corpus), "endpoint": public_endpoint(settings.llm_base_url)},
    )
    for q in questions:
        before, t0 = llm.usage.snapshot(), time.time()
        try:
            text, steps = ask(llm, lib, q["question"], args.max_steps, args.max_cost_per_question)
        except Exception as e:
            text, steps = f"ERROR: {e}", -1
        ans = to_answer(q["question"], text, files, round(time.time() - t0, 1), "agent")
        run.add(q, ans, delta(before, llm.usage.snapshot()), {"steps": steps})
        if run.spent() > args.max_cost:
            print(f"STOP: spend ${run.spent():.2f} passed --max-cost {args.max_cost}", flush=True)
            break
    run.save()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
