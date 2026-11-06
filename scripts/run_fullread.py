"""Method "full read": no index, no retrieval - the model reads every document for every question.

    # cloud: all documents in one long prompt
    uv run python scripts/run_fullread.py --mode single --corpus data/corpus-cuad-holdout2 \
        --golden evals/cuad-golden-holdout2.yaml --out evals/cmp/cloud-fullread --max-cost 25
    # local: the documents don't fit a small model's context, so read them part by part
    # (map) and answer from the notes (reduce)
    uv run python scripts/run_fullread.py --mode map --corpus ... --out evals/cmp/local-fullread

In map mode a document part is read once for all questions so the server's prompt cache
can be reused; that changes how long the run takes, not the answers. The reported time
per question is what a user would wait: every part read from scratch (the first, uncached
read of each part) plus the final answer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from compare_common import ANSWER_RULES, Run, delta, doc_text, load_corpus, load_golden, public_endpoint, to_answer

from docassist import get_settings
from docassist.llm import LLM

SINGLE_SYSTEM = "You answer questions about the documents provided.\n" + ANSWER_RULES

MAP_SYSTEM = """You read one part of one document and take notes for a question. The question may be about
this document or about many documents ("list all agreements that ...", "how many ..."); you see only this one,
so note whether THIS document matches and why. Other parts and documents are read separately.
- Quote or paraphrase everything in this part that helps answer the question, with the page, like (p. 3).
- If this part has nothing relevant, reply exactly: NONE
- Use only this part. Be brief."""

REDUCE_SYSTEM = "You answer a question from notes taken while reading each document in full.\n" + ANSWER_RULES


def parts_of(doc, max_chars: int) -> list[str]:
    text = doc_text(doc)
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)] or [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "map"], required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--golden", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--part-chars", type=int, default=36_000, help="map mode: characters per document part")
    ap.add_argument("--max-cost", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    documents, questions = load_golden(args.golden, args.limit)
    settings = get_settings()
    llm = LLM(settings)
    docs = load_corpus(args.corpus)
    files = [d.filename for d in docs]
    run = Run(
        f"fullread-{args.mode}",
        args.out,
        documents,
        {"model": settings.llm_model, "embed_model": "-", "corpus": str(args.corpus), "endpoint": public_endpoint(settings.llm_base_url)},
    )

    if args.mode == "single":
        context = "\n\n".join(f"===== FILE: {d.filename} =====\n{doc_text(d)}" for d in docs)
        print(f"{len(docs)} documents, {len(context):,} characters in every prompt", flush=True)
        for q in questions:
            before, t0 = llm.usage.snapshot(), time.time()
            try:
                text = llm.chat(SINGLE_SYSTEM, f"{context}\n\nQuestion: {q['question']}", max_tokens=1500)
            except Exception as e:
                text = f"ERROR: {e}"
            run.add(q, to_answer(q["question"], text, files, round(time.time() - t0, 2), "fullread"), delta(before, llm.usage.snapshot()))
            if run.spent() > args.max_cost:
                print(f"STOP: spend ${run.spent():.2f} passed --max-cost {args.max_cost}", flush=True)
                break
        run.save()
        return

    # ---- map: every part of every document, for every question
    parts = [(d.filename, i, p) for d in docs for i, p in enumerate(parts_of(d, args.part_chars), 1)]
    print(f"{len(docs)} documents in {len(parts)} parts; {len(parts) * len(questions)} reads", flush=True)
    notes = {q["id"]: [] for q in questions}
    wait = {q["id"]: 0.0 for q in questions}  # what a user would wait for the reads
    used = {q["id"]: dict.fromkeys(llm.usage.snapshot(), 0) for q in questions}
    for n, (fname, i, part) in enumerate(parts, 1):
        first = None
        for q in questions:
            before, t0 = llm.usage.snapshot(), time.time()
            try:
                note = llm.chat(MAP_SYSTEM, f"Document {fname}, part {i}:\n<<<\n{part}\n>>>\n\nQuestion: {q['question']}", max_tokens=300)
            except Exception as e:
                note = f"ERROR: {e}"
            secs = time.time() - t0
            first = secs if first is None else first
            d = delta(before, llm.usage.snapshot())
            used[q["id"]] = {k: used[q["id"]][k] + d[k] for k in d}
            # the first question on a part paid the full read; the others hit the prompt cache
            wait[q["id"]] += first
            if not note.strip().upper().startswith("NONE"):
                notes[q["id"]].append(f"[{fname}, part {i}] {note.strip()}")
        print(f"  read part {n}/{len(parts)} {fname} #{i} ({first:.1f}s uncached)", flush=True)

    for q in questions:
        before, t0 = llm.usage.snapshot(), time.time()
        body = "\n\n".join(notes[q["id"]]) or "(no document part had anything relevant)"
        try:
            text = llm.chat(REDUCE_SYSTEM, f"Notes:\n{body}\n\nQuestion: {q['question']}", max_tokens=1500)
        except Exception as e:
            text = f"ERROR: {e}"
        secs = round(wait[q["id"]] + time.time() - t0, 1)
        d = delta(before, llm.usage.snapshot())
        u = {k: round(used[q["id"]][k] + d[k], 4) for k in d}
        run.add(q, to_answer(q["question"], text, files, secs, "fullread"), u, {"notes": len(notes[q["id"]])})
    run.save()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
