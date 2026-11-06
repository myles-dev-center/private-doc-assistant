"""Shared plumbing for the method comparison: load a golden set, score, save, stop at a budget.

run_eval.py (our RAG pipeline), run_fullread.py (give the model all documents)
and run_agent.py (the model searches with tools) all go through this, so the
three methods are scored and costed the same way.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import yaml

from docassist.evaluation import score, summarize
from docassist.parsing import SUPPORTED, ParsedDoc, parse_file
from docassist.pipeline import NOT_FOUND, Answer, Source

# Shared by the methods that answer from raw text (no excerpt numbers to cite).
ANSWER_RULES = """Rules:
- Use ONLY the documents. Do not use outside knowledge. Do not guess.
- Cite every factual statement with the file name and page in square brackets, like [file.pdf p. 3].
- For questions about many documents, check every document and name each match by its file name.
- If the documents do not contain the answer, reply exactly: NOT_FOUND
- Be concise."""


def public_endpoint(url: str) -> str:
    """Endpoint as recorded in results: local servers by address, anything else generically."""
    host = urlparse(url).hostname or ""
    return url if host in {"localhost", "127.0.0.1", "host.docker.internal"} else "OpenAI-compatible API"


def load_golden(path: Path, limit: int = 0) -> tuple[dict, list[dict]]:
    golden = yaml.safe_load(path.read_text(encoding="utf-8"))
    questions = golden["questions"]
    return golden["documents"], questions[:limit] if limit else questions


def load_corpus(folder: Path) -> list[ParsedDoc]:
    return [parse_file(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in SUPPORTED]


def doc_text(doc: ParsedDoc) -> str:
    return "\n".join(f"--- page {p.number} ---\n{p.text}" if p.number else p.text for p in doc.pages)


def to_answer(question: str, text: str, files: list[str], seconds: float, route: str) -> Answer:
    """Turn a free-text reply into an Answer; a cited source is any file name the reply mentions."""
    if "NOT_FOUND" in text and len(text.strip()) < 40:
        return Answer(question, NOT_FOUND, "not_found", seconds=seconds)
    cited = sorted(f for f in files if f in text)
    return Answer(question, text, route, sources=[Source(i, f, None, "") for i, f in enumerate(cited, 1)], seconds=seconds)


class Run:
    """Collects scores and per-question usage for one method, writes results.json."""

    def __init__(self, name: str, out: Path, documents: dict, meta: dict):
        self.name, self.out, self.documents = name, out, documents
        self.meta = meta | {"started": time.strftime("%Y-%m-%d %H:%M")}
        self.scores, self.per_q = [], []

    def add(self, q: dict, ans: Answer, usage: dict, extra: dict | None = None) -> None:
        s = score(q, ans, self.documents)
        self.scores.append(s)
        self.per_q.append({"qid": q["id"], "seconds": ans.seconds} | usage | (extra or {}))
        mark = "ok " if s.correct else ("HAL" if s.hallucinated else "-- ")
        print(f"  {mark} {q['id']} {ans.seconds:>6.1f}s ${usage.get('usd', 0):.3f}  {q['question'][:64]}", flush=True)

    def spent(self) -> float:
        return sum(p.get("usd", 0) for p in self.per_q)

    def save(self, setup: dict | None = None) -> dict:
        summary = summarize(self.scores)
        summary["usage"] = {"setup": setup or {}, "questions": per_question_totals(self.per_q), "per_question": self.per_q}
        run = self.meta | {"configs": {self.name: {"settings": {}, "summary": summary, "scores": [asdict(s) for s in self.scores]}}}
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "results.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
        print(f"=> {self.name}: accuracy {summary['accuracy']}, hallucination {summary['hallucination_rate']}, spend ${self.spent():.2f}")
        return summary


def per_question_totals(per_q: list[dict]) -> dict:
    keys = ("calls", "input", "cached", "output", "embed_tokens", "usd")
    return {k: round(sum(p.get(k, 0) for p in per_q), 4) for k in keys}


def delta(before: dict, after: dict) -> dict:
    return {k: round(after[k] - before[k], 4) for k in after}
