"""Hybrid retrieval: BM25 (SQLite FTS5) + cosine similarity, fused with RRF."""

from __future__ import annotations

import re

import numpy as np

from .llm import LLM
from .store import Store, StoredChunk

_WORD = re.compile(r"[A-Za-z0-9]+")
# Chinese, Japanese and Korean script: no spaces between words, so FTS5's unicode61
# tokenizer would index a whole sentence as one token.
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_CJK_RUN = re.compile(_CJK.pattern + "+")


def fts_text(text: str) -> str:
    """Text as indexed for keyword search. CJK characters become single-character
    tokens and a CJK phrase becomes a phrase query, a common, dictionary-free way to
    search Chinese with FTS5. Text without CJK is unchanged."""
    if not _CJK.search(text):
        return text
    return _CJK_RUN.sub(lambda m: " " + " ".join(m.group()) + " ", text)


def unspace_cjk(text: str) -> str:
    """Undo fts_text for CJK runs (text without CJK is returned unchanged)."""
    if not _CJK.search(text):
        return text
    return re.sub(r" ?((?:" + _CJK.pattern + r" )*" + _CJK.pattern + r") ?", lambda m: m.group(1).replace(" ", ""), text)


_STOP = set(
    "a an and are as at be by can do does for from has have how i in is it its many much of on or our "
    "the their there this to under was we what when where which who with within".split()
)


def fts_query(question: str) -> str:
    """Build a safe FTS5 query: OR of quoted terms, plus phrases for identifiers like AP-SA-2024-017."""
    terms: list[str] = []
    for ident in re.findall(r"\b[A-Za-z0-9]+(?:[-/.][A-Za-z0-9]+)+\b", question):
        parts = _WORD.findall(ident)
        if any(p.isdigit() for p in parts):
            terms.append('"' + " ".join(parts) + '"')
    words = [w for w in _WORD.findall(question) if w.lower() not in _STOP]
    terms.extend(f'"{w}"' for w in dict.fromkeys(words))
    for run in dict.fromkeys(_CJK_RUN.findall(question)):
        # Overlapping two-character phrases: "付款条件" -> "付 款" "款 条" "条 件"
        pairs = [run[i : i + 2] for i in range(len(run) - 1)] or [run]
        terms.extend('"' + " ".join(p) + '"' for p in dict.fromkeys(pairs))
    return " OR ".join(terms)


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class Retriever:
    def __init__(self, store: Store, llm: LLM, *, hybrid: bool = True, candidates: int = 30):
        self.store, self.llm, self.hybrid, self.candidates = store, llm, hybrid, candidates

    def vector_search(self, question: str, limit: int, doc_ids: set[int] | None = None) -> list[int]:
        ids, docs, mat = self.store.vectors()
        if doc_ids:
            mask = np.isin(docs, list(doc_ids))
            ids, mat = ids[mask], mat[mask]
        if len(ids) == 0:
            return []
        q = self.llm.embed([question], query=True)[0]
        sims = mat @ q
        top = np.argsort(-sims)[:limit]
        return [int(ids[i]) for i in top]

    def search(self, question: str, top_k: int, doc_ids: set[int] | None = None) -> list[StoredChunk]:
        """Top chunks for the question, optionally restricted to some documents."""
        vec = self.vector_search(question, self.candidates, doc_ids)
        if not self.hybrid:
            return self.store.get_chunks(vec[:top_k])
        bm25 = self.store.fts_search(fts_query(question), self.candidates, doc_ids)
        return self.store.get_chunks(rrf([bm25, vec])[:top_k])
