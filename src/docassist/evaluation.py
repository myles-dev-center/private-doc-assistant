"""Accuracy evaluation against a golden question set.

Scoring is deterministic (no LLM judge), so results are reproducible and cheap:

* lookup        - the expected value must appear in the answer (dates and numbers
                  are normalized, so "31 March 2027" matches 2027-03-31), and the
                  answer must cite the right source file when it uses retrieval.
* aggregate     - recall and precision of the expected contract IDs in the answer
                  (or the SQL result), or the expected number for counts and sums.
                  Documents without an ID scheme are recognized by their aliases
                  (filename, party names) from the golden file's `documents` map.
* unanswerable  - the assistant must say the information is not there. Rejecting the
                  question's premise counts too: "it does not renew automatically" is
                  the right answer to "how much notice prevents its automatic renewal?".
                  Any answer that asserts something instead counts as a hallucination.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from .pipeline import NOT_FOUND, Answer

REFUSAL = re.compile(
    r"not (?:contain|include|mention|specif|state|provide|found|available)|no (?:information|mention|such|documents? match|matching)"
    r"|does not (?:appear|exist)|cannot find|could not find|isn't (?:mentioned|specified)|is not (?:mentioned|specified|stated)"
    # the premise of a renewal question is false
    r"|(?:does not|will not|doesn't|won't|not) (?:automatically |auto-?)?renew\b"
    r"|no automatic renewal|not subject to (?:automatic |auto-?)renewal",
    re.I,
)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


@dataclass
class Score:
    qid: str
    qtype: str
    tag: str
    question: str
    expected: object
    answer: str
    route: str
    correct: bool
    hallucinated: bool = False
    refused: bool = False
    citation_ok: bool | None = None
    recall: float | None = None
    precision: float | None = None
    seconds: float = 0.0
    detail: dict = field(default_factory=dict)


def dates_in(text: str) -> set[str]:
    found: set[str] = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    for d, m, y in re.findall(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", text):
        found.add(f"{y}-{int(m):02d}-{int(d):02d}")
    for d, mon, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? ({_MONTHS}),? (\d{{4}})", text):
        found.add(dt.datetime.strptime(f"{d} {mon} {y}", "%d %B %Y").date().isoformat())
    for mon, d, y in re.findall(rf"\b({_MONTHS}) (\d{{1,2}})(?:st|nd|rd|th)?,? (\d{{4}})", text):
        found.add(dt.datetime.strptime(f"{d} {mon} {y}", "%d %B %Y").date().isoformat())
    for d, mon, y in re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? day of ({_MONTHS}),? (\d{{4}})", text):
        found.add(dt.datetime.strptime(f"{d} {mon} {y}", "%d %B %Y").date().isoformat())
    # Slashes are read US-style (month/day/year), dots European (day.month.year).
    for m, d, y in re.findall(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", text):
        try:
            found.add(dt.date(int(y), int(m), int(d)).isoformat())
        except ValueError:
            pass
    return found


def numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*(million|m\b)", text, re.I):
        found.add(round(float(m.group(1).replace(",", ".")) * 1_000_000, 2))
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        try:
            found.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return found


def ids_in(text: str) -> set[str]:
    return set(re.findall(r"AP-[A-Z]+-\d{4}-\d{3}", text))


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


def docs_in(text: str, documents: dict[str, list[str]]) -> set[str]:
    """Documents mentioned in the text by any of their aliases (whole-word, case-insensitive)."""
    t = _norm(text)
    return {key for key, aliases in documents.items() if any(_norm(a) in t for a in aliases)}


def is_refusal(ans: Answer) -> bool:
    return ans.answer.strip() == NOT_FOUND or ans.route == "not_found" or bool(REFUSAL.search(ans.answer))


def score(q: dict, ans: Answer, documents: dict[str, list[str]] | None = None) -> Score:
    exp = q["expected"]
    s = Score(q["id"], q["type"], q.get("tag", ""), q["question"], exp.get("value"), ans.answer, ans.route, False, seconds=ans.seconds)
    s.refused = is_refusal(ans)
    kind = exp["kind"]

    if kind == "refusal":
        s.correct = s.refused
        s.hallucinated = not s.refused
        return s

    # The SQL result table is part of the answer the user sees.
    haystack = ans.answer
    if ans.rows:
        haystack += "\n" + "\n".join(" ".join(str(v) for v in r) for r in ans.rows)

    if kind in ("ids", "docs"):
        expected = set(exp["value"])
        got = ids_in(haystack) if kind == "ids" else docs_in(haystack, documents or {})
        tp = len(expected & got)
        s.recall = tp / len(expected) if expected else 1.0
        s.precision = tp / len(got) if got else (1.0 if not expected else 0.0)
        s.correct = s.recall == 1.0 and s.precision == 1.0
        s.detail = {"missing": sorted(expected - got), "extra": sorted(got - expected)}
    elif kind == "date":
        s.correct = exp["value"] in dates_in(haystack)
    elif kind == "number":
        s.correct = round(float(exp["value"]), 2) in numbers_in(haystack)
    elif kind == "text":
        options = exp["value"] if isinstance(exp["value"], list) else [exp["value"]]
        s.correct = any(o.lower() in haystack.lower() for o in options)

    if q["type"] == "lookup" and ans.route in ("search", "document"):
        s.citation_ok = any(src.filename == q.get("source") for src in ans.sources)
    # A wrong, confident answer is a hallucination; a refusal is just a miss.
    s.hallucinated = not s.correct and not s.refused
    return s


def summarize(scores: list[Score]) -> dict:
    def rate(items, attr="correct"):
        return round(sum(bool(getattr(x, attr)) for x in items) / len(items), 3) if items else None

    by_type = {}
    for t in ("lookup", "aggregate", "unanswerable"):
        items = [s for s in scores if s.qtype == t]
        by_type[t] = {"n": len(items), "accuracy": rate(items)}
    answerable = [s for s in scores if s.qtype != "unanswerable"]
    cited = [s for s in scores if s.citation_ok is not None]
    agg = [s for s in scores if s.recall is not None]
    return {
        "n": len(scores),
        "accuracy": rate(scores),
        "by_type": by_type,
        "hallucination_rate": rate(scores, "hallucinated"),
        "answerable_hallucination_rate": rate(answerable, "hallucinated"),
        "refusal_on_unanswerable": by_type["unanswerable"]["accuracy"],
        "citation_accuracy": rate(cited, "citation_ok"),
        "aggregate_recall": round(sum(s.recall for s in agg) / len(agg), 3) if agg else None,
        "median_seconds": sorted(s.seconds for s in scores)[len(scores) // 2] if scores else None,
    }
