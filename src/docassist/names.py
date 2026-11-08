"""Recognize the documents a question names - by full name, without the legal form, or by a short name.

Users rarely type "Bruce Jenner, Individual" or "Golfers Incorporated"; they write
"the Jenner deal" or "Golfers". A full name (or the name without its legal form) is a
strong match; one distinctive word of a name, used by no other document, is a weak
match that the router confirms or rejects.
"""

from __future__ import annotations

import re
from collections import Counter

_LEGAL = set(
    "inc incorporated corp corporation co company llc lc ltd limited lp llp plc nv bv sa ag gmbh kg se srl spa pty "
    "holding holdings group the a an individual".split()
)
# Words that can be part of a name but say nothing about which document is meant.
_GENERIC = set(
    "agreement contract amendment company international services service systems system technologies technology "
    "solutions global national american america united general new first bank trust partners partner capital "
    "management industries enterprises resources products communications media health healthcare pharmaceuticals "
    "energy software network networks digital financial investment investments north south east west state states "
    "city county law laws date term notice renewal clause party parties between with which what when does right "
    "rights about from that this have each other their these those".split()
)


def norm(text: str) -> str:
    """Lower-case words separated by single spaces, padded so ' x ' matches whole words."""
    return " " + " ".join(re.sub(r"[\W_]+", " ", text.lower()).split()) + " "


def variants(name: str) -> set[str]:
    """Full-name forms that are safe to look for in a question (normalized)."""
    out = set()
    base = re.sub(r"\(.*?\)", " ", name)
    for form in {base, base.split(",")[0]}:
        words = norm(form).split()
        while words and words[-1] in _LEGAL:  # "Adams Golf, Ltd." -> "adams golf"
            words.pop()
        while words and words[0] in {"the", "a", "an"}:
            words.pop(0)
        v = " ".join(words)
        if len(v) >= 6 or len(words) >= 2:
            out.add(f" {v} ")
    full = norm(name)
    if len(full.strip()) >= 6:
        out.add(full)
    return out


def tokens(name: str) -> set[str]:
    # Only the name itself: "The Right Solution, a Nevada Corporation" must not be
    # recognized from "Nevada".
    words = norm(re.sub(r"\(.*?\)", " ", name).split(",")[0]).split()
    return {w for w in words if len(w) >= 4 and w.isalpha() and w not in _LEGAL and w not in _GENERIC}


def match(question: str, names: dict[str, list[str]]) -> dict[str, str]:
    """{document: "full" | "partial"} for the documents a question names.

    `names` maps each document to its identifying values (party names, contract numbers).
    """
    q = norm(question)
    found: dict[str, str] = {}
    for doc, values in names.items():
        if any(v in q for value in values for v in variants(value)):
            found[doc] = "full"
    # A word that appears in the names of exactly one document identifies it.
    owners = Counter(t for values in names.values() for t in set().union(*(tokens(v) for v in values)))
    for doc, values in names.items():
        if doc in found:
            continue
        own = set().union(*(tokens(v) for v in values)) if values else set()
        if any(owners[t] == 1 and f" {t} " in q for t in own):
            found[doc] = "partial"
    return found
