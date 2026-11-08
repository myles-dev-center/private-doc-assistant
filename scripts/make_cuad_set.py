"""Build a real-contract test set from CUAD: 30 PDFs + a golden question set.

CUAD (Contract Understanding Atticus Dataset, The Atticus Project, CC BY 4.0)
is 510 commercial agreements from public SEC filings, labeled by lawyers.
Download CUAD_v1.zip from https://zenodo.org/records/4595826 and unzip it.

    uv run python scripts/make_cuad_set.py --cuad path/to/CUAD_v1
    # a second, disjoint set for a blind re-test:
    uv run python scripts/make_cuad_set.py --cuad path/to/CUAD_v1 --seed 2027 --exclude data/corpus-cuad \
        --out data/corpus-cuad-holdout --golden evals/cuad-golden-holdout.yaml

--extended adds two kinds of questions that real users ask and the template
does not: lookups that name an agreement by a short name ("our agreement with
Jenner"), and lists that name an agreement ("all agreements governed by the same
law as the agreement between A and B"). Without the flag, a seed produces exactly
the set it always has.

Unlike the synthetic corpus, these documents share no template and have no
contract numbers; questions name an agreement by its parties. Expected answers
come from the CUAD labels, and a lookup question is only asked when its answer
is written literally in the contract (a date the model would have to compute
from "five years after the Effective Date" is not used), so that a correct
answer can be scored deterministically.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import yaml
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docassist.evaluation import _norm, dates_in, numbers_in  # noqa: E402

SUFFIX = re.compile(
    r"[,\s]+(inc|incorporated|corp|corporation|co|company|llc|l\.l\.c|ltd|limited|lp|l\.p|plc|n\.v|s\.a|ag|gmbh|holdings?)\.?$",
    re.I,
)


def parties(label: str) -> list[str]:
    out = []
    for p in label.replace("\xa0", " ").split(";"):
        p = re.sub(r"\(.*?\)|[\"“”�]", " ", p)
        p = re.split(r"\bd/b/a\b|\bformerly\b", p, flags=re.I)[0]
        p = re.sub(r"\s+", " ", p).strip(" ,.")
        # skip collective short names like "Heritage and Premier" or "OICL and QIWI"
        if len(p) >= 3 and not re.fullmatch(r"(?i)(\w+ ){0,2}and \w+( \w+)?", p):
            out.append(p)
    return out


def usable_party(name: str, text: str) -> bool:
    """A name a user could ask about: not redacted, not a role ("Agent"), not a sentence, and in the text."""
    return (
        5 <= len(name) <= 70
        and not re.search(r"[\[\]*]|\b(each|herein|who|its subsidiaries|and)\b", name, re.I)
        and len(name.split()) >= 2
        and _norm(name) in _norm(text)
    )


def aliases(name: str) -> list[str]:
    """The name, and the name without a trailing legal-form suffix (Adams Golf, Ltd. -> Adams Golf)."""
    out = [name]
    short = SUFFIX.sub("", name).strip(" ,.")
    while short != out[-1]:
        out.append(short)
        short = SUFFIX.sub("", short).strip(" ,.")
    return [a for a in dict.fromkeys(out) if len(a) >= 5]


def iso(label: str) -> str | None:
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", label.strip())
    if not m:
        return None
    mo, d, y = (int(x) for x in m.groups())
    y = y if y > 99 else (2000 + y if y <= 30 else 1900 + y)
    return f"{y:04d}-{mo:02d}-{d:02d}"


def title(label: str) -> str:
    t = re.sub(r"\s+", " ", label).strip(' (“"')
    return t.title() if t.isupper() else t


def pdf_has_text(path: Path, txt_len: int) -> bool:
    try:
        text = "".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return False
    return len(text) >= 0.6 * txt_len


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuad", type=Path, required=True, help="unzipped CUAD_v1 folder")
    ap.add_argument("--out", type=Path, default=Path("data/corpus-cuad"))
    ap.add_argument("--golden", type=Path, default=Path("evals/cuad-golden.yaml"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--exclude", type=Path, action="append", default=[], help="folder of contracts already used; never pick them again")
    ap.add_argument("--extended", action="store_true", help="add short-name lookups and lists that name an agreement")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    pdfs = {p.name.lower(): p for p in (args.cuad / "full_contract_pdf").rglob("*") if p.suffix.lower() == ".pdf"}
    rows = list(csv.DictReader(open(args.cuad / "master_clauses.csv", encoding="utf-8")))
    rows.sort(key=lambda r: r["Filename"])
    rng.shuffle(rows)

    # Pick contracts of ordinary length (about 3-15 pages) with a text layer and
    # parties that are not shared with another picked contract.
    used = {p.name.lower() for d in args.exclude for p in d.iterdir()}
    picked, seen = [], set()
    for r in rows:
        if r["Filename"].lower() in used:
            continue
        pdf = pdfs.get(r["Filename"].lower())
        txt = args.cuad / "full_contract_txt" / (Path(r["Filename"]).stem + ".txt")
        if not pdf or not txt.exists() or re.search(r"[^\w\s.,()&-]", pdf.name):
            continue
        text = txt.read_text(encoding="utf-8", errors="ignore")
        names = parties(r["Parties-Answer"])
        keys = {_norm(a) for n in names for a in aliases(n)}
        if not (8_000 <= len(text) <= 60_000) or len(names) < 2 or keys & seen:
            continue
        if not (usable_party(names[0], text) and usable_party(names[1], text)):
            continue
        if not pdf_has_text(pdf, len(text)):
            continue
        seen |= keys
        picked.append({"row": r, "pdf": pdf, "text": text, "parties": [n for n in names if usable_party(n, text)]})
        if len(picked) == args.n:
            break

    args.out.mkdir(parents=True, exist_ok=True)
    for c in picked:
        shutil.copyfile(c["pdf"], args.out / c["pdf"].name)

    def name(c) -> str:
        return f"the {title(c['row']['Document Name-Answer'])} between {c['parties'][0]} and {c['parties'][1]}"

    documents = {c["pdf"].name: [c["pdf"].name] + [a for n in c["parties"] for a in aliases(n)] for c in picked}

    # -------------------------------------------------------------- lookups
    candidates: dict[str, list[dict]] = {"law": [], "date": [], "expiry": [], "notice": []}
    for c in picked:
        r, text, src = c["row"], c["text"], c["pdf"].name
        law = [x.strip() for x in r["Governing Law-Answer"].split(";") if x.strip()]
        opts = [o for x in law for o in dict.fromkeys([x, x.split(",")[0].strip()]) if o.lower() in text.lower()]
        if opts:
            candidates["law"].append(
                {"question": f"Which law governs {name(c)}?", "expected": {"kind": "text", "value": opts}, "source": src}
            )
        for key, label, q in [
            ("date", "Agreement Date-Answer", "What is the date of {}?"),
            ("expiry", "Expiration Date-Answer", "When does {} expire?"),
        ]:
            d = iso(r[label])
            if d and d in dates_in(text):
                candidates[key].append({"question": q.format(name(c)), "expected": {"kind": "date", "value": d}, "source": src})
        m = re.fullmatch(r"(\d+)\)?\s*days", r["Notice Period To Terminate Renewal- Answer"].strip())
        if m and float(m.group(1)) in numbers_in(text):
            candidates["notice"].append(
                {
                    "question": f"How many days of notice are required to prevent the renewal of {name(c)}?",
                    "expected": {"kind": "number", "value": int(m.group(1))},
                    "source": src,
                }
            )
    lookups: list[dict] = []
    quota = {"law": 8, "date": 8, "expiry": 6, "notice": 4}
    for key, n in quota.items():
        rng.shuffle(candidates[key])
        lookups += [dict(q, tag=key) for q in candidates[key][:n]]
    # top up from whatever is left if a category ran short
    rest = [dict(q, tag=k) for k in quota for q in candidates[k][quota[k] :]]
    rng.shuffle(rest)
    lookups += rest[: 26 - len(lookups)]

    # ------------------------------------------------------------ aggregates
    def files(pred) -> list[str]:
        return sorted(c["pdf"].name for c in picked if pred(c["row"]))

    def law_is(state: str):
        return lambda r: state.lower() in r["Governing Law-Answer"].lower()

    laws = Counter(x.strip() for c in picked for x in c["row"]["Governing Law-Answer"].split(";") if x.strip())
    aggregates = []
    for state, n in laws.most_common(3):
        if n >= 2:
            aggregates.append((f"List all agreements governed by the laws of {state}.", files(law_is(state))))
    clauses = [
        ("Non-Compete-Answer", "Which agreements contain a non-compete clause?"),
        ("Exclusivity-Answer", "Which agreements contain an exclusivity commitment?"),
        ("Termination For Convenience-Answer", "Which agreements can be terminated for convenience, without cause?"),
        ("Audit Rights-Answer", "Which agreements give a party audit rights?"),
        ("Most Favored Nation-Answer", "Which agreements contain a most-favored-nation clause?"),
    ]
    for col, q in clauses:
        aggregates.append((q, files(lambda r, col=col: r[col] == "Yes")))
    aggregates.append(
        ("Which agreements are perpetual, with no fixed end date?", files(lambda r: r["Expiration Date-Answer"].lower() == "perpetual"))
    )
    if laws:
        state, n = laws.most_common(1)[0]
        aggregates.append((f"How many agreements are governed by the laws of {state}?", n))
    agg_qs = []
    for q, exp in aggregates[:10]:
        e = {"kind": "number", "value": exp} if isinstance(exp, int) else {"kind": "docs", "value": exp}
        agg_qs.append({"type": "aggregate", "tag": "aggregate", "question": q, "expected": e})

    # --------------------------------------------------------- unanswerable
    unans = []
    for c in picked:
        r, text = c["row"], c["text"].lower()
        if not r["Governing Law-Answer"].strip() and "governed by" not in text and "laws of" not in text:
            unans.append(f"Which law governs {name(c)}?")
        if not r["Renewal Term-Answer"].strip() and "renew" not in text:
            unans.append(f"How many days of notice are required to prevent the automatic renewal of {name(c)}?")
    rng.shuffle(unans)
    unans = unans[:5]
    c1, c2, c3 = rng.sample(picked, 3)
    unans += [
        f"What was the share price of {c1['parties'][0]} on the date of {name(c1)}?",
        f"How many employees does {c2['parties'][1]} have according to {name(c2)}?",
        f"What is the late payment interest rate in {name(c3)}?"
        if "interest" not in c3["text"].lower()
        else f"Who was the CEO of {c3['parties'][0]} when {name(c3)} was signed?",
    ]

    shorts, named = extended(picked, candidates, files, law_is, args.seed) if args.extended else ([], [])

    questions = []
    for i, q in enumerate(lookups, 1):
        questions.append(
            {
                "id": f"L{i:02d}",
                "type": "lookup",
                "tag": q["tag"],
                "question": q["question"],
                "expected": q["expected"],
                "source": q["source"],
            }
        )
    for i, q in enumerate(agg_qs, 1):
        questions.append({"id": f"A{i:02d}", **q})
    for i, q in enumerate(unans, 1):
        questions.append({"id": f"U{i:02d}", "type": "unanswerable", "tag": "unanswerable", "question": q, "expected": {"kind": "refusal"}})
    for i, q in enumerate(shorts, 1):
        questions.append({"id": f"S{i:02d}", "type": "lookup", **q})
    for i, q in enumerate(named, 1):
        questions.append({"id": f"N{i:02d}", "type": "aggregate", "tag": "named-list", **q})

    header = (
        "# Generated by scripts/make_cuad_set.py from CUAD v1 (The Atticus Project, CC BY 4.0,\n"
        "# https://www.atticusprojectai.org/cuad) - do not edit by hand.\n"
    )
    golden = {"documents": documents, "questions": questions}
    args.golden.write_text(header + yaml.safe_dump(golden, sort_keys=False, allow_unicode=True, width=140), encoding="utf-8")
    counts = Counter(q["type"] for q in questions)
    print(f"{len(picked)} contracts -> {args.out}; golden {dict(counts)} -> {args.golden}")
    print("lookup tags:", dict(Counter(q["tag"] for q in lookups)))


_NOT_A_NAME = set(
    "agreement company corporation incorporated limited holdings holding group international services the and of "
    "inc corp llc ltd plc gmbh trust trustees family investors investment investments capital management partners "
    "racing pharmaceuticals pharmaceutical biosciences technologies technology systems solutions software media "
    "communications energy resources industries enterprises products global national american united first bank "
    "healthcare health medical sports entertainment marketing network networks digital financial equity city county "
    "state north south east west new mutual insurance".split()
)


SHORT_TEMPLATES = {
    "law": "Which law governs our agreement with {}?",
    "date": "When was the {} agreement signed?",
    "expiry": "When does the agreement with {} expire?",
    "notice": "How many days of notice does the {} agreement require to stop it from renewing?",
}


def short_name(c: dict, picked: list[dict]) -> str | None:
    """How people shorten a party name: its first distinctive word ("Movado Group, Inc" ->
    "Movado"), if no other picked agreement's parties use that word."""

    def words(name: str) -> list[str]:
        return [w for w in re.findall(r"[A-Za-z][A-Za-z'&-]+", name) if len(w) >= 4 and w.lower() not in _NOT_A_NAME]

    others = {w.lower() for o in picked if o is not c for n in o["parties"] for w in words(n)}
    for name in c["parties"]:
        first = words(name)[:1]
        if first and first[0].lower() not in others:
            return first[0]
    return None


def extended(picked, candidates, files, law_is, seed) -> tuple[list[dict], list[dict]]:
    """Short-name lookups (6) and lists that name an agreement (3); a separate random
    stream, so the rest of the set is identical with and without --extended."""
    rng = random.Random(f"{seed}-extended")
    by_src = {c["pdf"].name: c for c in picked}
    pool = [(key, q) for key, qs in candidates.items() for q in qs]
    rng.shuffle(pool)
    shorts, used = [], set()
    for key, q in pool:
        c = by_src[q["source"]]
        name = short_name(c, picked)
        if not name or q["source"] in used:
            continue
        used.add(q["source"])
        shorts.append(
            {"tag": f"short-{key}", "question": SHORT_TEMPLATES[key].format(name), "expected": q["expected"], "source": q["source"]}
        )
        if len(shorts) == 6:
            break

    def name(c) -> str:
        return f"the {title(c['row']['Document Name-Answer'])} between {c['parties'][0]} and {c['parties'][1]}"

    named = []
    order = picked[:]
    rng.shuffle(order)
    for c in order:
        law = [x.strip() for x in c["row"]["Governing Law-Answer"].split(";") if x.strip()]
        if len(law) != 1:
            continue
        same = files(law_is(law[0]))
        if len(same) < 2:
            continue
        named.append(
            {
                "question": f"List all agreements governed by the same law as {name(c)}, including that one.",
                "expected": {"kind": "docs", "value": same},
            }
        )
        if len(named) == 3:
            break
    return shorts, named


if __name__ == "__main__":
    main()
