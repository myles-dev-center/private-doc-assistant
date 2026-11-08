"""Summarize the model-combination comparison (evals/cmp4/<combo>/<set>.json) as Markdown + JSON.

    uv run python scripts/combos_report.py [--cuad path/to/CUAD_v1]

All sets are pooled per combination. Accuracy comes with a 95% Wilson interval;
pairs of combinations are compared on the same questions with an exact McNemar test
(only questions one got right and the other wrong count). With --cuad, the extracted
table of each chat model is also checked field by field against the lawyers' labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from itertools import combinations
from pathlib import Path

CMP = Path("evals/cmp4")
TYPES = [("lookup", "One agreement"), ("aggregate", "Across agreements"), ("unanswerable", "Not in the documents")]
TAGS = [("short", "Short name"), ("named-list", "List naming an agreement")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided p-value: b = only A right, c = only B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def load() -> dict[str, dict]:
    combos: dict[str, dict] = {}
    for f in sorted(CMP.glob("*/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        c = combos.setdefault(r["combo"], {"label": r["label"], "chat": r["chat_model"], "embed": r["embed_model"], "sets": {}, "rows": []})
        c["sets"][r["set"]] = r
        c["rows"] += [dict(s, set=r["set"]) for s in r["scores"]]
    return combos


def rate(rows: list[dict], key: str = "correct") -> tuple[int, int]:
    return sum(bool(r[key]) for r in rows), len(rows)


def pct(k: int, n: int) -> str:
    return f"{k / n:.0%}" if n else "–"


def stats(c: dict) -> dict:
    rows = c["rows"]
    k, n = rate(rows)
    lo, hi = wilson(k, n)
    out = {
        "label": c["label"],
        "chat_model": c["chat"],
        "embed_model": c["embed"],
        "sets": sorted(c["sets"]),
        "questions": n,
        "accuracy": round(k / n, 3) if n else None,
        "ci95": [round(lo, 3), round(hi, 3)],
        "hallucination": round(rate(rows, "hallucinated")[0] / n, 3) if n else None,
    }
    for t, _ in TYPES:
        sub = [r for r in rows if r["qtype"] == t]
        out[t] = list(rate(sub))
    for tag, _ in TAGS:
        sub = [r for r in rows if r["tag"].startswith(tag)]
        out[tag] = list(rate(sub))
    secs = sorted(r["seconds"] for r in rows)
    out["median_s"] = round(statistics.median(secs), 1) if secs else None
    out["p90_s"] = round(secs[int(0.9 * (len(secs) - 1))], 1) if secs else None
    ingest = [s["usage"]["ingest"] for s in c["sets"].values()]
    q_usd = sum(s["usage"]["questions"].get("usd", 0) for s in c["sets"].values())
    out["usd_per_question"] = round(q_usd / n, 4) if n else None
    out["usd_questions"] = round(q_usd, 2)
    out["usd_ingest"] = round(sum(u.get("usd", 0) for u in ingest), 2)
    out["ingest_reused_extraction"] = any(u.get("reused_extraction_from") for u in ingest)
    return out


def extraction_accuracy(cuad: Path, combos: dict, stores: Path | None = None) -> dict[str, dict]:
    """Field-level accuracy of the extracted table against CUAD labels, per chat model."""
    sys.path.insert(0, str(Path(__file__).parent))
    from check_cuad_extraction import BOOLEANS
    from make_cuad_set import iso

    from docassist.store import Store

    labels = {r["Filename"].lower(): r for r in csv.DictReader(open(cuad / "master_clauses.csv", encoding="utf-8"))}
    out: dict[str, dict] = {}
    for c in combos.values():
        if c["chat"] in out:
            continue
        tally: dict[str, list[int]] = {}
        for r in c["sets"].values():
            lib = stores / r["set"] / r["library"] if stores else None
            if not lib or not (lib / "docassist.db").exists():
                continue
            for rec in Store(lib / "docassist.db").extractions("agreement"):
                lab = labels.get(rec["filename"].lower())
                if not lab:
                    continue

                def t(field: str, ok: bool, tally: dict = tally) -> None:
                    tally.setdefault(field, [0, 0])
                    tally[field][0] += ok
                    tally[field][1] += 1

                law = [x.strip() for x in lab["Governing Law-Answer"].split(";") if x.strip()]
                if law:
                    t("governing_law", any(x.split(",")[0].lower() in (rec.get("governing_law") or "").lower() for x in law))
                d = iso(lab["Agreement Date-Answer"])
                if d:
                    t("agreement_date", d in (rec.get("agreement_date"), rec.get("effective_date")))
                for field, col in BOOLEANS.items():
                    t(field, bool(rec.get(field)) == (lab[col] == "Yes"))
        if tally:
            ok, n = sum(v[0] for v in tally.values()), sum(v[1] for v in tally.values())
            out[c["chat"]] = {"fields": tally, "total": [ok, n]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuad", type=Path, help="unzipped CUAD_v1, for extraction accuracy")
    ap.add_argument("--stores", type=Path, help="the --stores folder of run_combos.py, to find the libraries")
    args = ap.parse_args()
    combos = load()
    if not combos:
        raise SystemExit(f"no results in {CMP}")
    summary = {name: stats(c) for name, c in combos.items()}

    head = "Combination | Accuracy (95% CI) | One agreement | Across | Not in docs | Short name | Named list"
    head += " | Hallucination | Median s | p90 s | $ / question | Ingest $"
    lines = [f"| {head} |", "|---|" + "---:|" * 11]
    for s in summary.values():
        lines.append(
            f"| {s['label']} | **{s['accuracy']:.0%}** ({s['ci95'][0]:.0%}–{s['ci95'][1]:.0%}) | "
            + " | ".join(pct(*s[t]) for t, _ in TYPES)
            + " | "
            + " | ".join(f"{pct(*s[tag])} ({s[tag][0]}/{s[tag][1]})" for tag, _ in TAGS)
            + f" | {s['hallucination']:.0%} | {s['median_s']} | {s['p90_s']} | ${s['usd_per_question']:.4f} | ${s['usd_ingest']:.2f} |"
        )
    print("\n".join(lines))

    # Pairwise, on the same questions
    print("\n| A vs B | A right, B wrong | B right, A wrong | p (McNemar) |\n|---|---:|---:|---:|")
    pairs = []
    for a, b in combinations(combos, 2):
        ra = {(r["set"], r["qid"]): r["correct"] for r in combos[a]["rows"]}
        rb = {(r["set"], r["qid"]): r["correct"] for r in combos[b]["rows"]}
        common = ra.keys() & rb.keys()
        only_a = sum(ra[q] and not rb[q] for q in common)
        only_b = sum(rb[q] and not ra[q] for q in common)
        p = mcnemar(only_a, only_b)
        pairs.append({"a": a, "b": b, "only_a": only_a, "only_b": only_b, "p": round(p, 4), "questions": len(common)})
        print(f"| {summary[a]['label']} vs {summary[b]['label']} | {only_a} | {only_b} | {p:.3f}{' *' if p < 0.05 else ''} |")

    extraction = extraction_accuracy(args.cuad, combos, args.stores) if args.cuad else {}
    for model, e in extraction.items():
        print(f"\nExtraction vs lawyer labels, {model}: {e['total'][0]}/{e['total'][1]} = {e['total'][0] / e['total'][1]:.0%}")

    (CMP / "summary.json").write_text(json.dumps({"combos": summary, "pairs": pairs, "extraction": extraction}, indent=2), encoding="utf-8")
    print(f"\nwrote {CMP / 'summary.json'}")


if __name__ == "__main__":
    main()
