"""Field-level extraction accuracy: compare the extracted table with ground truth.

uv run python scripts/check_extraction.py [--store data/eval-enriched]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from docassist import Assistant, get_settings

COMPARE = [
    "contract_id",
    "contract_type",
    "counterparty",
    "counterparty_country",
    "effective_date",
    "end_date",
    "auto_renewal",
    "renewal_notice_days",
    "contract_value",
    "currency",
    "payment_terms_days",
    "termination_notice_days",
    "liability_cap",
    "governing_law",
]


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(b, bool):
        return bool(a) == b
    if isinstance(b, (int, float)):
        try:
            return abs(float(a) - float(b)) < 0.01
        except (TypeError, ValueError):
            return False
    norm = lambda s: str(s).lower().removeprefix("the ").strip(" .")  # noqa: E731
    return norm(a) == norm(b)


def check(records: list[dict], truth: list[dict]) -> dict:
    by_file = {r["filename"]: r for r in records}
    per_field = {f: [0, 0] for f in COMPARE}
    errors = []
    for t in truth:
        rec = by_file.get(t["filename"], {})
        for f in COMPARE:
            ok = same(rec.get(f), t[f])
            per_field[f][0] += ok
            per_field[f][1] += 1
            if not ok:
                errors.append({"file": t["filename"], "field": f, "expected": t[f], "got": rec.get(f), "scanned": t["scanned"]})
    total_ok = sum(v[0] for v in per_field.values())
    total = sum(v[1] for v in per_field.values())
    return {
        "field_accuracy": round(total_ok / total, 3),
        "fields_checked": total,
        "per_field": {f: round(ok / n, 3) for f, (ok, n) in per_field.items()},
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=Path("data/eval-enriched"))
    ap.add_argument("--truth", type=Path, default=Path("data/corpus/ground_truth.json"))
    ap.add_argument("--out", type=Path, default=Path("evals/results/extraction.json"))
    args = ap.parse_args()
    assistant = Assistant(get_settings().model_copy(update={"data_dir": args.store}))
    result = check(assistant.records(), json.loads(args.truth.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"field accuracy {result['field_accuracy']:.1%} over {result['fields_checked']} fields")
    for f, v in result["per_field"].items():
        print(f"  {f:<25} {v:.0%}")
    for e in result["errors"]:
        print(f"  x {e['file']:<32} {e['field']:<24} expected {e['expected']!r:<24} got {e['got']!r}{'  (scan)' if e['scanned'] else ''}")


if __name__ == "__main__":
    main()
