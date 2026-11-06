"""Field-level extraction accuracy on CUAD: the extracted table vs the lawyers' labels.

    uv run python scripts/check_cuad_extraction.py --cuad path/to/CUAD_v1 \
        --store data/eval-enriched-cmp-local --corpus data/corpus-cuad-holdout2

Compares the fields of schemas/agreement.yaml that have a direct CUAD label.
Dates count only when CUAD gives a full date; governing law matches if the
extracted value names the labeled jurisdiction.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_cuad_set import iso  # noqa: E402

from docassist import Assistant, get_settings  # noqa: E402

BOOLEANS = {
    "non_compete": "Non-Compete-Answer",
    "exclusivity": "Exclusivity-Answer",
    "termination_for_convenience": "Termination For Convenience-Answer",
    "cap_on_liability": "Cap On Liability-Answer",
    "audit_rights": "Audit Rights-Answer",
    "most_favored_nation": "Most Favored Nation-Answer",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuad", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="write the per-field result as JSON")
    args = ap.parse_args()

    labels = {r["Filename"].lower(): r for r in csv.DictReader(open(args.cuad / "master_clauses.csv", encoding="utf-8"))}
    settings = get_settings().model_copy(update={"data_dir": args.store, "schema_path": Path("schemas/agreement.yaml")})
    records = Assistant(settings).records()
    files = {p.name for p in args.corpus.iterdir()}
    records = [r for r in records if r["filename"] in files]

    result: dict[str, list[int]] = {}

    def tally(field: str, ok: bool) -> None:
        result.setdefault(field, [0, 0])
        result[field][0] += ok
        result[field][1] += 1

    for rec in records:
        lab = labels[rec["filename"].lower()]
        law = [x.strip() for x in lab["Governing Law-Answer"].split(";") if x.strip()]
        if law:
            got = (rec.get("governing_law") or "").lower()
            tally("governing_law", any(x.split(",")[0].lower() in got for x in law))
        d = iso(lab["Agreement Date-Answer"])
        if d:
            tally("agreement_date", d in (rec.get("agreement_date"), rec.get("effective_date")))
        d = iso(lab["Expiration Date-Answer"])
        if d:
            tally("expiration_date", rec.get("expiration_date") == d)
        m = re.fullmatch(r"(\d+)\)?\s*days", lab["Notice Period To Terminate Renewal- Answer"].strip())
        if m:
            tally("renewal_notice_days", rec.get("renewal_notice_days") == float(m.group(1)))
        for field, col in BOOLEANS.items():
            tally(field, bool(rec.get(field)) == (lab[col] == "Yes"))

    total = [sum(v[0] for v in result.values()), sum(v[1] for v in result.values())]
    for field, (ok, n) in result.items():
        print(f"{field:28s} {ok:>3}/{n:<3} {ok / n:.0%}")
    print(f"{'all compared fields':28s} {total[0]:>3}/{total[1]:<3} {total[0] / total[1]:.0%}")
    if args.out:
        args.out.write_text(json.dumps({"fields": result, "total": total}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
