"""Re-apply the current refusal rule to saved results, without asking any model again.

    uv run python scripts/rescore.py evals/cmp4 evals/cmp4b      # any folders of result files

Only whether an answer counts as "not found" can change between scorer versions, and
with it `correct` on unanswerable questions and `hallucinated` everywhere. Every other
check needs the original answer object (SQL rows, sources) and is kept as saved.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

from docassist.evaluation import Score, is_refusal, summarize
from docassist.pipeline import Answer

FIELDS = {f.name for f in fields(Score)}


def rescore(scores: list[dict]) -> int:
    changed = 0
    for s in scores:
        refused = is_refusal(Answer(s["question"], s["answer"] or "", s["route"]))
        correct = refused if s["qtype"] == "unanswerable" else s["correct"]
        hallucinated = not correct and not refused
        changed += (correct, hallucinated, refused) != (s["correct"], s["hallucinated"], s["refused"])
        s.update(correct=correct, hallucinated=hallucinated, refused=refused)
    return changed


def resummarize(summary: dict, scores: list[dict]) -> None:
    new = summarize([Score(**{k: v for k, v in s.items() if k in FIELDS}) for s in scores])
    new.pop("median_seconds")  # unchanged, and some runners compute it differently
    summary.update(new)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    for folder in ap.parse_args().folders:
        for f in sorted(folder.rglob("*.json")):
            run = json.loads(f.read_text(encoding="utf-8"))
            if "scores" in run:  # run_combos.py
                n = rescore(run["scores"])
                resummarize(run["summary"], run["scores"])
            elif "configs" in run:  # run_eval.py and the other methods
                n = 0
                for cfg in run["configs"].values():
                    n += rescore(cfg["scores"])
                    resummarize(cfg["summary"], cfg["scores"])
            else:
                continue
            f.write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
            print(f"{f}: {n} answers rescored")


if __name__ == "__main__":
    main()
