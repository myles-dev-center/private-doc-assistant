"""Compare model combinations (chat model x embedding model) on blind CUAD sets.

    uv run python scripts/run_combos.py --combos combos.yaml --sets 4,5,6 --stores ../stores-cmp4

Each combination is a set of model settings (like a UI preset). Extraction at ingest
depends only on the chat model, so each set is ingested once per chat model; a
combination with another embedding model gets a copy of that library re-embedded
(same text, chunks and extracted rows - only the vectors differ). That gives exactly
the library a fresh ingest with the combination would build, at a fraction of the cost.

Results: evals/cmp4/<combo>/<set>.json (one file per combination and set; finished files
are skipped on a re-run, so an interrupted run resumes). Progress and a Stop button
appear in the web UI's activity panel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from compare_common import public_endpoint  # noqa: E402

from docassist import Assistant, Settings, get_settings  # noqa: E402
from docassist.activity import Activity, Cancelled  # noqa: E402
from docassist.evaluation import score, summarize  # noqa: E402
from docassist.llm import Usage  # noqa: E402
from docassist.pipeline import Answer  # noqa: E402

_VAR = re.compile(r"\$\{(\w+)\}")


def load_combos(path: Path) -> dict[str, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = {}
    for name, spec in raw["combos"].items():
        spec = {k: _VAR.sub(lambda m: os.environ.get(m.group(1), ""), v) if isinstance(v, str) else v for k, v in spec.items()}
        unknown = set(spec) - set(Settings.model_fields) - {"label"}
        if unknown:
            raise SystemExit(f"{path}: combo {name} has unknown settings {sorted(unknown)}")
        out[name] = spec
    return out


def settings_for(base: Settings, spec: dict, data_dir: Path) -> Settings:
    return base.model_copy(update={k: v for k, v in spec.items() if k != "label"} | {"data_dir": data_dir})


def library(stores: Path, set_name: str, chat: str, embed: str) -> Path:
    def slug(s: str) -> str:
        return re.sub(r"[^a-z0-9.]+", "-", s.lower()).strip("-")

    return stores / set_name / f"{slug(chat)}__{slug(embed)}"


def ensure_library(base: Settings, stores: Path, set_name: str, corpus: Path, spec: dict, combos: dict, act: Activity) -> tuple[Path, dict]:
    """The library for this combination and set; built by ingesting with the combination's
    chat model, or copied from the same chat model's library and re-embedded."""
    chat, embed = spec["llm_model"], spec["embed_model"]
    target = library(stores, set_name, chat, embed)
    files = sorted(p for p in corpus.iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt", ".md"})
    usage = {}
    if (target / "docassist.db").exists():
        a = Assistant(settings_for(base, spec, target))
        if len(a.store.documents()) == len(files) and a.index_problem() is None:
            return target, usage
    # A finished library of the same chat model with another embedding model: reuse its extraction.
    donor = None
    for other in combos.values():
        if other["llm_model"] == chat and other["embed_model"] != embed:
            d = library(stores, set_name, chat, other["embed_model"])
            if (d / "docassist.db").exists():
                src = Assistant(settings_for(base, other, d))
                if len(src.store.documents()) == len(files) and src.index_problem() is None:
                    donor = d
                    break
    t0 = time.time()
    if donor:
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(donor / "docassist.db", target / "docassist.db")
        a = Assistant(settings_for(base, spec, target))
        docs = a.store.documents()
        act.update(total=len(docs), done=0, unit="documents", step=f"re-embedding with {embed} (extraction reused)")
        for i, d in enumerate(docs):
            act.check()
            act.update(detail=d["filename"])
            a.reembed(d["filename"])
            act.update(done=i + 1)
        usage = a.llm.usage.snapshot() | {"seconds": round(time.time() - t0, 1), "reused_extraction_from": donor.name}
    else:
        a = Assistant(settings_for(base, spec, target))
        act.update(total=len(files), done=0, unit="documents", step=f"ingesting with {chat} + {embed}")
        for i, p in enumerate(files):
            act.check()
            act.update(detail=p.name)

            def step(stage, done=0, total=0, **info):
                act.check()

            a.ingest_file(p, progress=step)
            act.update(done=i + 1)
        usage = a.llm.usage.snapshot() | {"seconds": round(time.time() - t0, 1)}
    (target / "ingest_usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    return target, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", type=Path, required=True, help="YAML: combos: {name: {label, llm_*, embed_*...}}")
    ap.add_argument("--only", default="", help="comma-separated combo names (default: all)")
    ap.add_argument("--sets", default="4,5,6", help="CUAD set numbers: data/corpus-cuad-set<N> + evals/cuad-golden-set<N>.yaml")
    ap.add_argument("--corpus", type=Path, help="instead of --sets: one corpus folder ...")
    ap.add_argument("--golden", type=Path, help="... and its golden file")
    ap.add_argument("--stores", type=Path, required=True, help="where the libraries are built")
    ap.add_argument("--out", type=Path, default=Path("evals/cmp4"))
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke test)")
    args = ap.parse_args()

    base = get_settings()
    combos = load_combos(args.combos)
    names = [n for n in args.only.split(",") if n] or list(combos)
    if args.corpus:
        sets = [(args.corpus.name, args.corpus, args.golden)]
    else:
        sets = [(f"set{n}", Path(f"data/corpus-cuad-set{n}"), Path(f"evals/cuad-golden-set{n}.yaml")) for n in args.sets.split(",")]

    for name in names:
        spec = combos[name]
        for set_name, corpus, golden_path in sets:
            out = args.out / name / f"{set_name}.json"
            if out.exists() and not args.limit:
                print(f"skip {name} / {set_name} (done)", flush=True)
                continue
            golden = yaml.safe_load(golden_path.read_text(encoding="utf-8"))
            documents, questions = golden["documents"], golden["questions"]
            if args.limit:
                questions = questions[: args.limit]
            act = Activity(
                "eval",
                f"Model comparison · {spec.get('label', name)} · {set_name}",
                model=spec["llm_model"],
                local_model="localhost" in spec.get("llm_base_url", "") or "127.0.0.1" in spec.get("llm_base_url", ""),
            )
            try:
                with act:
                    print(f"\n== {name} / {set_name}: library", flush=True)
                    lib, ingest = ensure_library(base, args.stores, set_name, corpus, spec, combos, act)
                    a = Assistant(settings_for(base, spec, lib))
                    a.llm.usage = Usage()
                    act.update(total=len(questions), done=0, unit="questions", step="answering", detail="")
                    scores = []
                    for i, q in enumerate(questions):
                        act.check()
                        act.update(detail=q["question"], step=f"{sum(s.correct for s in scores)} of {i} correct so far")
                        try:
                            ans = a.ask(q["question"])
                        except Cancelled:
                            raise
                        except Exception as e:  # a crash counts as a wrong answer
                            ans = Answer(q["question"], f"ERROR: {e}", "error")
                        s = score(q, ans, documents)
                        row = asdict(s) | {"usage": ans.usage, "model": ans.model}
                        scores.append(s)
                        scores[-1].__dict__["_row"] = row
                        mark = "ok " if s.correct else ("HAL" if s.hallucinated else "-- ")
                        print(f"  {mark} {q['id']} [{s.route}] {s.seconds:>5.1f}s  {q['question'][:70]}", flush=True)
                        act.update(done=i + 1)
                    summary = summarize(scores)
                    result = {
                        "combo": name,
                        "label": spec.get("label", name),
                        "chat_model": spec["llm_model"],
                        "embed_model": spec["embed_model"],
                        "endpoint": public_endpoint(spec.get("llm_base_url", base.llm_base_url)),
                        "set": set_name,
                        "library": lib.name,
                        "started": time.strftime("%Y-%m-%d %H:%M"),
                        "summary": summary,
                        "usage": {"ingest": ingest, "questions": a.llm.usage.snapshot()},
                        "scores": [s.__dict__["_row"] for s in scores],
                    }
                    if not args.limit:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                    act.finish("done", f"accuracy {summary['accuracy']:.0%} on {len(scores)} questions")
                    acc, hal = summary["accuracy"], summary["hallucination_rate"]
                    print(f"  => {name} / {set_name}: accuracy {acc}, hallucination {hal}", flush=True)
            except Cancelled:
                print("stopped by user", flush=True)
                sys.exit(1)


if __name__ == "__main__":
    main()
