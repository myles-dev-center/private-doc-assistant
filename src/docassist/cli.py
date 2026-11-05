"""Command line: ingest a folder, ask a question, export the extracted table.

uv run docassist ingest data/corpus
uv run docassist ask "Which contracts renew automatically?"
uv run docassist export contracts.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import get_settings
from .export import export_xlsx
from .pipeline import Assistant


def main() -> None:
    ap = argparse.ArgumentParser(prog="docassist")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest", help="ingest all supported files in a folder")
    p.add_argument("folder", type=Path)
    p.add_argument("--force", action="store_true", help="re-ingest unchanged files")
    p = sub.add_parser("ask", help="ask a question")
    p.add_argument("question")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("export", help="export the extracted table to Excel")
    p.add_argument("path", type=Path)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    assistant = Assistant(get_settings())
    if args.cmd == "ingest":

        def progress(i, n, r):
            extra = (
                f"{r.get('chunks', '')} chunks, {r.get('ocr_pages', 0)} OCR pages, {r.get('seconds', '')}s"
                if r["status"] == "ingested"
                else r.get("error", "")
            )
            print(f"[{i}/{n}] {r['status']:<9} {r['filename']}  {extra}")

        assistant.ingest_dir(args.folder, force=args.force, progress=progress)
    elif args.cmd == "ask":
        ans = assistant.ask(args.question)
        if args.json:
            print(json.dumps(ans.to_dict(), indent=2, default=str))
            return
        print(ans.answer)
        if ans.sql:
            print(f"\nSQL: {ans.sql}")
        for s in ans.sources:
            print(f"  [{s.n}] {s.filename}" + (f", p. {s.page}" if s.page else ""))
        print(f"({ans.route}, {ans.seconds}s)")
    elif args.cmd == "export":
        export_xlsx(assistant.records(), assistant.schema, args.path)
        print(f"wrote {args.path}")


if __name__ == "__main__":
    main()
