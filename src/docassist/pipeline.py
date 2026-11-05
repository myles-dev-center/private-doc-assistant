"""Ingestion and question answering - the two things the API and UI call."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .chunking import chunk_doc
from .config import Settings
from .llm import LLM
from .parsing import SUPPORTED, parse_file
from .retrieval import Retriever
from .store import Store, StoredChunk
from .structured import Schema, extract, fix_sql, records_db, route, rows_to_json, run_sql

log = logging.getLogger(__name__)

DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "contract.yaml"
NOT_FOUND = "The documents do not contain this information."

ANSWER_SYSTEM = """You answer questions using ONLY the numbered excerpts provided.
Rules:
- Every factual statement must cite its excerpt like [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: NOT_FOUND
- Do not use outside knowledge. Do not guess. Be careful to use the excerpt about the document the question asks about.
- Be concise: one to three sentences."""

TABLE_ANSWER_SYSTEM = """You answer a question from the result of a database query over extracted document data.
Rules:
- Use only the rows given. If there are no rows, say that no documents match.
- For lists, mention every row by its identifier. For counts or sums, state the number.
- Be concise."""


@dataclass
class Source:
    n: int
    filename: str
    page: int | None
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    route: str  # "search" | "sql" | "not_found"
    sources: list[Source] = field(default_factory=list)
    sql: str | None = None
    columns: list[str] | None = None
    rows: list[list] | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Assistant:
    def __init__(self, settings: Settings, schema_path: Path | None = DEFAULT_SCHEMA):
        self.s = settings
        self.llm = LLM(settings)
        self.store = Store(settings.db_path)
        self.schema = Schema.load(schema_path) if schema_path else None
        self.retriever = Retriever(self.store, self.llm, hybrid=settings.hybrid)

    # ------------------------------------------------------------ ingest

    def ingest_file(self, path: Path, *, force: bool = False) -> dict:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = self.store.find_document(path.name)
        if existing and existing["sha256"] == sha and not force:
            return {"filename": path.name, "status": "unchanged"}
        if existing:
            self.store.delete_document(path.name)

        t0 = time.time()
        doc = parse_file(path)
        record = None
        if self.schema and (self.s.structured or self.s.context_headers):
            record = extract(self.llm, self.schema, doc.text)
        header = self._header(record)

        chunks = chunk_doc(doc, self.s.chunk_tokens, self.s.chunk_overlap)
        rows = [(c.ordinal, c.page, c.text, f"{header}\n{c.text}" if header else c.text) for c in chunks]
        vectors = self.llm.embed([r[3] for r in rows])
        doc_id = self.store.add_document(path.name, sha, len(doc.pages), doc.ocr_pages, header)
        self.store.add_chunks(doc_id, rows, vectors)
        if record is not None and self.schema:
            self.store.save_extraction(doc_id, self.schema.name, record)
        self.store.commit()
        return {
            "filename": path.name,
            "status": "ingested",
            "pages": len(doc.pages),
            "ocr_pages": doc.ocr_pages,
            "chunks": len(chunks),
            "seconds": round(time.time() - t0, 1),
        }

    def ingest_dir(self, folder: Path, *, force: bool = False, progress=None) -> list[dict]:
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
        results = []
        for i, p in enumerate(files, 1):
            try:
                results.append(self.ingest_file(p, force=force))
            except Exception as e:  # one bad file must not stop a batch
                log.exception("failed to ingest %s", p.name)
                results.append({"filename": p.name, "status": "error", "error": str(e)})
            if progress:
                progress(i, len(files), results[-1])
        return results

    def _header(self, record: dict | None) -> str | None:
        if not (record and self.schema and self.s.context_headers):
            return None
        parts = [str(record[f]) for f in self.schema.header_fields if record.get(f)]
        return " | ".join(parts) or None

    # ------------------------------------------------------------ ask

    def ask(self, question: str) -> Answer:
        t0 = time.time()
        ans = None
        if self.s.structured and self.schema:
            ans = self._ask_table(question)
        if ans is None:
            ans = self._ask_search(question)
        ans.seconds = round(time.time() - t0, 2)
        return ans

    def records(self) -> list[dict]:
        return self.store.extractions(self.schema.name) if self.schema else []

    def _ask_table(self, question: str) -> Answer | None:
        rows = self.records()
        if not rows:
            return None
        decision = route(self.llm, self.schema, question, dt.date.today().isoformat())
        if decision["route"] != "sql":
            return None
        db = records_db(self.schema, rows)
        sql = decision["sql"]
        try:
            cols, result = run_sql(db, sql)
        except Exception as e:
            sql = fix_sql(self.llm, self.schema, question, sql, str(e))
            try:
                cols, result = run_sql(db, sql)
            except Exception:
                log.warning("SQL failed twice, falling back to search: %s", sql)
                return None
        if not result:
            # An empty result usually means an over-strict filter (e.g. exact text match),
            # not a true "nothing matches"; let document search have a go.
            log.info("SQL returned no rows, falling back to search: %s", sql)
            return None
        text = self.llm.chat(
            TABLE_ANSWER_SYSTEM,
            f"Question: {question}\nSQL: {sql}\nRows ({len(result)}): {rows_to_json(cols, result)}",
            max_tokens=600,
        )
        return Answer(question, text, "sql", sql=sql, columns=cols, rows=result)

    def scope(self, question: str) -> set[int] | None:
        """Documents the question names explicitly (by an identifying field), if any.

        Contracts built from one template share most of their wording, so a
        question about "contract AP-MSA-2022-090" otherwise retrieves the same
        clause from a dozen other contracts.
        """
        if not (self.s.doc_scoping and self.schema):
            return None
        q = normalize(question)
        names = self.store.doc_ids_by_filename()
        hits = set()
        for rec in self.records():
            for f in self.schema.header_fields:
                v = rec.get(f)
                if isinstance(v, str) and len(v) >= 6 and f != "contract_type" and normalize(v) in q:
                    hits.add(names[rec["filename"]])
        return hits or None

    def _ask_search(self, question: str) -> Answer:
        hits = self.retriever.search(question, self.s.top_k, self.scope(question))
        if not hits:
            return Answer(question, NOT_FOUND, "not_found")
        context = "\n\n".join(self._excerpt(i, h) for i, h in enumerate(hits, 1))
        text = self.llm.chat(ANSWER_SYSTEM, f"Excerpts:\n{context}\n\nQuestion: {question}", max_tokens=400)
        if "NOT_FOUND" in text:
            return Answer(question, NOT_FOUND, "not_found")
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        sources = [Source(n, hits[n - 1].filename, hits[n - 1].page, hits[n - 1].text) for n in cited]
        return Answer(question, text, "search", sources=sources)

    def _excerpt(self, n: int, h: StoredChunk) -> str:
        # Naive RAG passes only the chunk text; with context headers on, the model
        # also sees which document the chunk came from.
        if self.s.context_headers:
            where = f"{h.filename}, p. {h.page}" if h.page else h.filename
            label = f"[{n}] ({h.header or ''} — {where})"
        else:
            label = f"[{n}]"
        return f"{label}\n{h.text}"


def normalize(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
