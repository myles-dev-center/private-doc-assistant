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
    route: str  # "search" | "document" | "sql" | "not_found"
    sources: list[Source] = field(default_factory=list)
    sql: str | None = None
    columns: list[str] | None = None
    rows: list[list] | None = None
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Assistant:
    def __init__(self, settings: Settings, schema_path: Path | None = None):
        self.s = settings
        self.llm = LLM(settings)
        self.store = Store(settings.db_path)
        self.schema = Schema.load(schema_path or settings.schema_path or DEFAULT_SCHEMA)
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
        self.store.add_pages(doc_id, [(p.number, p.text) for p in doc.pages])
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
            try:
                ans = self._ask_table(question)
            except Exception:  # a bad model reply must not cost the user an answer
                log.exception("table route failed, falling back to search")
        if ans is None:
            ans = self._ask_search(question)
        ans.seconds = round(time.time() - t0, 2)
        return ans

    def records(self) -> list[dict]:
        return self.store.extractions(self.schema.name) if self.schema else []

    def _ask_table(self, question: str) -> Answer | None:
        rows = self.records()
        if not rows or self.scope(question):
            # A question that names a document is answered from its text, with citations;
            # its table row may be missing the very field asked about.
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
        if not has_values(cols, result, {"filename", *self.schema.header_fields}):
            # No rows usually means an over-strict filter (e.g. exact text match), and rows
            # with only NULLs mean extraction missed the field - not a true "nothing matches";
            # let document search have a go.
            log.info("SQL returned nothing usable, falling back to search: %s", sql)
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
            for f in self.schema.scope_fields:
                v = rec.get(f)
                if isinstance(v, str) and len(v) >= 6 and normalize(v) in q:
                    hits.add(names[rec["filename"]])
        return hits or None

    def _ask_search(self, question: str) -> Answer:
        scope = self.scope(question)
        hits = self._whole_document(scope)
        route = "document" if hits else "search"
        if not hits:
            hits = self.retriever.search(question, self.s.top_k, scope)
        if not hits:
            return Answer(question, NOT_FOUND, "not_found")
        context = "\n\n".join(self._excerpt(i, h) for i, h in enumerate(hits, 1))
        text = self.llm.chat(ANSWER_SYSTEM, f"Excerpts:\n{context}\n\nQuestion: {question}", max_tokens=400)
        if "NOT_FOUND" in text:
            return Answer(question, NOT_FOUND, "not_found")
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
        sources = [Source(n, hits[n - 1].filename, hits[n - 1].page, hits[n - 1].text) for n in cited]
        return Answer(question, text, route, sources=sources)

    def _whole_document(self, scope: set[int] | None) -> list[StoredChunk]:
        """All pages of the one document a question names, plus its extracted record, if it fits.

        Search can miss the clause that answers a question about a known document
        (e.g. it returns "a corporation organized under the laws of Delaware" for a
        governing-law question). A typical agreement fits in the context window,
        so the model may as well read all of it.
        """
        if not (scope and len(scope) == 1 and self.s.full_doc_chars):
            return []
        doc_id = next(iter(scope))
        pages = self.store.document_pages(doc_id)
        if not pages or sum(len(t) for _, t in pages) > self.s.full_doc_chars:
            return []
        doc = self.store.document_by_id(doc_id)
        hits = [StoredChunk(-1, doc_id, doc["filename"], p, t, doc["header"]) for p, t in pages]
        rec = next((r for r in self.records() if r["filename"] == doc["filename"]), None)
        if rec:
            facts = "; ".join(f"{k}: {v}" for k, v in rec.items() if k != "filename" and v is not None)
            note = "Fields extracted from this document at ingest (check them against the pages):"
            hits.insert(0, StoredChunk(-1, doc_id, doc["filename"], None, f"{note} {facts}", doc["header"]))
        return hits

    def _excerpt(self, n: int, h: StoredChunk) -> str:
        # Naive RAG passes only the chunk text; with context headers on, the model
        # also sees which document the chunk came from.
        if self.s.context_headers:
            where = f"{h.filename}, p. {h.page}" if h.page else h.filename
            label = f"[{n}] ({h.header or ''} — {where})"
        else:
            label = f"[{n}]"
        return f"{label}\n{h.text}"


def has_values(cols: list[str], rows: list[list], id_cols: set[str]) -> bool:
    data = [i for i, c in enumerate(cols) if c not in id_cols]
    return bool(rows) and (not data or any(r[i] is not None for r in rows for i in data))


def normalize(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "
