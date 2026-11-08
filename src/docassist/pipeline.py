"""Ingestion and question answering - the two things the API and UI call."""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import names
from .activity import Activity, Cancelled
from .chunking import chunk_doc
from .config import Settings
from .llm import LLM, is_local_url
from .parsing import SUPPORTED, Page, ParsedDoc, parse_file
from .retrieval import Retriever, fts_text, unspace_cjk
from .store import Store, StoredChunk
from .structured import Schema, extract, fix_sql, locate, records_db, relax_sql, route, rows_to_json, run_sql

log = logging.getLogger(__name__)

DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "schemas" / "contract.yaml"
NOT_FOUND = "The documents do not contain this information."
NO_MATCH = "No documents match."
# A question that names a document but asks about several ("which other agreements
# does Acme have?") needs the router, not a jump to that one document.
_MANY = re.compile(r"\b(agreements|contracts|documents|deals)\b|哪些|所有|几份|多少份|其他|其它|列出", re.I)

ANSWER_SYSTEM = """You answer questions using ONLY the numbered excerpts provided.
Rules:
- Every factual statement must cite its excerpt like [1] or [2][3].
- If the excerpts do not contain the answer, reply exactly: NOT_FOUND
- Do not use outside knowledge. Do not guess. Be careful to use the excerpt about the document the question asks about.
- Be concise: one to three sentences, in the language of the question."""

TABLE_ANSWER_SYSTEM = """You answer a question from the result of a database query over extracted document data.
Rules:
- Use only the rows given. If there are no rows, say that no documents match.
- For lists, mention every row by its identifier. For counts or sums, state the number.
- Be concise, and answer in the language of the question."""


@dataclass
class Source:
    n: int
    filename: str
    page: int | None
    text: str


@dataclass
class Evidence:
    """Where a value in the extracted table comes from."""

    filename: str
    field: str
    value: object
    quote: str | None
    page: int | None
    verified: bool  # the quote was found in the document


@dataclass
class Answer:
    question: str
    answer: str
    route: str  # "search" | "document" | "sql" | "not_found"
    sources: list[Source] = field(default_factory=list)
    sql: str | None = None
    columns: list[str] | None = None
    rows: list[list] | None = None
    evidence: list[Evidence] = field(default_factory=list)  # for table answers: the quote behind each value used
    seconds: float = 0.0
    model: str = ""
    usage: dict | None = None  # tokens and list-price cost of this answer

    def to_dict(self) -> dict:
        return asdict(self)


# How ingestion steps are described in the activity panel.
STAGE_TEXT = {
    "parse": "reading pages",
    "extract": "model is extracting fields",
    "verify": "checking quotes",
    "chunk": "splitting",
    "embed": "indexing",
    "save": "saving",
}


def stage_text(stage: str, done: int, total: int, **info) -> str:
    text = STAGE_TEXT.get(stage, stage)
    if stage == "parse" and total:
        text = f"reading page {done} of {total}" + (" (OCR)" if info.get("ocr") else "")
    elif stage == "extract" and total > 1:
        text += f" (part {done} of {total})"
    elif stage == "embed" and total:
        text += f" ({done}/{total} passages)"
    return text


def _quiet(stage: str, done: int = 0, total: int = 0, **info) -> None:
    pass


class IndexMismatch(RuntimeError):
    """The library was indexed with a different embedding model than the current one."""


class Assistant:
    def __init__(self, settings: Settings, schema_path: Path | None = None, store: Store | None = None):
        self.s = settings
        self.llm = LLM(settings)
        self.store = store or Store(settings.db_path)
        self.schema = Schema.load(schema_path or settings.schema_path or DEFAULT_SCHEMA)
        self.retriever = Retriever(self.store, self.llm, hybrid=settings.hybrid)
        self._local = threading.local()

    # ------------------------------------------------------------ ingest

    def index_problem(self) -> str | None:
        """Why this library can't be searched with the current embedding model, if it can't.

        Vectors from different embedding models are not comparable: mixing them silently
        returns nonsense, so ingest and search refuse instead.
        """
        built = self.store.get_meta("embed_model")
        if built and built != self.s.embed_model:
            return (
                f"This library was indexed with {built}, but the embedding model is now {self.s.embed_model}. "
                "Switch back, or delete and re-ingest the documents."
            )
        return None

    def _check_index(self, rebuilding: bool = False) -> None:
        """Refuse to mix vectors of different embedding models. Re-processing every document
        of a library with the new model is allowed: it replaces all of the old vectors."""
        if problem := self.index_problem():
            if rebuilding:
                return
            raise IndexMismatch(problem)

    def ingest_file(self, path: Path, *, force: bool = False, progress=None) -> dict:
        """Parse, extract, chunk, embed and store one file.

        `progress(stage, done, total, **info)` reports each step: parse (per page),
        extract (per part of the document), verify, chunk, embed (per batch), save.
        """
        self._check_index()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        existing = self.store.find_document(path.name)
        if existing and existing["sha256"] == sha and not force:
            return {"filename": path.name, "status": "unchanged"}
        step = progress or _quiet
        step("parse", 0, 0)
        doc = parse_file(path, lambda i, n, ocr: step("parse", i, n, ocr=ocr))
        return self._process(doc, sha, existing["id"] if existing else None, step)

    def reprocess(self, filename: str, *, progress=None) -> dict:
        """Extract and index a stored document again with the current models, from its saved
        page text (no re-upload, no OCR). Keeps its number and any manual edits."""
        self._check_index(rebuilding=True)
        row = self.store.find_document(filename)
        if row is None:
            raise KeyError(filename)
        pages = self.store.document_pages(row["id"])
        doc = ParsedDoc(filename, [Page(p, t) for p, t in pages])
        ocr = row["ocr_pages"] or 0
        result = self._process(doc, row["sha256"], row["id"], progress or _quiet, ocr_pages=ocr)
        return {**result, "status": "reprocessed"}

    def reembed(self, filename: str, *, progress=None) -> dict:
        """Embed a stored document again with the current embedding model, keeping its text,
        keyword index, extracted row, quotes and edits - no chat model call. Switching a
        library to another embedding model this way costs only the embedding calls."""
        step = progress or _quiet
        row = self.store.find_document(filename)
        if row is None:
            raise KeyError(filename)
        t0 = time.time()
        chunks = self.store.chunk_texts(row["id"])
        # search_text went through fts_text, which only spaces out CJK characters; the
        # embedding input was the same text without those spaces
        texts = [unspace_cjk(t) for _, t in chunks]
        step("embed", 0, len(texts))
        vectors = self.llm.embed(texts, progress=lambda i, n: step("embed", i, n))
        step("save", 0, 1)
        with self.store.transaction():
            self.store.set_embeddings(row["id"], [cid for cid, _ in chunks], vectors, self.s.embed_model)
            others = [d for d in self.store.documents() if d["embed_model"] and d["embed_model"] != self.s.embed_model]
            if not others:
                self.store.set_meta("embed_model", self.s.embed_model)
            self.store.commit()
        step("save", 1, 1)
        return {"filename": filename, "status": "re-embedded", "chunks": len(chunks), "seconds": round(time.time() - t0, 1)}

    def _process(self, doc, sha: str, doc_id: int | None, step, ocr_pages: int | None = None) -> dict:
        """Extract, verify, chunk, embed and store a parsed document; a new one or in place of doc_id.

        Nothing is written until every model call has succeeded, so a stopped or failed
        run leaves the library as it was."""
        t0 = time.time()
        path = Path(doc.filename)
        pages = [(p.number, p.text) for p in doc.pages]
        record, evidence = None, []
        if self.schema and (self.s.structured or self.s.context_headers):
            ex = extract(self.llm, self.schema, doc.text, progress=lambda i, n: step("extract", i, n))
            step("verify", 0, len(ex.quotes))
            record, evidence = self._verify(ex.record, ex.quotes, pages)
            step("verify", len(ex.quotes), len(ex.quotes), verified=sum(e[3] for e in evidence), quoted=len(evidence))
        header = self._header(record)

        chunks = chunk_doc(doc, self.s.chunk_tokens, self.s.chunk_overlap)
        step("chunk", len(chunks), len(chunks))
        texts = [f"{header}\n{c.text}" if header else c.text for c in chunks]
        step("embed", 0, len(texts))
        vectors = self.llm.embed(texts, progress=lambda i, n: step("embed", i, n))
        step("save", 0, 1)
        rows = [(c.ordinal, c.page, c.text, fts_text(t)) for c, t in zip(chunks, texts)]
        ocr = doc.ocr_pages if ocr_pages is None else ocr_pages
        meta = (
            sha,
            len(doc.pages),
            ocr,
            header,
            self.s.embed_model,
            self.s.llm_model if record is not None else None,
            round(time.time() - t0, 1),
        )
        # All writes of one document as one unit: parallel uploads can't interleave them.
        with self.store.transaction():
            if doc_id is None:
                again = self.store.find_document(path.name)  # the same file, uploaded twice at once
                doc_id = again["id"] if again else None
            if doc_id is None:
                doc_id = self.store.add_document(path.name, *meta)
            else:
                self.store.replace_document(doc_id, *meta)
            self.store.add_chunks(doc_id, rows, vectors)
            self.store.add_pages(doc_id, pages)
            if record is not None and self.schema:
                self.store.save_extraction(doc_id, self.schema.name, record)
                self.store.save_evidence(doc_id, evidence)
            self.store.set_meta("embed_model", self.s.embed_model)
            self.store.commit()
        step("save", 1, 1)
        return {
            "filename": path.name,
            "status": "ingested",
            "pages": len(doc.pages),
            "ocr_pages": ocr,
            "chunks": len(chunks),
            "seconds": round(time.time() - t0, 1),
        }

    def uses_local_model(self) -> bool:
        return is_local_url(self.s.llm_base_url)

    def ingest_dir(self, folder: Path, *, force: bool = False, progress=None, title: str | None = None) -> list[dict]:
        """Ingest every supported file in a folder; progress is also shown in the UI's activity panel."""
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED)
        results = []
        act = Activity(
            "ingest",
            title or f"Adding {len(files)} documents from {folder.name}",
            total=len(files),
            unit="documents",
            model=self.s.llm_model,
            local_model=self.uses_local_model(),
        )
        t_work, n_work = 0.0, 0
        with act:
            for i, p in enumerate(files, 1):
                act.update(done=i - 1, detail=p.name, step="checking")
                t0 = time.time()

                def step(stage, done=0, total=0, **info):
                    act.check()
                    act.update(step=stage_text(stage, done, total, **info))

                try:
                    results.append(self.ingest_file(p, force=force, progress=step))
                except Cancelled:
                    raise
                except Exception as e:  # one bad file must not stop a batch
                    log.exception("failed to ingest %s", p.name)
                    results.append({"filename": p.name, "status": "error", "error": str(e)})
                if results[-1]["status"] == "ingested":
                    t_work, n_work = t_work + time.time() - t0, n_work + 1
                left = len(files) - i
                act.update(done=i, eta=round(t_work / n_work * left) if n_work and left else None)
                if progress:
                    progress(i, len(files), results[-1])
            failed = sum(r["status"] == "error" for r in results)
            act.finish("done", f"{len(files) - failed} of {len(files)} documents ready" + (f", {failed} failed" if failed else ""))
        return results

    def _verify(self, record: dict, quotes: dict, pages: list) -> tuple[dict, list[tuple]]:
        """Look up each field's quote in the document. A "yes" for a clause (boolean field)
        with no quote that can be found in the text is not trusted: it becomes unknown
        (NULL), so a list of "agreements with a non-compete clause" only contains
        agreements where the clause was actually seen."""
        evidence = []
        for f in self.schema.fields:
            value = record.get(f.name)
            if value is None or value is False:
                continue
            quote = quotes.get(f.name)
            found, page = locate(quote, pages) if quote else (False, None)
            if f.type == "boolean" and not found and self.s.require_clause_quote:
                record[f.name] = None
            evidence.append((f.name, quote, page, found))
        return record, evidence

    def _header(self, record: dict | None) -> str | None:
        if not (record and self.schema and self.s.context_headers):
            return None
        parts = [str(record[f]) for f in self.schema.header_fields if record.get(f)]
        return " | ".join(parts) or None

    # ------------------------------------------------------------ ask

    def ask(self, question: str, document: str | None = None, progress=None) -> Answer:
        """Answer a question; `document` (a filename) restricts it to that one document.

        `progress(stage, done, total, **info)` reports each step: match (named documents),
        route, sql / read / search, answer.
        """
        self._local.step = progress or _quiet  # per thread: questions may run side by side
        t0 = time.time()
        with self.llm.track() as used:  # the cost of this answer only, not of parallel uploads
            self._check_index()
            ids = self.store.doc_ids_by_filename()
            if document:
                if document not in ids:
                    raise KeyError(document)
                ans = self._ask_search(question, {ids[document]})
            else:
                ans = self._route(question, ids)
        ans.seconds = round(time.time() - t0, 2)
        ans.model = self.s.llm_model
        ans.usage = used.snapshot()
        return ans

    def _step(self, stage: str, done: int = 0, total: int = 0, **info) -> None:
        getattr(self._local, "step", _quiet)(stage, done, total, **info)

    def records(self) -> list[dict]:
        return self.store.extractions(self.schema.name) if self.schema else []

    def _route(self, question: str, ids: dict[str, int]) -> Answer:
        """Pick the source of the answer: one named document, the extracted table, or search.

        A question that names one document by its full name and asks about that document
        goes straight to it (no routing call). Anything less certain - a short name that
        may be wrong, or a question that names a document but asks about several - is
        decided by the router, which sees the candidate documents.
        """
        named = self.named_documents(question)
        full = [f for f, how in named.items() if how == "full"]
        self._step("match", 1, 1, documents=named)
        if len(full) == 1 and not _MANY.search(question):
            return self._ask_search(question, {ids[full[0]]})
        records = self.records()
        if not (self.s.structured and self.schema and records):
            return self._ask_search(question, {ids[f] for f in full} or None)
        candidates = [r for r in records if r["filename"] in named]
        try:
            self._step("route", 0, 1)
            decision = route(self.llm, self.schema, question, dt.date.today().isoformat(), candidates)
            self._step("route", 1, 1, route=decision["route"])
            if decision["route"] == "document":
                return self._ask_search(question, {ids[decision["document"]]})
            if decision["route"] == "sql" and (ans := self._ask_table(question, decision["sql"], records)):
                return ans
        except Exception:  # a bad model reply must not cost the user an answer
            log.exception("table route failed, falling back to search")
        return self._ask_search(question, {ids[f] for f in full} or None)

    def _ask_table(self, question: str, sql: str, records: list[dict]) -> Answer | None:
        self._step("sql", 0, 1, sql=sql)
        db = records_db(self.schema, records)
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
            # Either nothing matches - a real answer - or the filter is stricter than the
            # question (exact text, a spelling the table doesn't use). The model sees the
            # stored values and may loosen it once; if it stands, the answer is "none".
            self._step("relax", 0, 1)
            relaxed = relax_sql(self.llm, self.schema, db, question, sql)
            self._step("relax", 1, 1, changed=bool(relaxed))
            if relaxed:
                try:
                    cols2, result2 = run_sql(db, relaxed)
                    if result2:
                        sql, cols, result = relaxed, cols2, result2
                except Exception:
                    log.info("relaxed SQL failed: %s", relaxed)
        if not result:
            return Answer(question, NO_MATCH, "sql", sql=sql, columns=cols, rows=[])
        if not has_values(cols, result, {"filename", *self.schema.header_fields}):
            # Rows with only NULLs mean extraction missed the field - not an answer.
            log.info("SQL returned only empty values, falling back to search: %s", sql)
            return None
        self._step("sql", 1, 1, sql=sql, rows=len(result))
        self._step("answer", 0, 1)
        text = self.llm.chat(
            TABLE_ANSWER_SYSTEM,
            f"Question: {question}\nSQL: {sql}\nRows ({len(result)}): {rows_to_json(cols, result)}",
            max_tokens=600,
        )
        return Answer(question, text, "sql", sql=sql, columns=cols, rows=result, evidence=self._evidence_for(sql, cols, result, records))

    def _evidence_for(self, sql: str, cols: list[str], rows: list[list], records: list[dict]) -> list[Evidence]:
        """The quote behind every table value the answer relies on: the fields the query
        selects or filters on, for every row it returned."""
        fields = [f.name for f in self.schema.fields if f.name not in self.schema.header_fields and re.search(rf"\b{f.name}\b", sql)]
        if not fields:
            fields = [f.name for f in self.schema.fields if f.name in cols]
        files = []
        for r in rows:
            row = dict(zip(cols, r))
            if row.get("filename"):
                files.append(row["filename"])
                continue
            keys = {c: v for c, v in row.items() if c in self.schema.header_fields and v is not None}
            match = [rec["filename"] for rec in records if keys and all(rec.get(c) == v for c, v in keys.items())]
            if len(match) == 1:
                files.append(match[0])
        by_file = {rec["filename"]: rec for rec in records}
        quotes = self.store.evidence()
        out = []
        for fn in dict.fromkeys(files):
            for f in fields:
                q = quotes.get((fn, f))
                out.append(Evidence(fn, f, by_file.get(fn, {}).get(f), q and q["quote"], q and q["page"], bool(q and q["verified"])))
        return out

    def named_documents(self, question: str) -> dict[str, str]:
        """{filename: "full" | "partial"} for documents the question names (by an identifying field).

        Contracts built from one template share most of their wording, so a
        question about "contract AP-MSA-2022-090" otherwise retrieves the same
        clause from a dozen other contracts.
        """
        if not (self.s.doc_scoping and self.schema):
            return {}
        values = {r["filename"]: [r[f] for f in self.schema.scope_fields if isinstance(r.get(f), str)] for r in self.records()}
        return names.match(question, values)

    def scope(self, question: str) -> set[int] | None:
        """Ids of the documents a question names by full name, if any."""
        ids = self.store.doc_ids_by_filename()
        hits = {ids[f] for f, how in self.named_documents(question).items() if how == "full"}
        return hits or None

    def _ask_search(self, question: str, scope: set[int] | None = None) -> Answer:
        step = self._step
        hits = self._whole_document(scope)
        route = "document" if hits else "search"
        if hits:
            step("read", 1, 1, document=hits[0].filename, pages=sum(h.page is not None for h in hits))
        else:
            step("search", 0, 1)
            hits = self.retriever.search(question, self.s.top_k, scope)
            step("search", 1, 1, excerpts=len(hits))
        if not hits:
            return Answer(question, NOT_FOUND, "not_found")
        context = "\n\n".join(self._excerpt(i, h) for i, h in enumerate(hits, 1))
        step("answer", 0, 1)
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
