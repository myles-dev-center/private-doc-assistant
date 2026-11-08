"""SQLite storage: documents, chunks (+FTS5 index, embeddings) and extracted records.

One file, no server. Vectors are kept as float32 blobs and searched with numpy;
for a few hundred thousand chunks that is fast enough and keeps deployment trivial.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    sha256 TEXT NOT NULL,
    pages INTEGER,
    ocr_pages INTEGER DEFAULT 0,
    header TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embedding BLOB
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    search_text, content='chunks', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, search_text) VALUES (new.id, new.search_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, search_text) VALUES ('delete', old.id, old.search_text);
END;
CREATE TABLE IF NOT EXISTS pages (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS extractions (
    doc_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    schema_name TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field TEXT NOT NULL,
    quote TEXT,
    page INTEGER,
    verified INTEGER NOT NULL,
    PRIMARY KEY (doc_id, field)
);
CREATE TABLE IF NOT EXISTS edits (
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    field TEXT NOT NULL,
    value TEXT,
    original TEXT,
    edited_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, field)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class StoredChunk:
    id: int
    doc_id: int
    filename: str
    page: int | None
    text: str
    header: str | None


def _serialized(cls):
    """One SQLite connection is shared by request threads and parallel uploads: every
    public method holds the store's lock while it touches the database. Model calls happen
    outside the store, so they still run side by side."""
    for name, fn in list(vars(cls).items()):
        if callable(fn) and not name.startswith("_") and name != "transaction":

            def wrap(fn):
                @functools.wraps(fn)
                def locked(self, *a, **kw):
                    with self.lock:
                        return fn(self, *a, **kw)

                return locked

            setattr(cls, name, wrap(fn))
    return cls


@_serialized
class Store:
    def __init__(self, path: Path):
        self.lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        # Libraries created before these columns existed get them (NULL = not recorded).
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(documents)")}
        for col, kind in (("embed_model", "TEXT"), ("extract_model", "TEXT"), ("ingest_seconds", "REAL")):
            if col not in cols:
                self.db.execute(f"ALTER TABLE documents ADD COLUMN {col} {kind}")
        self.db.commit()
        self._vec_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @contextmanager
    def transaction(self):
        """Hold the lock across several writes that belong together (saving one document),
        so another thread's writes can't interleave with them."""
        with self.lock:
            yield self

    # ------------------------------------------------------------ meta

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ------------------------------------------------------------ documents

    def find_document(self, filename: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM documents WHERE filename = ?", (filename,)).fetchone()

    def delete_document(self, filename: str) -> None:
        self.db.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        if not self.db.execute("SELECT 1 FROM documents LIMIT 1").fetchone():
            self.db.execute("DELETE FROM meta")  # meta describes the index; an empty library has none
        self.db.commit()
        self._vec_cache = None

    def add_document(
        self,
        filename: str,
        sha256: str,
        pages: int,
        ocr_pages: int,
        header: str | None,
        embed_model: str | None = None,
        extract_model: str | None = None,
        ingest_seconds: float | None = None,
    ) -> int:
        """`embed_model` made the document's vectors; `extract_model` (a chat model) filled in its table row."""
        cur = self.db.execute(
            "INSERT INTO documents(filename, sha256, pages, ocr_pages, header, embed_model, extract_model, ingest_seconds) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (filename, sha256, pages, ocr_pages, header, embed_model, extract_model, ingest_seconds),
        )
        return cur.lastrowid

    def replace_document(
        self,
        doc_id: int,
        sha256: str,
        pages: int,
        ocr_pages: int,
        header: str | None,
        embed_model: str | None,
        extract_model: str | None,
        ingest_seconds: float | None,
    ) -> None:
        """Re-processing a document: keep its id (its number in the library) and its manual
        edits, drop its chunks, pages, extracted row and evidence so they can be written anew."""
        for table in ("chunks", "pages", "extractions", "evidence"):
            self.db.execute(f"DELETE FROM {table} WHERE doc_id = ?", (doc_id,))
        self.db.execute(
            "UPDATE documents SET sha256=?, pages=?, ocr_pages=?, header=?, embed_model=?, extract_model=?, ingest_seconds=?, "
            "ingested_at=CURRENT_TIMESTAMP WHERE id=?",
            (sha256, pages, ocr_pages, header, embed_model, extract_model, ingest_seconds, doc_id),
        )
        self._vec_cache = None

    def chunk_texts(self, doc_id: int) -> list[tuple[int, str]]:
        """(chunk id, the text that was embedded: document header + chunk) for one document."""
        rows = self.db.execute("SELECT id, search_text FROM chunks WHERE doc_id = ? ORDER BY ordinal", (doc_id,)).fetchall()
        return [(r["id"], r["search_text"]) for r in rows]

    def set_embeddings(self, doc_id: int, ids: list[int], embeddings: np.ndarray, embed_model: str) -> None:
        """Replace one document's vectors (a different embedding model); text, index and extraction stay."""
        self.db.executemany(
            "UPDATE chunks SET embedding = ? WHERE id = ?", [(embeddings[i].astype(np.float32).tobytes(), cid) for i, cid in enumerate(ids)]
        )
        self.db.execute("UPDATE documents SET embed_model = ? WHERE id = ?", (embed_model, doc_id))
        self._vec_cache = None

    def add_chunks(self, doc_id: int, rows: list[tuple[int, int | None, str, str]], embeddings: np.ndarray) -> None:
        self.db.executemany(
            "INSERT INTO chunks(doc_id, ordinal, page, text, search_text, embedding) VALUES (?,?,?,?,?,?)",
            [(doc_id, o, p, t, st, embeddings[i].astype(np.float32).tobytes()) for i, (o, p, t, st) in enumerate(rows)],
        )
        self._vec_cache = None

    def commit(self) -> None:
        self.db.commit()

    def documents(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks FROM documents d ORDER BY d.id"
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ chunks

    def get_chunks(self, ids: list[int]) -> list[StoredChunk]:
        if not ids:
            return []
        q = (
            "SELECT c.id, c.doc_id, d.filename, c.page, c.text, d.header FROM chunks c "
            f"JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({','.join('?' * len(ids))})"
        )
        by_id = {r["id"]: StoredChunk(**dict(r)) for r in self.db.execute(q, ids)}
        return [by_id[i] for i in ids if i in by_id]

    def vectors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(chunk ids, document ids, normalized embedding matrix) - cached until the next write."""
        if self._vec_cache is None:
            rows = self.db.execute("SELECT id, doc_id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            docs = np.array([r["doc_id"] for r in rows], dtype=np.int64)
            mat = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), np.float32)
            self._vec_cache = (ids, docs, mat)
        return self._vec_cache

    def fts_search(self, match: str, limit: int, doc_ids: set[int] | None = None) -> list[int]:
        if not match:
            return []
        sql = "SELECT f.rowid FROM chunks_fts f"
        params: list = [match]
        if doc_ids:
            sql += f" JOIN chunks c ON c.id = f.rowid WHERE chunks_fts MATCH ? AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            params += sorted(doc_ids)
        else:
            sql += " WHERE chunks_fts MATCH ?"
        rows = self.db.execute(sql + " ORDER BY bm25(chunks_fts) LIMIT ?", (*params, limit)).fetchall()
        return [r[0] for r in rows]

    def add_pages(self, doc_id: int, pages: list[tuple[int | None, str]]) -> None:
        self.db.executemany("INSERT INTO pages(doc_id, page, text) VALUES (?,?,?)", [(doc_id, p, t) for p, t in pages])

    def document_pages(self, doc_id: int) -> list[tuple[int | None, str]]:
        rows = self.db.execute("SELECT page, text FROM pages WHERE doc_id = ? ORDER BY rowid", (doc_id,)).fetchall()
        return [(r["page"], r["text"]) for r in rows]

    def document_by_id(self, doc_id: int) -> sqlite3.Row:
        return self.db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()

    def doc_ids_by_filename(self) -> dict[str, int]:
        return {r["filename"]: r["id"] for r in self.db.execute("SELECT id, filename FROM documents")}

    # ------------------------------------------------------------ extractions

    def save_extraction(self, doc_id: int, schema_name: str, data: dict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO extractions(doc_id, schema_name, data) VALUES (?,?,?)",
            (doc_id, schema_name, json.dumps(data)),
        )

    def save_evidence(self, doc_id: int, rows: list[tuple[str, str | None, int | None, bool]]) -> None:
        self.db.execute("DELETE FROM evidence WHERE doc_id = ?", (doc_id,))
        self.db.executemany(
            "INSERT INTO evidence(doc_id, field, quote, page, verified) VALUES (?,?,?,?,?)",
            [(doc_id, f, q, p, int(v)) for f, q, p, v in rows],
        )

    def evidence(self) -> dict[tuple[str, str], dict]:
        """{(filename, field): {"quote", "page", "verified"}} for every extracted value."""
        rows = self.db.execute(
            "SELECT d.filename, e.field, e.quote, e.page, e.verified FROM evidence e JOIN documents d ON d.id = e.doc_id"
        ).fetchall()
        return {(r["filename"], r["field"]): {"quote": r["quote"], "page": r["page"], "verified": bool(r["verified"])} for r in rows}

    def extractions(self, schema_name: str) -> list[dict]:
        """One record per document, with manual edits applied."""
        rows = self.db.execute(
            "SELECT d.id, d.filename, e.data FROM extractions e JOIN documents d ON d.id = e.doc_id WHERE e.schema_name = ? ORDER BY d.id",
            (schema_name,),
        ).fetchall()
        edits: dict[int, dict] = {}
        for r in self.db.execute("SELECT doc_id, field, value FROM edits"):
            edits.setdefault(r["doc_id"], {})[r["field"]] = json.loads(r["value"])
        return [{"filename": r["filename"], **json.loads(r["data"]), **edits.get(r["id"], {})} for r in rows]

    def edit_value(self, doc_id: int, field: str, value, original) -> None:
        """A user's correction of an extracted value; the first original is kept for undo."""
        self.db.execute(
            "INSERT INTO edits(doc_id, field, value, original) VALUES (?,?,?,?) "
            "ON CONFLICT(doc_id, field) DO UPDATE SET value=excluded.value, edited_at=CURRENT_TIMESTAMP",
            (doc_id, field, json.dumps(value), json.dumps(original)),
        )
        self.db.commit()

    def undo_edit(self, doc_id: int, field: str) -> None:
        self.db.execute("DELETE FROM edits WHERE doc_id = ? AND field = ?", (doc_id, field))
        self.db.commit()

    def edits(self) -> dict[tuple[str, str], dict]:
        """{(filename, field): {"value", "original", "edited_at"}}."""
        rows = self.db.execute(
            "SELECT d.filename, e.field, e.value, e.original, e.edited_at FROM edits e JOIN documents d ON d.id = e.doc_id"
        ).fetchall()
        return {
            (r["filename"], r["field"]): {
                "value": json.loads(r["value"]),
                "original": json.loads(r["original"]),
                "edited_at": r["edited_at"],
            }
            for r in rows
        }
