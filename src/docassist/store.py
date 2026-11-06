"""SQLite storage: documents, chunks (+FTS5 index, embeddings) and extracted records.

One file, no server. Vectors are kept as float32 blobs and searched with numpy;
for a few hundred thousand chunks that is fast enough and keeps deployment trivial.
"""

from __future__ import annotations

import json
import sqlite3
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
"""


@dataclass
class StoredChunk:
    id: int
    doc_id: int
    filename: str
    page: int | None
    text: str
    header: str | None


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        self._vec_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    # ------------------------------------------------------------ documents

    def find_document(self, filename: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM documents WHERE filename = ?", (filename,)).fetchone()

    def delete_document(self, filename: str) -> None:
        self.db.execute("DELETE FROM documents WHERE filename = ?", (filename,))
        self.db.commit()
        self._vec_cache = None

    def add_document(self, filename: str, sha256: str, pages: int, ocr_pages: int, header: str | None) -> int:
        cur = self.db.execute(
            "INSERT INTO documents(filename, sha256, pages, ocr_pages, header) VALUES (?,?,?,?,?)",
            (filename, sha256, pages, ocr_pages, header),
        )
        return cur.lastrowid

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
            "SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunks FROM documents d ORDER BY filename"
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

    def extractions(self, schema_name: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT d.filename, e.data FROM extractions e JOIN documents d ON d.id = e.doc_id WHERE e.schema_name = ? ORDER BY d.filename",
            (schema_name,),
        ).fetchall()
        return [{"filename": r["filename"], **json.loads(r["data"])} for r in rows]
