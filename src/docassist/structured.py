"""Structured extraction at ingest time, and text-to-SQL over the extracted table."""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import yaml

from .llm import LLM

SQL_TYPES = {"text": "TEXT", "date": "TEXT", "number": "REAL", "boolean": "INTEGER"}


@dataclass
class Field:
    name: str
    type: str
    description: str


@dataclass
class Schema:
    name: str
    description: str
    fields: list[Field]
    header_fields: list[str]

    @classmethod
    def load(cls, path: Path) -> Schema:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        fields = [Field(**f) for f in raw["fields"]]
        for f in fields:
            if f.type not in SQL_TYPES:
                raise ValueError(f"{path}: field {f.name} has unknown type {f.type}")
        return cls(raw["name"], raw.get("description", ""), fields, raw.get("header_fields", []))

    @property
    def columns(self) -> list[str]:
        return ["filename"] + [f.name for f in self.fields]

    def describe(self) -> str:
        return "\n".join(f"- {f.name} ({f.type}): {f.description}" for f in self.fields)


# ------------------------------------------------------------------ extraction

EXTRACT_SYSTEM = """You extract structured data from business documents.
Return a single JSON object with exactly the requested keys.
Rules:
- Copy values from the document; never guess. Use null when the document does not state a value.
- Dates: ISO format yyyy-mm-dd.
- Numbers: plain numbers without currency symbols or thousands separators (e.g. 2500000, 45).
- Booleans: true or false."""


def extract(llm: LLM, schema: Schema, text: str, max_chars: int = 24000) -> dict:
    user = (
        f"Document type: {schema.description}\n\nFields:\n{schema.describe()}\n\n"
        f"Document:\n<<<\n{text[:max_chars]}\n>>>\n\nReturn the JSON object."
    )
    raw = llm.chat_json(EXTRACT_SYSTEM, user, max_tokens=700)
    return {f.name: coerce(raw.get(f.name), f.type) for f in schema.fields}


def coerce(value, ftype: str):
    """Normalize model output to the declared type; anything unparseable becomes None."""
    if value is None or (isinstance(value, str) and value.strip().lower() in {"", "null", "none", "n/a"}):
        return None
    try:
        if ftype == "number":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            m = re.search(r"-?\d[\d,]*\.?\d*", str(value))
            return float(m.group().replace(",", "")) if m else None
        if ftype == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"true", "yes", "1"}
        if ftype == "date":
            return normalize_date(str(value))
        return str(value).strip()
    except (ValueError, TypeError):
        return None


_DATE_FORMATS = ["%Y-%m-%d", "%d %B %Y", "%B %d, %Y", "%d.%m.%Y", "%d/%m/%Y", "%B %d %Y"]


def normalize_date(s: str) -> str | None:
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    return None


# ------------------------------------------------------------------ querying


def records_db(schema: Schema, rows: list[dict]) -> sqlite3.Connection:
    """Fresh in-memory DB holding only the extracted table, opened read-only for queries."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    cols = ", ".join(f"{c} {SQL_TYPES[f.type]}" for c, f in zip(schema.columns[1:], schema.fields))
    db.execute(f"CREATE TABLE {schema.name}s (filename TEXT, {cols})")
    db.executemany(
        f"INSERT INTO {schema.name}s VALUES ({','.join('?' * len(schema.columns))})",
        [[r.get(c) for c in schema.columns] for r in rows],
    )
    db.commit()
    db.execute("PRAGMA query_only = ON")
    return db


ROUTE_SYSTEM = """You route questions about a collection of documents.
The collection has also been extracted into one SQLite table (one row per document).
Decide:
- "sql" when the question needs data across MANY documents: listing, filtering, counting, summing, comparing, "which/all contracts ...".
- "search" when the question is about ONE specific document or needs wording/clauses not in the table.
For "sql", write ONE SQLite SELECT statement over the table. Always select the identifying columns plus the columns the question is about.
SQL rules: compare text with LIKE '%...%' (case-insensitive), never with =, because stored values may be longer or worded differently.
Dates are ISO text (yyyy-mm-dd), so compare them as strings. Booleans are 0/1.
Today's date is {today}.
Return JSON: {{"route": "sql" | "search", "sql": "<SELECT ...>" | null}}"""

_SQL_OK = re.compile(r"^\s*(select|with)\b", re.I)


def route(llm: LLM, schema: Schema, question: str, today: str) -> dict:
    table = f"Table {schema.name}s columns:\n- filename (text)\n{schema.describe()}"
    out = llm.chat_json(ROUTE_SYSTEM.format(today=today), f"{table}\n\nQuestion: {question}", max_tokens=400)
    if out.get("route") != "sql" or not out.get("sql"):
        return {"route": "search", "sql": None}
    return {"route": "sql", "sql": str(out["sql"]).strip().rstrip(";")}


def run_sql(db: sqlite3.Connection, sql: str, limit: int = 500) -> tuple[list[str], list[list]]:
    if not _SQL_OK.match(sql) or ";" in sql:
        raise ValueError("only a single SELECT statement is allowed")
    cur = db.execute(sql)
    cols = [d[0] for d in cur.description]
    return cols, [list(r) for r in cur.fetchmany(limit)]


def fix_sql(llm: LLM, schema: Schema, question: str, sql: str, error: str) -> str:
    out = llm.chat_json(
        'You fix SQLite queries. Return JSON: {"sql": "<SELECT ...>"}',
        f"Table {schema.name}s columns: {', '.join(schema.columns)}\nQuestion: {question}\nQuery:\n{sql}\nError: {error}",
        max_tokens=400,
    )
    return str(out.get("sql", "")).strip().rstrip(";")


def rows_to_json(cols: list[str], rows: list[list]) -> str:
    return json.dumps([dict(zip(cols, r)) for r in rows], default=str)
