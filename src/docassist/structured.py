"""Structured extraction at ingest time, and text-to-SQL over the extracted table."""

from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    scope_fields: list[str]
    examples: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> Schema:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        fields = [Field(**f) for f in raw["fields"]]
        for f in fields:
            if f.type not in SQL_TYPES:
                raise ValueError(f"{path}: field {f.name} has unknown type {f.type}")
        header = raw.get("header_fields", [])
        # An example is a question, or {question, shows} where `shows` names the route it demonstrates.
        examples = [e if isinstance(e, dict) else {"question": e} for e in raw.get("examples", [])]
        return cls(raw["name"], raw.get("description", ""), fields, header, raw.get("scope_fields", header), examples)

    @property
    def columns(self) -> list[str]:
        return ["filename"] + [f.name for f in self.fields]

    def describe(self) -> str:
        return "\n".join(f"- {f.name} ({f.type}): {f.description}" for f in self.fields)


# ------------------------------------------------------------------ extraction

EXTRACT_SYSTEM = """You extract structured data from business documents.
Return a single JSON object with exactly the requested keys, plus "evidence".
Rules:
- Copy values from the document; never guess. Use null when the document does not state a value.
- Dates: ISO format yyyy-mm-dd.
- Numbers: plain numbers without currency symbols or thousands separators (e.g. 2500000, 45).
- Booleans: true or false. A boolean is true only if you can quote the clause that makes it true.
- "evidence": an object mapping every field that is not null and not false to the shortest exact quote
  from the document (at most 30 words, copied character for character) that states it."""


@dataclass
class Extraction:
    record: dict  # field -> value
    quotes: dict  # field -> verbatim quote the model gave for the value


def extract(llm: LLM, schema: Schema, text: str, max_chars: int = 24000, overlap: int = 1000, progress=None) -> Extraction:
    """One record per document, with a supporting quote per field. Long documents are read
    in overlapping parts, because clauses such as governing law often sit at the very end;
    the first value found wins, and a boolean is true if any part says so."""
    parts = [text[i : i + max_chars] for i in range(0, max(len(text), 1), max_chars - overlap)]
    record: dict = {f.name: None for f in schema.fields}
    quotes: dict = {}
    done = 0
    lock = threading.Lock()

    def read(n: int, part: str) -> dict:
        nonlocal done
        where = f" (part {n} of {len(parts)}; use null for anything this part does not state)" if len(parts) > 1 else ""
        user = (
            f"Document type: {schema.description}\n\nFields:\n{schema.describe()}\n\n"
            f"Document{where}:\n<<<\n{part}\n>>>\n\nReturn the JSON object."
        )
        raw = llm.chat_json(EXTRACT_SYSTEM, user, max_tokens=1500)
        with lock:
            done += 1
            if progress:
                progress(done, len(parts))
        return raw

    if progress:
        progress(0, len(parts))
    # Parts are read side by side (up to the client's concurrency) and merged in document
    # order, so the result is the same as reading them one after another.
    workers = max(1, min(len(parts), getattr(llm, "concurrency", 1)))
    if workers == 1:
        raws = [read(n, part) for n, part in enumerate(parts, 1)]
    else:
        with ThreadPoolExecutor(workers) as pool:
            raws = list(pool.map(read, range(1, len(parts) + 1), parts))
    for raw in raws:
        evidence = raw.get("evidence") if isinstance(raw.get("evidence"), dict) else {}
        for f in schema.fields:
            v = coerce(raw.get(f.name), f.type)
            if v is not None and (record[f.name] is None or (f.type == "boolean" and v and not record[f.name])):
                record[f.name] = v
                q = evidence.get(f.name)
                if isinstance(q, str) and q.strip():
                    quotes[f.name] = q.strip()
                else:
                    quotes.pop(f.name, None)
    return Extraction(record, quotes)


def locate(quote: str, pages: list[tuple[int | None, str]]) -> tuple[bool, int | None]:
    """Find a model-given quote in the document: (found, page). Tolerates differences in
    whitespace, case and punctuation, and quotes the model shortened at either end."""
    flat = [(page, _flat(text)) for page, text in pages]
    # A quote shortened with "..." is checked piece by piece: every piece must be on the page.
    pieces = [_flat(x) for x in re.split(r"\.\.\.|\u2026", quote)]
    pieces = [x for x in pieces if len(x) >= 4]
    if not pieces:
        return False, None
    if len(pieces) > 1:
        for page, text in flat:
            if all(x in text for x in pieces):
                return True, page
        return False, None
    q = pieces[0]
    words = q.split()
    probes = [q]
    if len(words) >= 8:
        probes += [" ".join(words[:8]), " ".join(words[-8:])]
    for probe in probes:
        for page, text in flat:
            if probe in text:
                return True, page
    return False, None


def _flat(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())


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
- "sql" when the question needs data across MANY documents: listing, filtering, counting, summing, comparing, "which/all documents ...".
- "search" when the question is about ONE specific document or needs wording/clauses not in the table.
For "sql", write ONE SQLite SELECT statement over the table.
Always select filename, the identifying columns and the columns the question is about.
SQL rules: compare text with LIKE '%...%' (case-insensitive), never with =, because stored values may be longer or worded differently.
Dates are ISO text (yyyy-mm-dd), so compare them as strings. Booleans are 0/1.
Today's date is {today}.
Return JSON: {{"route": "sql" | "search", "sql": "<SELECT ...>" | null}}"""

_SQL_OK = re.compile(r"^\s*(select|with)\b", re.I)


ROUTE_CANDIDATES = """

The question may refer to one of these documents (matched by a name or part of a name - the match may be wrong):
{candidates}
Third option, "document": the question asks about ONE of these documents - what it says, its terms, dates, parties or
clauses. This is preferred over "search" whenever the question refers to one of them (a short name like "the Acme deal"
counts); return its filename as "document".
Use "sql" if the question lists, counts or compares several documents - even with one of these names in it - and filter on the name.
Use "search" only if the question is not about any of these documents.
Return JSON: {{"route": "sql" | "search" | "document", "sql": "<SELECT ...>" | null, "document": "<filename>" | null}}"""


def route(llm: LLM, schema: Schema, question: str, today: str, candidates: list[dict] | None = None) -> dict:
    """How to answer: {"route": "sql" | "search" | "document", "sql": ..., "document": ...}.

    `candidates` are the records of documents the question may name; only then can the
    route be "document" (answer from that one document)."""
    table = f"Table {schema.name}s columns:\n- filename (text)\n{schema.describe()}"
    system = ROUTE_SYSTEM.format(today=today)
    if candidates:
        listed = "\n".join("- " + " | ".join([c["filename"]] + [str(c[f]) for f in schema.header_fields if c.get(f)]) for c in candidates)
        system += ROUTE_CANDIDATES.format(candidates=listed)
    out = llm.chat_json(system, f"{table}\n\nQuestion: {question}", max_tokens=400)
    if out.get("route") == "document" and out.get("document") in {c["filename"] for c in candidates or []}:
        return {"route": "document", "sql": None, "document": out["document"]}
    if out.get("route") != "sql" or not out.get("sql"):
        return {"route": "search", "sql": None, "document": None}
    return {"route": "sql", "sql": str(out["sql"]).strip().rstrip(";"), "document": None}


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


def relax_sql(llm: LLM, schema: Schema, db: sqlite3.Connection, question: str, sql: str) -> str | None:
    """A query returned no rows. Show the model the values actually stored in the text
    columns it filters on, so an over-strict filter (exact text, a spelling the data does
    not use) can be loosened. None = the query stands, and "nothing matches" is the answer."""
    values = []
    for f in schema.fields:
        if f.type == "text" and re.search(rf"\b{f.name}\b", sql):
            found = [r[0] for r in db.execute(f"SELECT DISTINCT {f.name} FROM {schema.name}s WHERE {f.name} IS NOT NULL LIMIT 41")]
            if len(found) <= 40:
                values.append(f"{f.name}: {json.dumps(found)}")
    if not values:
        return None
    out = llm.chat_json(
        'You check SQLite queries that returned no rows. Return JSON: {"sql": "<SELECT ...>"}',
        f"Table {schema.name}s columns: {', '.join(schema.columns)}\nQuestion: {question}\nQuery (returned no rows):\n{sql}\n"
        "Values stored in the table:\n" + "\n".join(values) + "\n\n"
        "If the filter is stricter than the question (exact text, a different spelling or wording than the stored values), "
        "rewrite it to match them. If the query is right and nothing matches, return it unchanged.",
        max_tokens=400,
    )
    fixed = str(out.get("sql", "")).strip().rstrip(";")
    return fixed if fixed and fixed != sql else None


def rows_to_json(cols: list[str], rows: list[list]) -> str:
    return json.dumps([dict(zip(cols, r)) for r in rows], default=str)
