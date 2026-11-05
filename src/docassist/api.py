"""HTTP API. Run: uv run uvicorn docassist.api:app --reload"""

from __future__ import annotations

import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from . import __doc__ as package_doc
from .config import get_settings
from .export import xlsx_bytes
from .parsing import SUPPORTED
from .pipeline import Assistant

app = FastAPI(title="Private Document Assistant", description=package_doc, version="0.1.0")


@lru_cache
def assistant() -> Assistant:
    return Assistant(get_settings())


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "model": s.llm_model, "embed_model": s.embed_model, "documents": len(assistant().store.documents())}


@app.get("/documents")
def documents() -> list[dict]:
    return assistant().store.documents()


@app.post("/documents")
def upload(files: list[UploadFile]) -> list[dict]:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = Path(f.filename or "").name
            if Path(name).suffix.lower() not in SUPPORTED:
                results.append({"filename": name, "status": "error", "error": "unsupported file type"})
                continue
            path = Path(tmp) / name
            with path.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            try:
                results.append(assistant().ingest_file(path))
            except Exception as e:
                results.append({"filename": name, "status": "error", "error": str(e)})
    return results


@app.delete("/documents/{filename}")
def delete(filename: str) -> dict:
    if not assistant().store.find_document(filename):
        raise HTTPException(404, "document not found")
    assistant().store.delete_document(filename)
    return {"deleted": filename}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return assistant().ask(req.question).to_dict()


@app.get("/records")
def records() -> list[dict]:
    return assistant().records()


@app.get("/records.xlsx")
def records_xlsx() -> Response:
    a = assistant()
    cols = a.schema.columns
    data = xlsx_bytes(cols, [[r.get(c) for c in cols] for r in a.records()], a.schema.name + "s")
    return Response(
        data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{a.schema.name}s.xlsx"'},
    )
