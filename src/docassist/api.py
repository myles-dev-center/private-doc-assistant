"""HTTP API and web UI. Run: uv run uvicorn docassist.api:app  (UI at http://localhost:8000)"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __doc__ as package_doc
from . import activity
from .activity import Activity, Cancelled
from .config import get_settings
from .export import xlsx_bytes
from .hardware import MAX_CONCURRENCY, detect, recommend_local
from .llm import resolve_concurrency
from .parsing import SUPPORTED
from .pipeline import Assistant, IndexMismatch, stage_text
from .profiles import Profile, describe, load_profiles, settings_for
from .store import Store
from .structured import coerce

WEB = Path(__file__).parent / "web"

app = FastAPI(title="Private Document Assistant", description=package_doc, version="0.2.0")


class Profiles:
    """One Assistant per model preset; presets with the same library share its database."""

    def __init__(self):
        base = get_settings()
        self.base = base
        self.list, self.active = load_profiles(base.profiles_path)
        self.by_name: dict[str, Profile] = {p.name: p for p in self.list}
        self._assistants: dict[str, Assistant] = {}
        self._stores: dict[str, Store] = {}
        self.lock = threading.RLock()  # switching presets
        self.concurrency: dict[str, int] = {}  # per preset, changed in the UI; 0/absent = automatic

    def settings(self, name: str):
        return settings_for(self.base, self.by_name[name])

    def get(self, name: str | None = None) -> Assistant:
        name = name or self.active
        if name not in self._assistants:
            s = self.settings(name)
            key = str(s.db_path.resolve())
            if key not in self._stores:
                self._stores[key] = Store(s.db_path)
            self._assistants[name] = Assistant(s, store=self._stores[key])
            if self.concurrency.get(name):
                self._assistants[name].llm.set_concurrency(self.concurrency[name])
        return self._assistants[name]

    def set_concurrency(self, name: str, n: int) -> None:
        self.concurrency[name] = n
        self.get(name).llm.set_concurrency(n)

    def describe(self, name: str) -> dict:
        return describe(self.by_name[name], self.settings(name))


@lru_cache
def profiles() -> Profiles:
    return Profiles()


def assistant() -> Assistant:
    return profiles().get()


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    document: str | None = Field(None, description="answer from this document only (a filename)")


class ProfileRequest(BaseModel):
    name: str


@app.get("/health")
def health() -> dict:
    p, a = profiles(), assistant()
    return {
        "status": "ok",
        "model": a.s.llm_model,
        "embed_model": a.s.embed_model,
        "documents": len(a.store.documents()),
        "schema": a.schema.name,
        "fields": [{"name": f.name, "type": f.type, "description": f.description} for f in a.schema.fields],
        "header_fields": a.schema.header_fields,
        "examples": a.schema.examples,
        "profile": p.describe(p.active),
        "index_problem": a.index_problem(),
    }


class ConcurrencyRequest(BaseModel):
    value: int = Field(ge=1, le=MAX_CONCURRENCY)


@app.get("/concurrency")
def get_concurrency() -> dict:
    """Parallel model requests for the active preset, and what is recommended for it."""
    p = profiles()
    a = assistant()
    local = a.uses_local_model()
    if local:
        rec, why = recommend_local()
    else:
        rec, why = 4, "Cloud API: limited by the provider's rate limit and your budget, not this machine. 4 is a safe start."
    return {
        "value": a.llm.concurrency,
        "automatic": not p.concurrency.get(p.active) and not a.s.concurrency,
        "recommended": rec,
        "why": why,
        "local": local,
        "max": MAX_CONCURRENCY,
        "hardware": detect(),
        "default": resolve_concurrency(a.s),
    }


@app.put("/concurrency")
def set_concurrency(req: ConcurrencyRequest) -> dict:
    p = profiles()
    p.set_concurrency(p.active, req.value)
    return get_concurrency()


@app.get("/profiles")
def list_profiles() -> dict:
    p = profiles()
    return {"active": p.active, "profiles": [p.describe(x.name) for x in p.list]}


@app.put("/profiles/active")
def set_profile(req: ProfileRequest) -> dict:
    p = profiles()
    if req.name not in p.by_name:
        raise HTTPException(404, "unknown profile")
    info = p.describe(req.name)
    if not info["available"]:
        raise HTTPException(409, f"profile {req.name} needs environment variables: {', '.join(info['missing'])}")
    with p.lock:
        p.active = req.name
    return list_profiles()


@app.get("/documents")
def documents() -> list[dict]:
    return assistant().store.documents()


@app.get("/documents/{filename}/pages")
def pages(filename: str) -> dict:
    a = assistant()
    doc = a.store.find_document(filename)
    if not doc:
        raise HTTPException(404, "document not found")
    rec = next((r for r in a.records() if r["filename"] == filename), None)
    evidence = [{"field": f, **v} for (fn, f), v in a.store.evidence().items() if fn == filename]
    return {
        "filename": filename,
        "pages": [{"page": p, "text": t} for p, t in a.store.document_pages(doc["id"])],
        "record": rec,
        "evidence": evidence,
    }


def _save_uploads(files: list[UploadFile], folder: Path) -> list[tuple[str, Path | None]]:
    saved = []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in SUPPORTED:
            saved.append((name, None))
            continue
        path = folder / name
        with path.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append((name, path))
    return saved


def _ingest(a: Assistant, saved: list[tuple[str, Path | None]], emit=None, act: Activity | None = None) -> list[dict]:
    """Ingest uploaded files, up to the model client's concurrency side by side."""
    results: list[dict | None] = [None] * len(saved)

    def one(i: int) -> None:
        name, path = saved[i]
        if act:
            act.check()
        if path is None:
            results[i] = {"filename": name, "status": "error", "error": "unsupported file type"}
        else:
            if emit:
                emit({"type": "file", "index": i, "files": len(saved), "filename": name})

            def step(stage, done=0, total=0, **info):
                if act:
                    act.check()
                if emit:
                    emit({"type": "step", "index": i, "filename": name, "stage": stage, "done": done, "total": total, **info})

            try:
                results[i] = a.ingest_file(path, progress=step)
            except Cancelled:
                raise
            except Exception as e:
                results[i] = {"filename": name, "status": "error", "error": str(e)}
        if emit:
            emit({"type": "result", "index": i, "files": len(saved), **results[i]})

    _side_by_side(one, len(saved), a.llm.concurrency)
    return [r for r in results if r is not None]


def _side_by_side(fn, n: int, workers: int) -> None:
    """fn(0..n-1) on up to `workers` threads; the first Cancelled (or other error) stops new work."""
    from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

    if workers <= 1 or n <= 1:
        for i in range(n):
            fn(i)
        return
    with ThreadPoolExecutor(min(workers, n)) as pool:
        futures = [pool.submit(fn, i) for i in range(n)]
        done, pending = wait(futures, return_when=FIRST_EXCEPTION)
        for f in pending:
            f.cancel()
        for f in futures:
            if f.done() and not f.cancelled() and f.exception():
                raise f.exception()


@app.post("/documents")
def upload(files: list[UploadFile]) -> list[dict]:
    """Upload and ingest; returns when all files are done. The UI uses /ingest-jobs for progress."""
    with tempfile.TemporaryDirectory() as tmp:
        return _ingest(assistant(), _save_uploads(files, Path(tmp)))


class Job:
    """A background ingestion; its events are kept so a reconnecting page can replay them."""

    def __init__(self, files: int):
        self.id = uuid.uuid4().hex[:12]
        self.events: list[dict] = []
        self.done = False
        self.files = files
        self.cond = threading.Condition()
        self.started = time.time()

    def emit(self, event: dict) -> None:
        with self.cond:
            self.events.append({**event, "t": round(time.time() - self.started, 1)})
            self.cond.notify_all()


JOBS: dict[str, Job] = {}


@app.post("/ingest-jobs")
def start_ingest(files: list[UploadFile]) -> dict:
    """Start ingesting in the background; follow progress at /ingest-jobs/{id}/events."""
    tmp = Path(tempfile.mkdtemp(prefix="docassist-"))
    saved = _save_uploads(files, tmp)
    job = Job(len(saved))
    JOBS[job.id] = job
    a = assistant()
    act = Activity(
        "upload",
        f"Uploading {len(saved)} document{'s' if len(saved) != 1 else ''}",
        total=len(saved),
        unit="documents",
        model=a.s.llm_model,
        local_model=a.uses_local_model(),
    )

    finished = 0

    def track(event: dict) -> None:
        nonlocal finished
        job.emit(event)
        if event["type"] == "file":
            act.update(detail=event["filename"], step="waiting for the model")
        elif event["type"] == "step":
            act.update(detail=event["filename"], step=stage_text(event["stage"], event["done"], event["total"], ocr=event.get("ocr")))
        elif event["type"] == "result":
            finished += 1
            act.update(done=finished)

    def run():
        try:
            results = _ingest(a, saved, track, act)
            job.emit({"type": "done", "results": results})
            failed = sum(r["status"] == "error" for r in results)
            act.finish("done", f"{len(results) - failed} of {len(results)} documents ready" + (f", {failed} failed" if failed else ""))
        except Cancelled:
            job.emit({"type": "done", "cancelled": True, "results": []})
            act.finish("cancelled", "stopped by user")
        except Exception as e:  # never leave a job hanging
            job.emit({"type": "done", "error": str(e), "results": []})
            act.finish("error", str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            with job.cond:
                job.done = True
                job.cond.notify_all()

    threading.Thread(target=run, daemon=True).start()
    return {"id": job.id, "files": [n for n, _ in saved]}


@app.get("/ingest-jobs/{job_id}/events")
def ingest_events(job_id: str) -> StreamingResponse:
    """Newline-delimited JSON, one event per line, from the start of the job until it ends."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    def stream():
        sent = 0
        while True:
            with job.cond:
                if sent == len(job.events) and not job.done:
                    job.cond.wait(timeout=15)
                new = job.events[sent:]
                finished = job.done and sent + len(new) == len(job.events)
            sent += len(new)
            # an empty line keeps the connection alive during a long step
            yield "".join(json.dumps(e, default=str) + "\n" for e in new) or "\n"
            if finished:
                return

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.delete("/documents/{filename}")
def delete(filename: str) -> dict:
    if not assistant().store.find_document(filename):
        raise HTTPException(404, "document not found")
    assistant().store.delete_document(filename)
    return {"deleted": filename}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return assistant().ask(req.question, req.document).to_dict()
    except IndexMismatch as e:
        raise HTTPException(409, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, "document not found") from e


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """Like /ask, as newline-delimited JSON: one {"type": "step"} per step, then {"type": "answer"}."""
    a = assistant()
    events: queue.Queue = queue.Queue()
    t0 = time.time()

    def run():
        act = Activity("question", req.question[:120], model=a.s.llm_model, local_model=a.uses_local_model())

        def step(stage, done=0, total=0, **info):
            events.put({"type": "step", "stage": stage, "done": done, "total": total, "t": round(time.time() - t0, 1), **info})
            act.update(step=stage)

        try:
            act.update(step="waiting for other work to finish")
            ans = a.ask(req.question, req.document, progress=step)
            act.finish("done", ans.route)
            events.put({"type": "answer", **ans.to_dict()})
        except IndexMismatch as e:
            events.put({"type": "error", "error": str(e)})
        except KeyError:
            events.put({"type": "error", "error": "document not found"})
        except Exception as e:
            events.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
        if act.data["status"] == "running":
            act.finish("error")
        events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while (e := events.get()) is not None:
            yield json.dumps(e, default=str) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/activity")
def list_activity() -> dict:
    """Running and recently finished work on this machine: ingestion (UI or command line),
    evaluation runs and questions; `local_busy` = the local model is in use by other work."""
    tasks = activity.current()
    return {"tasks": tasks, "local_busy": any(t["status"] == "running" and t["local_model"] for t in tasks)}


@app.post("/activity/{task_id}/stop")
def stop_task(task_id: str) -> dict:
    """Ask a running task to stop after its current step (also tasks started from the command line)."""
    if not activity.request_stop(task_id):
        raise HTTPException(404, "unknown task")
    return {"stopping": task_id}


class FilesRequest(BaseModel):
    filenames: list[str] = Field(min_length=1)


@app.post("/documents/delete")
def delete_many(req: FilesRequest) -> dict:
    a = assistant()
    deleted = [f for f in req.filenames if a.store.find_document(f)]
    for f in deleted:
        a.store.delete_document(f)
    return {"deleted": deleted, "not_found": sorted(set(req.filenames) - set(deleted))}


@app.post("/documents/reprocess")
def reprocess(req: FilesRequest) -> dict:
    """Extract and index the documents again with the current models, in the background;
    progress (and a stop button) appear in the activity panel."""
    a = assistant()
    names = list(dict.fromkeys(req.filenames))
    act = Activity(
        "reprocess",
        f"Re-processing {len(names)} document{'s' if len(names) != 1 else ''} with {a.s.llm_model}",
        total=len(names),
        unit="documents",
        model=a.s.llm_model,
        local_model=a.uses_local_model(),
    )

    def run():
        done = failed = 0
        count = threading.Lock()

        def one(i: int) -> None:
            nonlocal done, failed
            name = names[i]
            act.check()

            def step(stage, d=0, total=0, **info):
                act.check()
                act.update(detail=name, step=stage_text(stage, d, total, **info))

            try:
                a.reprocess(name, progress=step)
                ok = True
            except Cancelled:
                raise
            except Exception as e:
                ok = False
                act.update(detail=f"{name}: {e}")
            with count:
                done, failed = done + ok, failed + (not ok)
                act.update(done=done + failed)

        try:
            _side_by_side(one, len(names), a.llm.concurrency)
            act.finish("done", f"{done} of {len(names)} re-processed" + (f", {failed} failed" if failed else ""))
        except Cancelled:
            act.finish("cancelled", f"stopped by user after {done} of {len(names)}")

    threading.Thread(target=run, daemon=True).start()
    return {"task": act.id, "documents": names}


class EditRequest(BaseModel):
    filename: str
    field: str
    value: str | float | bool | None


@app.put("/records/value")
def edit_value(req: EditRequest) -> dict:
    """Correct one extracted value by hand. Questions over the table use the corrected value."""
    a = assistant()
    f = next((x for x in a.schema.fields if x.name == req.field), None)
    if f is None:
        raise HTTPException(400, f"unknown field {req.field}")
    doc = a.store.find_document(req.filename)
    if not doc:
        raise HTTPException(404, "document not found")
    value = coerce(req.value, f.type) if req.value not in (None, "") else None
    if req.value not in (None, "") and value is None:
        raise HTTPException(400, f"{req.value!r} is not a valid {f.type}")
    prior = a.store.edits().get((req.filename, req.field))
    current = next((r for r in a.records() if r["filename"] == req.filename), {}).get(req.field)
    a.store.edit_value(doc["id"], req.field, value, prior["original"] if prior else current)
    return {"filename": req.filename, "field": req.field, "value": value}


@app.delete("/records/value")
def undo_edit(filename: str, field: str) -> dict:
    a = assistant()
    doc = a.store.find_document(filename)
    if not doc:
        raise HTTPException(404, "document not found")
    a.store.undo_edit(doc["id"], field)
    return {"filename": filename, "field": field, "restored": True}


@app.get("/edits")
def list_edits() -> list[dict]:
    return [{"filename": fn, "field": f, **v} for (fn, f), v in assistant().store.edits().items()]


@app.get("/records")
def records() -> list[dict]:
    return assistant().records()


@app.get("/evidence")
def evidence() -> list[dict]:
    """The quote and page behind every extracted value."""
    return [{"filename": fn, "field": f, **v} for (fn, f), v in assistant().store.evidence().items()]


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


@app.get("/", include_in_schema=False)
def index() -> Response:
    # Stamp asset links with their modification time so a browser never runs stale JS/CSS.
    html = (WEB / "index.html").read_text(encoding="utf-8")
    for name in ("app.css", "app.js"):
        html = html.replace(f"/static/{name}", f"/static/{name}?v={int((WEB / name).stat().st_mtime)}")
    return Response(html, media_type="text/html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=WEB), name="static")
