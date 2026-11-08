"""Progress of long-running work - ingestion, evaluation runs, questions - shared between processes.

Every task writes a small JSON file to a shared folder (default: <temp>/docassist-activity),
so the web UI can show work started from the command line or by scripts, not only its own.
A task rewrites its file on every update and at least every 15 s; one that stops doing so
(crashed, killed) is shown as stopped.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

HEARTBEAT = 15  # seconds between rewrites while nothing else happens
STALE = 90  # no heartbeat for this long: the process is gone
KEEP = {"done": 6, "cancelled": 6}  # seconds a task stays listed after it ended, by status
KEEP_DEFAULT = 120  # failed or stopped: long enough to notice


def folder() -> Path:
    return Path(os.environ.get("DOCASSIST_ACTIVITY_DIR") or Path(tempfile.gettempdir()) / "docassist-activity")


class Cancelled(Exception):
    """A user stopped the task from the UI."""


def request_stop(task_id: str) -> bool:
    """Ask a running task (in any process) to stop after its current step."""
    path = folder() / f"{task_id}.json"
    if not path.exists():
        return False
    (folder() / f"{task_id}.stop").touch()
    return True


class Activity:
    """One task. Use as a context manager, or call finish(); update() as work progresses.

    kind: "ingest" | "upload" | "eval" | "question"; `local_model` marks work that keeps the
    local model server busy (questions on a local preset then have to wait).
    """

    def __init__(self, kind: str, title: str, *, total: int = 0, unit: str = "", model: str = "", local_model: bool = False):
        now = time.time()
        self.id = f"{kind}-{uuid.uuid4().hex[:10]}"
        self.data = {
            "id": self.id,
            "kind": kind,
            "title": title,
            "unit": unit,
            "model": model,
            "local_model": local_model,
            "done": 0,
            "total": total,
            "step": "",
            "detail": "",
            "eta": None,
            "status": "running",
            "started": now,
            "updated": now,
            "pid": os.getpid(),
        }
        self.path = folder() / f"{self.id}.json"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write()
        threading.Thread(target=self._beat, daemon=True).start()

    def update(self, **fields) -> None:
        with self._lock:
            self.data.update(fields)
            self._write()

    @property
    def stop_requested(self) -> bool:
        return (self.path.with_suffix(".stop")).exists()

    def check(self) -> None:
        """Raise Cancelled if a user asked this task to stop; call between steps."""
        if self.stop_requested:
            raise Cancelled(self.data["title"])

    def finish(self, status: str = "done", detail: str | None = None) -> None:
        with self._lock:
            self.data["status"] = status
            if detail is not None:
                self.data["detail"] = detail
            self.data["finished"] = time.time()
            self._write()
        self._stop.set()
        self.path.with_suffix(".stop").unlink(missing_ok=True)

    def __enter__(self) -> Activity:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.data["status"] == "running":
            if exc_type is Cancelled:
                self.finish("cancelled", "stopped by user")
            else:
                self.finish("error", f"{exc_type.__name__}: {exc}") if exc else self.finish()

    def _beat(self) -> None:
        while not self._stop.wait(HEARTBEAT):
            with self._lock:
                self._write()

    def _write(self) -> None:
        self.data["updated"] = time.time()
        tmp = self.path.with_name(self.path.name + ".tmp")
        # On Windows the replace fails while another process is reading the file. A lost
        # progress update is harmless, but a lost final status is not: the heartbeat has
        # stopped, so the task would look "running" until it goes stale. Retry briefly.
        for attempt in range(20):
            try:
                tmp.write_text(json.dumps(self.data, default=str), encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except OSError:
                time.sleep(0.01 * (attempt + 1))


def current() -> list[dict]:
    """Running tasks first (newest first), then recently finished ones."""
    now = time.time()
    out = []
    base = folder()
    if not base.exists():
        return out
    for p in base.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("status") == "running" and now - d.get("updated", 0) > STALE:
            d["status"] = "stopped"
            d.setdefault("finished", d.get("updated"))
        d["stopping"] = d.get("status") == "running" and (base / f"{d['id']}.stop").exists()
        if d.get("status") != "running" and now - d.get("finished", d.get("updated", 0)) > KEEP.get(d.get("status"), KEEP_DEFAULT):
            for f in (p, p.with_suffix(".stop")):
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
            continue
        d["elapsed"] = round((d.get("finished") if d.get("status") != "running" else now) - d["started"], 1)
        out.append(d)
    return sorted(out, key=lambda d: (d["status"] != "running", -d["started"]))
