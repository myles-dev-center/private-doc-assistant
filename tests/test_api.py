import json

import pytest
from fastapi.testclient import TestClient

import docassist.api as api_module
from docassist.profiles import Profile


class FixedProfiles:
    """The API's profile registry, serving the test assistant."""

    def __init__(self, assistant):
        import threading

        self.a = assistant
        self.active = "default"
        self.list = [Profile("default", "Test", {}, "", [])]
        self.by_name = {"default": self.list[0]}
        self.lock = threading.RLock()
        self.concurrency = {}

    def get(self, name=None):
        return self.a

    def set_concurrency(self, name, n):
        self.concurrency[name] = n
        self.a.llm.concurrency = n

    def describe(self, name):
        from docassist.profiles import describe

        return describe(self.by_name[name], self.a.s)


@pytest.fixture
def client(assistant, monkeypatch):
    monkeypatch.setattr(api_module, "profiles", lambda: FixedProfiles(assistant))
    return TestClient(api_module.app)


def ndjson(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_api_roundtrip(client):
    c = client
    health = c.get("/health").json()
    assert health["documents"] == 0 and health["schema"] == "contract" and health["examples"]
    assert health["profile"]["local"] is True and health["index_problem"] is None

    r = c.post(
        "/documents",
        files=[("files", ("a.txt", b"Contract No.: AP-SA-2024-017. Payment within 45 days.")), ("files", ("evil.exe", b"MZ"))],
    )
    assert [x["status"] for x in r.json()] == ["ingested", "error"]
    assert [d["filename"] for d in c.get("/documents").json()] == ["a.txt"]

    ans = c.post("/ask", json={"question": "What are the payment terms?"}).json()
    assert ans["route"] == "search" and ans["sources"][0]["filename"] == "a.txt"
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={"question": "Payment terms?", "document": "nope.txt"}).status_code == 404

    assert c.get("/records").json()[0]["contract_id"] == "AP-SA-2024-017"
    assert {e["field"] for e in c.get("/evidence").json()} >= {"contract_id"}
    x = c.get("/records.xlsx")
    assert x.status_code == 200 and x.content[:2] == b"PK"

    pages = c.get("/documents/a.txt/pages").json()
    assert "Payment within 45 days" in pages["pages"][0]["text"] and pages["record"]["contract_id"] == "AP-SA-2024-017"

    assert c.delete("/documents/a.txt").status_code == 200
    assert c.delete("/documents/a.txt").status_code == 404


def test_ask_stream_reports_steps_then_the_answer(client):
    client.post("/documents", files=[("files", ("a.txt", b"Contract No.: AP-SA-2024-017. Payment within 45 days."))])
    events = ndjson(client.post("/ask/stream", json={"question": "What are the payment terms?"}))
    stages = [e["stage"] for e in events if e["type"] == "step"]
    assert stages[0] == "match" and "search" in stages and stages[-1] == "answer"
    assert events[-1]["type"] == "answer" and events[-1]["sources"][0]["filename"] == "a.txt"


def test_ingest_job_reports_every_stage(client):
    job = client.post("/ingest-jobs", files=[("files", ("a.txt", b"Contract No.: AP-SA-2024-017. Payment within 45 days."))]).json()
    events = ndjson(client.get(f"/ingest-jobs/{job['id']}/events"))
    stages = list(dict.fromkeys(e["stage"] for e in events if e["type"] == "step"))
    assert stages == ["parse", "extract", "verify", "chunk", "embed", "save"]
    assert events[-1]["type"] == "done" and events[-1]["results"][0]["status"] == "ingested"
    assert client.get("/ingest-jobs/nope/events").status_code == 404


def test_web_ui_is_served(client):
    assert "Document Assistant" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


def test_profiles_keep_keys_on_the_server(tmp_path, monkeypatch):
    from docassist import Settings
    from docassist.profiles import describe, load_profiles, settings_for

    p = tmp_path / "profiles.yaml"
    p.write_text(
        "default: local\n"
        "profiles:\n"
        "  local: {label: Local, llm_model: qwen}\n"
        "  cloud: {library: cloud, llm_base_url: 'https://api.example.com/v1', llm_api_key: '${TEST_KEY}', llm_model: big}\n"
        "prices:\n  big: [1, 0.1, 2]\n"
    )
    monkeypatch.delenv("TEST_KEY", raising=False)
    profiles, default = load_profiles(p)
    assert default == "local" and profiles[1].missing == ["TEST_KEY"]
    monkeypatch.setenv("TEST_KEY", "sk-secret")
    profiles, _ = load_profiles(p)
    base = Settings(data_dir=tmp_path / "data", _env_file=None)
    s = settings_for(base, profiles[1])
    assert s.llm_api_key == "sk-secret" and s.data_dir == tmp_path / "data" / "cloud"
    info = describe(profiles[1], s)
    assert info["local"] is False and info["priced"] is True and "sk-secret" not in json.dumps(info)

    p.write_text("profiles:\n  x: {llm_modle: typo}\n")
    with pytest.raises(ValueError):
        load_profiles(p)


def test_activity_shows_work_from_any_process(client, tmp_path, monkeypatch):
    import time

    from docassist import activity

    monkeypatch.setenv("DOCASSIST_ACTIVITY_DIR", str(tmp_path / "activity"))
    with activity.Activity("ingest", "Adding 2 documents", total=2, unit="documents", local_model=True) as act:
        act.update(done=1, step="model is extracting fields")
        data = client.get("/activity").json()
        (t,) = data["tasks"]
        assert t["status"] == "running" and t["done"] == 1 and data["local_busy"] is True
    assert client.get("/activity").json()["tasks"][0]["status"] == "done"

    # a task whose process died stops sending heartbeats and is shown as stopped
    dead = activity.Activity("eval", "Evaluation", total=10)
    dead._stop.set()
    dead.data["updated"] = time.time() - 100
    dead.path.write_text(__import__("json").dumps(dead.data))
    assert {t["title"]: t["status"] for t in activity.current()}["Evaluation"] == "stopped"

    # finished work leaves the list a few seconds after it ends
    old = activity.Activity("ingest", "Old import")
    old.finish()
    old.data["finished"] = time.time() - 60
    old.path.write_text(__import__("json").dumps(old.data))
    assert "Old import" not in {t["title"] for t in activity.current()}


def test_batch_ingest_reports_activity(assistant, tmp_path, monkeypatch):
    from docassist import activity

    monkeypatch.setenv("DOCASSIST_ACTIVITY_DIR", str(tmp_path / "activity"))
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("Contract No.: AP-SA-2024-017.", encoding="utf-8")
    assistant.ingest_dir(docs)
    (t,) = activity.current()
    assert t["kind"] == "ingest" and t["status"] == "done" and t["done"] == 1 and "1 of 1" in t["detail"]


def test_manage_documents_and_values(client):
    for name in ("a.txt", "b.txt", "c.txt"):
        client.post("/documents", files=[("files", (name, f"Contract No.: AP-SA-2024-0{len(name)}{name[0]}.".encode()))])
    r = client.put("/records/value", json={"filename": "a.txt", "field": "contract_value", "value": "EUR 2,000"}).json()
    assert r["value"] == 2000.0
    assert client.put("/records/value", json={"filename": "a.txt", "field": "contract_value", "value": "lots"}).status_code == 400
    assert client.put("/records/value", json={"filename": "a.txt", "field": "nope", "value": 1}).status_code == 400
    (e,) = client.get("/edits").json()
    assert e["value"] == 2000.0 and e["original"] == 1200.5
    assert client.delete("/records/value", params={"filename": "a.txt", "field": "contract_value"}).status_code == 200
    assert client.get("/edits").json() == []

    r = client.post("/documents/delete", json={"filenames": ["b.txt", "c.txt", "zzz.txt"]}).json()
    assert r == {"deleted": ["b.txt", "c.txt"], "not_found": ["zzz.txt"]}
    assert [d["filename"] for d in client.get("/documents").json()] == ["a.txt"]


def test_reprocess_runs_in_background_and_can_be_stopped(client, tmp_path):
    import time

    from docassist import activity

    client.post("/documents", files=[("files", ("a.txt", b"Contract No.: AP-SA-2024-017."))])
    task = client.post("/documents/reprocess", json={"filenames": ["a.txt"]}).json()["task"]
    for _ in range(100):
        t = next((x for x in activity.current() if x["id"] == task), None)
        if t and t["status"] != "running":
            break
        time.sleep(0.05)
    assert t["status"] == "done" and t["done"] == 1

    assert client.post("/activity/nope/stop").status_code == 404
    act = activity.Activity("eval", "Long run", total=5)
    assert client.post(f"/activity/{act.id}/stop").json() == {"stopping": act.id}
    assert act.stop_requested and next(x for x in activity.current() if x["id"] == act.id)["stopping"]
    try:
        with act:
            act.check()
    except activity.Cancelled:
        pass
    assert next(x for x in activity.current() if x["id"] == act.id)["status"] == "cancelled"


def test_parallel_uploads_and_concurrency_api(client):
    files = [("files", (f"{c}.txt", f"Contract No.: AP-SA-2024-{c}.".encode())) for c in ("011", "022", "033", "044", "055")]
    assert client.put("/concurrency", json={"value": 4}).json()["value"] == 4
    results = client.post("/documents", files=files).json()
    assert [r["status"] for r in results] == ["ingested"] * 5
    assert sorted(d["filename"] for d in client.get("/documents").json()) == [f"{c}.txt" for c in ("011", "022", "033", "044", "055")]
    assert len({d["id"] for d in client.get("/documents").json()}) == 5
    c = client.get("/concurrency").json()
    assert c["value"] == 4 and c["max"] == 10 and c["recommended"] >= 1 and c["why"]
    assert client.put("/concurrency", json={"value": 11}).status_code == 422
