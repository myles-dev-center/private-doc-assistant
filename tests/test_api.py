from fastapi.testclient import TestClient

import docassist.api as api_module


def test_api_roundtrip(assistant, monkeypatch, tmp_path):
    monkeypatch.setattr(api_module, "assistant", lambda: assistant)
    c = TestClient(api_module.app)
    health = c.get("/health").json()
    assert health["documents"] == 0 and health["schema"] == "contract" and health["examples"]

    r = c.post(
        "/documents", files=[("files", ("a.txt", b"Contract No.: AP-SA-2024-017. Payment within 45 days.")), ("files", ("evil.exe", b"MZ"))]
    )
    assert [x["status"] for x in r.json()] == ["ingested", "error"]
    assert [d["filename"] for d in c.get("/documents").json()] == ["a.txt"]

    ans = c.post("/ask", json={"question": "What are the payment terms?"}).json()
    assert ans["route"] == "search" and ans["sources"][0]["filename"] == "a.txt"
    assert c.post("/ask", json={"question": ""}).status_code == 422

    assert c.get("/records").json()[0]["contract_id"] == "AP-SA-2024-017"
    x = c.get("/records.xlsx")
    assert x.status_code == 200 and x.content[:2] == b"PK"

    assert c.delete("/documents/a.txt").status_code == 200
    assert c.delete("/documents/a.txt").status_code == 404
