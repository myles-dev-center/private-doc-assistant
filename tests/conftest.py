"""Shared fixtures: a deterministic fake LLM so pipeline tests run offline in CI."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest

from docassist import Assistant, Settings


class FakeLLM:
    """Bag-of-words embeddings and canned chat replies - enough to exercise the plumbing."""

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.json_mode_supported = True

    def embed(self, texts, *, query=False, batch=64):
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

    def chat(self, system, user, *, max_tokens=800, json_mode=False):
        self.calls.append((system, user))
        if "numbered excerpts" in system:
            m = re.search(r"\[1\][^\n]*\n(.*?)(?:\n\n\[2\]|\n\nQuestion:)", user, re.S)
            return f"{m.group(1)[:80]} [1]" if m and "Zorblax" not in user else "NOT_FOUND"
        if "database query" in system:
            return "Rows: " + user.split("Rows", 1)[1]
        return "ok"

    def chat_json(self, system, user, *, max_tokens=800):
        if system.startswith("You extract"):
            cid = re.search(r"AP-[A-Z]+-\d{4}-\d{3}", user.split("<<<", 1)[-1])  # document body, not the prompt's example
            return {
                "contract_id": cid.group() if cid else None,
                "counterparty": "Test Co",
                "end_date": "2027-01-31",
                "contract_type": "Supply Agreement",
                "auto_renewal": "yes",
                "contract_value": "EUR 1,200.50",
            }
        if system.startswith("You route"):
            if user.rstrip().endswith("?") and "List" in user:
                return {"route": "sql", "sql": "SELECT contract_id, end_date FROM contracts WHERE end_date LIKE '2027%'"}
            return {"route": "search", "sql": None}
        return json.loads("{}")


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def assistant(tmp_path, fake_llm):
    a = Assistant(Settings(data_dir=tmp_path / "store", _env_file=None))
    a.llm = fake_llm
    a.retriever.llm = fake_llm
    return a
