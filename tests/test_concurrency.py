"""Parallel model requests: the cap is respected, results don't depend on it, costs stay per answer."""

import threading
import time
from types import SimpleNamespace

from docassist import Settings
from docassist.hardware import recommend_local
from docassist.llm import LLM, resolve_concurrency


def test_recommendation_follows_hardware():
    assert recommend_local({"gpu": None, "vram_gb": None, "ram_gb": 32})[0] == 1
    assert recommend_local({"gpu": "Intel Arc", "vram_gb": None, "ram_gb": 32})[0] == 1
    assert recommend_local({"gpu": "RTX 4060", "vram_gb": 8.0, "ram_gb": 32})[0] == 1
    assert recommend_local({"gpu": "RTX 4070", "vram_gb": 12.0, "ram_gb": 32})[0] == 2
    assert recommend_local({"gpu": "RTX 4090", "vram_gb": 24.0, "ram_gb": 64})[0] == 4


def test_setting_is_capped_and_cloud_defaults_to_4():
    local = Settings(_env_file=None, llm_base_url="http://localhost:1234/v1")
    cloud = Settings(_env_file=None, llm_base_url="https://api.example.com/v1")
    assert resolve_concurrency(cloud) == 4
    assert resolve_concurrency(local.model_copy(update={"concurrency": 50})) == 10
    assert resolve_concurrency(cloud.model_copy(update={"concurrency": 7})) == 7


class SlowClient:
    """Stands in for the OpenAI client: records how many calls overlap."""

    def __init__(self):
        self.now = self.peak = 0
        self.lock = threading.Lock()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.05)
        with self.lock:
            self.now -= 1
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2, prompt_tokens_details=None)
        return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


def test_requests_in_flight_never_exceed_the_setting():
    llm = LLM(Settings(_env_file=None, llm_base_url="https://api.example.com/v1", concurrency=3))
    llm.chat_client = SlowClient()
    threads = [threading.Thread(target=llm.chat, args=("s", "u")) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert llm.chat_client.peak == 3 and llm.usage.calls == 12


def test_usage_of_one_answer_excludes_parallel_work():
    llm = LLM(Settings(_env_file=None, llm_base_url="https://api.example.com/v1", concurrency=4))
    llm.chat_client = SlowClient()
    background = [threading.Thread(target=llm.chat, args=("s", "u")) for _ in range(5)]
    for t in background:
        t.start()
    with llm.track() as mine:
        llm.chat("s", "u")
        llm.chat("s", "u")
    for t in background:
        t.join()
    assert mine.calls == 2 and mine.input == 20 and llm.usage.calls == 7


def test_cloud_calls_share_one_cache_key_local_calls_have_none():
    cloud = LLM(Settings(_env_file=None, llm_base_url="https://api.example.com/v1"))
    sent = []

    def create(**kw):
        sent.append(kw)
        return SlowClient().create()

    cloud.chat_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cloud.chat("s", "u")
    cloud.chat("s", "v")
    assert cloud.cache_key and all(k["extra_body"]["prompt_cache_key"] == cloud.cache_key for k in sent)
    assert all(k["extra_headers"]["session-id"] == cloud.cache_key for k in sent)
    assert LLM(Settings(_env_file=None, llm_base_url="http://localhost:1234/v1")).cache_key is None
    assert LLM(Settings(_env_file=None, llm_base_url="https://api.example.com/v1", llm_cache_key=False)).cache_key is None


def test_parallel_extraction_matches_sequential():
    from docassist.pipeline import DEFAULT_SCHEMA
    from docassist.structured import Schema, extract

    schema = Schema.load(DEFAULT_SCHEMA)

    class PartLLM:
        def __init__(self, n):
            self.concurrency = n

        def chat_json(self, system, user, *, max_tokens=800):
            time.sleep(0.01 * (5 - int(user.split("part ")[1].split(" ")[0])))  # later parts finish first
            if "part 1 of" in user:
                return {"contract_id": "AP-SA-2024-017", "evidence": {"contract_id": "No. AP-SA-2024-017"}}
            if "part 4 of" in user:
                return {"contract_id": "WRONG", "governing_law": "Germany", "evidence": {"governing_law": "laws of Germany"}}
            return {"governing_law": "France" if "part 3 of" in user else None}

    text = "x" * 90_000
    one = extract(PartLLM(1), schema, text, max_chars=24_000)
    many = extract(PartLLM(4), schema, text, max_chars=24_000)
    assert one == many and many.record["contract_id"] == "AP-SA-2024-017" and many.record["governing_law"] == "France"


def test_final_status_survives_a_reader_holding_the_file(monkeypatch):
    # On Windows, replacing a file fails while another process reads it. The last write
    # of a task must still land, or the UI shows it running until it goes stale.
    import os

    from docassist import activity

    act = activity.Activity("eval", "t", total=1)
    real, fails = os.replace, {"n": 3}

    def busy(src, dst):
        if fails["n"]:
            fails["n"] -= 1
            raise PermissionError("in use")
        real(src, dst)

    monkeypatch.setattr(activity.os, "replace", busy)
    act.finish("done")
    assert next(x for x in activity.current() if x["id"] == act.id)["status"] == "done"
