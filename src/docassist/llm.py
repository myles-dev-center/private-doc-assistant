"""Thin wrapper over the OpenAI-compatible API, plus a robust JSON helper."""

from __future__ import annotations

import json
import re
import threading
import uuid
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from openai import BadRequestError, OpenAI

from .config import Settings

# Tried in order. LM Studio rejects json_object but accepts json_schema, which also
# constrains generation, so long SQL strings can't come back as broken JSON.
JSON_FORMATS = [
    {"type": "json_object"},
    {"type": "json_schema", "json_schema": {"name": "result", "schema": {"type": "object"}}},
]


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.chat_client = make_client(settings.llm_base_url, settings.llm_api_key, settings.llm_timeout)
        self.embed_client = make_client(
            settings.embed_base_url or settings.llm_base_url, settings.embed_api_key or settings.llm_api_key, settings.llm_timeout
        )
        self.json_format = 0  # index into JSON_FORMATS; len(JSON_FORMATS) = prompt-only
        self.cache_key = f"docassist-{uuid.uuid4()}" if settings.llm_cache_key and not is_local_url(settings.llm_base_url) else None
        self.usage = Usage()
        self._local = threading.local()  # usage of the calls made on this thread, see track()
        self.set_concurrency(resolve_concurrency(settings))

    def set_concurrency(self, n: int) -> None:
        """At most n model requests in flight at once, across all threads using this client."""
        self.concurrency = n
        self._gate = threading.BoundedSemaphore(n)

    @contextmanager
    def track(self):
        """Usage of the calls made inside this block on this thread only - so the cost of one
        answer is right even while uploads run side by side on the same client."""
        mine = Usage()
        outer = getattr(self._local, "usage", None)
        self._local.usage = mine
        try:
            yield mine
        finally:
            self._local.usage = outer
            if outer is not None:
                outer.merge(mine)

    def _count(self, kind: str, u, model: str) -> None:
        for target in (self.usage, getattr(self._local, "usage", None)):
            if target is not None:
                (target.add if kind == "chat" else target.add_embed)(u, model)

    def chat(self, system: str, user: str, *, max_tokens: int = 800, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {}
        if self.s.llm_reasoning_effort:
            kwargs["reasoning_effort"] = self.s.llm_reasoning_effort
        if json_mode:
            kwargs["response_format"] = JSON_FORMATS[self.json_format]
        if self.cache_key:
            kwargs["extra_body"] = {"prompt_cache_key": self.cache_key}
            kwargs["extra_headers"] = {"session-id": self.cache_key}
        with self._gate:
            resp = self.chat_client.chat.completions.create(
                model=self.s.llm_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0,
                max_tokens=max_tokens,
                **kwargs,
            )
        self._count("chat", resp.usage, self.s.llm_model)
        return strip_think(resp.choices[0].message.content or "")

    def chat_json(self, system: str, user: str, *, max_tokens: int = 800) -> dict:
        """Ask for JSON in the strictest mode the server accepts; prompt-only JSON as a last resort."""
        while self.json_format < len(JSON_FORMATS):
            try:
                return parse_json(self.chat(system, user, max_tokens=max_tokens, json_mode=True))
            except BadRequestError:
                self.json_format += 1
        return parse_json(self.chat(system, user + "\n\nRespond with JSON only.", max_tokens=max_tokens))

    def embed(self, texts: list[str], *, query: bool = False, batch: int = 64, progress=None) -> np.ndarray:
        """`progress(done, total)` is called after each batch."""
        prefix = self.s.embed_query_prefix if query else self.s.embed_doc_prefix
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            with self._gate:
                resp = self.embed_client.embeddings.create(model=self.s.embed_model, input=[prefix + t for t in texts[i : i + batch]])
            out.extend(d.embedding for d in resp.data)
            self._count("embed", getattr(resp, "usage", None), self.s.embed_model)
            if progress:
                progress(len(out), len(texts))
        vecs = np.asarray(out, dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


# Official list prices, USD per 1M tokens: (input, cached input, output, long-prompt threshold).
# A request over the threshold is billed at 2x input and 1.5x output as a whole.
# Models not listed (local ones) cost nothing.
PRICES = {
    "gpt-6.1-sol": (2.00, 0.10, 10.00, 272_000),
    "openai/text-embedding-3-small": (0.02, 0.02, 0.0, 0),
}


def call_cost(model: str, inp: int, cached: int, out: int) -> float:
    if model not in PRICES:
        return 0.0
    p_in, p_cached, p_out, long_at = PRICES[model]
    k_in, k_out = (2.0, 1.5) if long_at and inp > long_at else (1.0, 1.0)
    return ((inp - cached) * p_in * k_in + cached * p_cached * k_in + out * p_out * k_out) / 1e6


class Usage:
    """Token counts and list-price cost across calls, for cost reports."""

    def __init__(self):
        self.calls = self.input = self.cached = self.output = self.embed_tokens = 0
        self.usd = 0.0
        self._lock = threading.Lock()

    def merge(self, other: Usage) -> None:
        with self._lock:
            self.calls += other.calls
            self.input += other.input
            self.cached += other.cached
            self.output += other.output
            self.embed_tokens += other.embed_tokens
            self.usd += other.usd

    def add(self, u, model: str = "") -> None:
        with self._lock:
            self.calls += 1
            if u is None:
                return
            inp, out = u.prompt_tokens or 0, u.completion_tokens or 0
            details = getattr(u, "prompt_tokens_details", None)
            cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
            self.input, self.output, self.cached = self.input + inp, self.output + out, self.cached + cached
            self.usd += call_cost(model, inp, cached, out)

    def add_embed(self, u, model: str) -> None:
        with self._lock:
            n = (u.prompt_tokens or 0) if u is not None else 0
            self.embed_tokens += n
            self.usd += call_cost(model, n, 0, 0)

    def snapshot(self) -> dict:
        return {
            "calls": self.calls,
            "input": self.input,
            "cached": self.cached,
            "output": self.output,
            "embed_tokens": self.embed_tokens,
            "usd": round(self.usd, 4),
        }


def is_local_url(url: str) -> bool:
    """True for a model server on this machine or the LAN."""
    host = urlparse(url).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"} or host.startswith(("192.168.", "10."))


def resolve_concurrency(s: Settings) -> int:
    """The concurrency to use: the setting, or the recommendation when it is 0 (automatic)."""
    from .hardware import MAX_CONCURRENCY, recommend_local

    if s.concurrency:
        return max(1, min(MAX_CONCURRENCY, s.concurrency))
    return recommend_local()[0] if is_local_url(s.llm_base_url) else 4


def make_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    # A system-wide HTTP proxy (common on corporate and VPN machines) must not
    # capture calls to a model server on this machine or the LAN.
    http_client = httpx.Client(trust_env=False, timeout=timeout) if is_local_url(base_url) else None
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, http_client=http_client)


def strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that some reasoning models leave in content."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])
