"""Thin wrapper over the OpenAI-compatible API, plus a robust JSON helper."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import numpy as np
from openai import BadRequestError, OpenAI

from .config import Settings


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.chat_client = make_client(settings.llm_base_url, settings.llm_api_key, settings.llm_timeout)
        self.embed_client = make_client(
            settings.embed_base_url or settings.llm_base_url, settings.embed_api_key or settings.llm_api_key, settings.llm_timeout
        )
        self.json_mode_supported = True

    def chat(self, system: str, user: str, *, max_tokens: int = 800, json_mode: bool = False) -> str:
        kwargs: dict[str, Any] = {}
        if self.s.llm_reasoning_effort:
            kwargs["reasoning_effort"] = self.s.llm_reasoning_effort
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.chat_client.chat.completions.create(
            model=self.s.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=max_tokens,
            **kwargs,
        )
        return strip_think(resp.choices[0].message.content or "")

    def chat_json(self, system: str, user: str, *, max_tokens: int = 800) -> dict:
        """Ask for JSON; fall back to prompt-only JSON if the server rejects json_object mode."""
        if self.json_mode_supported:
            try:
                return parse_json(self.chat(system, user, max_tokens=max_tokens, json_mode=True))
            except BadRequestError:  # e.g. LM Studio only accepts json_schema
                self.json_mode_supported = False
        return parse_json(self.chat(system, user + "\n\nRespond with JSON only.", max_tokens=max_tokens))

    def embed(self, texts: list[str], *, query: bool = False, batch: int = 64) -> np.ndarray:
        prefix = self.s.embed_query_prefix if query else self.s.embed_doc_prefix
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            resp = self.embed_client.embeddings.create(model=self.s.embed_model, input=[prefix + t for t in texts[i : i + batch]])
            out.extend(d.embedding for d in resp.data)
        vecs = np.asarray(out, dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def make_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    # A system-wide HTTP proxy (common on corporate and VPN machines) must not
    # capture calls to a model server on this machine or the LAN.
    host = urlparse(base_url).hostname or ""
    local = host in {"localhost", "127.0.0.1", "::1", "host.docker.internal"} or host.startswith(("192.168.", "10."))
    http_client = httpx.Client(trust_env=False, timeout=timeout) if local else None
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
