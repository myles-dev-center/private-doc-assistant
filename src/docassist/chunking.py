"""Split page texts into overlapping chunks that keep their page number."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .parsing import ParsedDoc

_SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+|(?<=[\u3002\uff1b\uff01\uff1f])|\n+")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


@dataclass
class Chunk:
    ordinal: int
    page: int | None
    text: str


def approx_tokens(text: str) -> int:
    # About 1.3 tokens per English word; a CJK character is about one token.
    cjk = len(_CJK.findall(text))
    return max(1, round(len(_CJK.sub(" ", text).split()) * 1.3) + cjk)


def chunk_doc(doc: ParsedDoc, max_tokens: int = 220, overlap: int = 40) -> list[Chunk]:
    """Greedy sentence packing per page. Chunks never span pages, so citations stay exact."""
    chunks: list[Chunk] = []
    for page in doc.pages:
        sentences = [s.strip() for s in _SENT_SPLIT.split(page.text) if s and s.strip()]
        buf: list[str] = []
        size = 0
        for sent in sentences:
            n = approx_tokens(sent)
            if buf and size + n > max_tokens:
                chunks.append(Chunk(len(chunks), page.number, " ".join(buf)))
                # carry the tail sentences over as overlap
                keep: list[str] = []
                kept = 0
                for s in reversed(buf):
                    kept += approx_tokens(s)
                    if kept > overlap:
                        break
                    keep.insert(0, s)
                buf, size = keep, sum(approx_tokens(s) for s in keep)
            buf.append(sent)
            size += n
        if buf:
            chunks.append(Chunk(len(chunks), page.number, " ".join(buf)))
    return chunks
