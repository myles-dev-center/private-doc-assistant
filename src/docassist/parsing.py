"""Turn files into page texts. PDF (with OCR fallback for scanned pages) and DOCX."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

log = logging.getLogger(__name__)

SUPPORTED = {".pdf", ".docx", ".txt", ".md"}
# A page with less extractable text than this but with images is treated as scanned.
MIN_TEXT_CHARS = 40


@dataclass
class Page:
    number: int | None  # 1-based; None for formats without pages (DOCX, TXT)
    text: str
    ocr: bool = False


@dataclass
class ParsedDoc:
    filename: str
    pages: list[Page] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    @property
    def ocr_pages(self) -> int:
        return sum(p.ocr for p in self.pages)


def parse_file(path: Path, progress=None) -> ParsedDoc:
    """`progress(page, pages, ocr)` is called after each PDF page."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path, progress)
    if ext == ".docx":
        return parse_docx(path)
    if ext in {".txt", ".md"}:
        return ParsedDoc(path.name, [Page(None, path.read_text(encoding="utf-8", errors="replace"))])
    raise ValueError(f"unsupported file type: {path.name}")


def parse_pdf(path: Path, progress=None) -> ParsedDoc:
    doc = ParsedDoc(path.name)
    reader = PdfReader(str(path))
    n = len(reader.pages)
    for i, page in enumerate(reader.pages, 1):
        text = (page.extract_text() or "").strip()
        if len(text) < MIN_TEXT_CHARS and page.images:
            ocr_text = ocr_page_images([img.data for img in page.images])
            if ocr_text:
                doc.pages.append(Page(i, ocr_text, ocr=True))
                if progress:
                    progress(i, n, True)
                continue
        doc.pages.append(Page(i, text))
        if progress:
            progress(i, n, False)
    return doc


def parse_docx(path: Path) -> ParsedDoc:
    d = DocxDocument(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        parts.extend(" ".join(cell.text for cell in row.cells) for row in table.rows)
    return ParsedDoc(path.name, [Page(None, "\n".join(parts))])


_ocr_engine = None


def ocr_page_images(images: list[bytes]) -> str:
    """OCR with RapidOCR (ONNX, CPU). Optional dependency: `uv sync --extra ocr`."""
    global _ocr_engine
    try:
        from PIL import Image
        from rapidocr import RapidOCR
    except ImportError:
        log.warning("scanned page found but OCR is not installed (uv sync --extra ocr)")
        return ""
    if _ocr_engine is None:
        # The text-direction classifier flips some blurred lines by 180 degrees and
        # turns them into garbage; business scans are upright, so it is disabled.
        _ocr_engine = RapidOCR(params={"Global.use_cls": False, "Global.log_level": "warning"})
    lines: list[str] = []
    for data in images:
        result = _ocr_engine(Image.open(io.BytesIO(data)).convert("RGB"))
        lines.extend(result.txts or ())
    return "\n".join(lines)
