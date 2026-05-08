"""PDF text + layout extraction using PyMuPDF (fitz).

Returns structured blocks with font size, position, and bold flags so
downstream feature extractors can use layout cues (e.g. title detection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import fitz  # PyMuPDF


@dataclass
class TextSpan:
    text: str
    size: float
    font: str
    flags: int
    bbox: tuple
    page: int

    @property
    def is_bold(self) -> bool:
        return bool(self.flags & 16) or "Bold" in self.font or "bold" in self.font


@dataclass
class TextLine:
    text: str
    size: float
    bbox: tuple
    page: int
    is_bold: bool
    spans: List[TextSpan] = field(default_factory=list)


@dataclass
class PdfDocument:
    path: str
    pages_text: List[str]
    lines: List[TextLine]
    raw_text: str

    @property
    def first_page_text(self) -> str:
        return self.pages_text[0] if self.pages_text else ""

    def text_of_first_n_pages(self, n: int = 2) -> str:
        return "\n".join(self.pages_text[:n])


def read_pdf(path: str, max_pages: int | None = None) -> PdfDocument:
    """Read a PDF and return text + structured lines."""
    doc = fitz.open(path)
    pages_text: List[str] = []
    lines: List[TextLine] = []

    page_count = len(doc) if max_pages is None else min(len(doc), max_pages)

    for page_idx in range(page_count):
        page = doc[page_idx]
        page_text = page.get_text("text") or ""
        pages_text.append(page_text)

        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 = text
                continue
            for line in block.get("lines", []):
                spans_data = line.get("spans", [])
                if not spans_data:
                    continue
                spans: List[TextSpan] = []
                line_text_parts = []
                sizes = []
                for span in spans_data:
                    txt = span.get("text", "")
                    if not txt:
                        continue
                    s = TextSpan(
                        text=txt,
                        size=float(span.get("size", 0.0)),
                        font=span.get("font", ""),
                        flags=int(span.get("flags", 0)),
                        bbox=tuple(span.get("bbox", (0, 0, 0, 0))),
                        page=page_idx,
                    )
                    spans.append(s)
                    line_text_parts.append(txt)
                    sizes.append(s.size)
                if not spans:
                    continue
                line_text = "".join(line_text_parts).strip()
                if not line_text:
                    continue
                avg_size = sum(sizes) / len(sizes)
                bold = any(s.is_bold for s in spans)
                lines.append(
                    TextLine(
                        text=line_text,
                        size=avg_size,
                        bbox=tuple(line.get("bbox", (0, 0, 0, 0))),
                        page=page_idx,
                        is_bold=bold,
                        spans=spans,
                    )
                )

    doc.close()
    raw_text = "\n".join(pages_text)
    return PdfDocument(path=path, pages_text=pages_text, lines=lines, raw_text=raw_text)
