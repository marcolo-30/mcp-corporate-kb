"""
chunking.py

Splits Markdown documents into embedding-ready chunks, respecting the
heading structure (##) instead of cutting blindly at a fixed length.

Why this matters for the project: the corpus documents (policies,
contracts) organize their information by clause/section. Splitting on
headings keeps each chunk semantically whole (e.g. "Clause 9 — SLA"
never gets cut in half), which directly affects the Retrieval
Precision@k metric measured in eval/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_title: str
    text: str
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


_HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


def _split_by_headers(markdown_text: str) -> list[tuple[str, str, int]]:
    """Splits the text into sections (title, body, start_offset) using
    Markdown headings (#, ##, ###) as cut points."""
    matches = list(_HEADER_RE.finditer(markdown_text))

    if not matches:
        return [("(untitled)", markdown_text.strip(), 0)]

    sections: list[tuple[str, str, int]] = []
    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        body = markdown_text[start:end].strip()
        if body:
            sections.append((title, body, match.start()))
    return sections


def _split_long_section(
    title: str,
    body: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """If a section exceeds max_chars, splits it by paragraphs with
    overlap (overlap_chars) so context isn't lost at the boundary."""
    if len(body) <= max_chars:
        return [body]

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    pieces: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                pieces.append(current)
            # overlap: carries the tail of the previous chunk into the next
            overlap_text = current[-overlap_chars:] if current else ""
            current = f"{overlap_text}\n\n{para}".strip()

    if current:
        pieces.append(current)

    return pieces or [body]


def chunk_markdown_file(
    file_path: str,
    doc_id: str,
    max_chars: int = 800,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Reads a Markdown file and converts it into a list of Chunk objects,
    one per section (or several if a section is too long)."""
    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    sections = _split_by_headers(text)
    chunks: list[Chunk] = []
    running_offset = 0

    for section_index, (title, body, header_offset) in enumerate(sections):
        pieces = _split_long_section(title, body, max_chars, overlap_chars)
        for piece_index, piece in enumerate(pieces):
            chunk_id = f"{doc_id}::s{section_index}::p{piece_index}"
            char_start = running_offset
            char_end = char_start + len(piece)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    section_title=title,
                    text=piece,
                    char_start=char_start,
                    char_end=char_end,
                    metadata={"section_index": section_index, "piece_index": piece_index},
                )
            )
            running_offset = char_end

    return chunks
