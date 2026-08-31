"""
chunking.py

Divide documentos Markdown en chunks aptos para embeddings, respetando la
estructura de encabezados (##) en lugar de cortar por longitud fija a ciegas.

Por qué esto importa para el proyecto: los documentos del corpus (políticas,
contratos) tienen su información organizada por cláusula/sección. Cortar por
encabezado mantiene cada chunk semánticamente completo (ej. "Cláusula 9 — SLA"
no se parte a la mitad), lo cual afecta directamente la métrica de
Retrieval Precision@k que medimos en eval/.
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
    """Divide el texto en secciones (title, body, start_offset) usando
    encabezados Markdown (#, ##, ###) como puntos de corte."""
    matches = list(_HEADER_RE.finditer(markdown_text))

    if not matches:
        return [("(sin título)", markdown_text.strip(), 0)]

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
    """Si una sección excede max_chars, la subdivide por párrafos con
    solapamiento (overlap_chars) para no perder contexto en la frontera."""
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
            # overlap: arrastra el final del chunk anterior al siguiente
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
    """Lee un archivo Markdown y lo convierte en una lista de Chunk,
    uno por sección (o varios si la sección es muy larga)."""
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
