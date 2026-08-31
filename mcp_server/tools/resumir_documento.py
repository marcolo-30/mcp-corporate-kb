"""
tools/resumir_documento.py

*** SUPUESTO EXPLÍCITO ***
VectorStore solo almacena chunks, no el documento completo, y
reconstruirlo concatenando chunks es frágil si tu chunking.py usa
overlap entre fragmentos (duplicaría texto). Así que este tool lee el
archivo original directamente de config.CORPUS_DIR/{doc_id}.md — igual
que hace (asumo) ingestion/build_index.py antes de trocearlo.

Resumen extractivo por defecto (encabezados + primer párrafo de cada
sección) para que el tool funcione sin depender de una API key de LLM.
Si quieres un resumen generativo real, el punto de extensión está
marcado abajo — no lo activo por defecto porque cambiaría el tool de
"determinista y gratis" a "depende de un proveedor externo y cuesta
tokens", y esa es una decisión de producto, no algo que deba decidir
yo en silencio.
"""

from __future__ import annotations

import re

from mcp_server import config
from mcp_server.doc_registry import lookup
from mcp_server.guardrails import PermissionDenied, check_category_access, log_tool_call


def _extractive_summary(text: str, max_sections: int = 6) -> list[dict]:
    """
    Resumen extractivo simple basado en encabezados Markdown (#, ##, ###)
    + el primer párrafo no vacío bajo cada uno. Sin dependencias externas.
    """
    lines = text.splitlines()
    sections: list[dict] = []
    current_heading = None
    current_para_lines: list[str] = []

    def flush():
        if current_heading is not None:
            first_para = " ".join(current_para_lines).strip()
            sections.append(
                {
                    "encabezado": current_heading,
                    "primer_parrafo": first_para[:400],
                }
            )

    for line in lines:
        heading_match = re.match(r"^(#{1,3})\s+(.*)", line.strip())
        if heading_match:
            flush()
            current_heading = heading_match.group(2).strip()
            current_para_lines = []
        elif line.strip():
            if not current_para_lines:
                current_para_lines.append(line.strip())
        if len(sections) >= max_sections:
            break
    flush()
    return sections[:max_sections]


def resumir_documento(doc_id: str, role: str | None = None) -> dict:
    """
    Devuelve un resumen extractivo (encabezados + primer párrafo por
    sección) de un documento del corpus, identificado por doc_id.

    EXTENSIÓN OPCIONAL: para un resumen generativo real, reemplaza el
    cuerpo de esta función por una llamada a tu LLM de preferencia con
    el texto completo (`raw_text`) como contexto, y post-procesa el
    resultado para seguir devolviendo la misma forma de dict.
    """
    meta = lookup(doc_id)
    try:
        check_category_access(meta.category, role)
    except PermissionDenied as e:
        log_tool_call(
            "resumir_documento",
            {"doc_id": doc_id},
            role,
            result_summary=f"denegado: {e}",
            authorized=False,
        )
        return {"encontrado": False, "resumen": None, "motivo": str(e)}

    # doc_id ya incluye la extensión ".md" (build_index.py usa
    # `doc_id = md_file.name`, no el stem) — no volver a añadirla aquí.
    doc_path = config.CORPUS_DIR / doc_id
    if not doc_path.exists():
        log_tool_call(
            "resumir_documento",
            {"doc_id": doc_id},
            role,
            result_summary=f"archivo no encontrado: {doc_path}",
        )
        return {
            "encontrado": False,
            "resumen": None,
            "motivo": f"No se encontró el archivo {doc_path} en el corpus.",
        }

    raw_text = doc_path.read_text(encoding="utf-8")
    secciones = _extractive_summary(raw_text)

    log_tool_call(
        "resumir_documento",
        {"doc_id": doc_id},
        role,
        result_summary=f"{len(secciones)} secciones resumidas",
    )
    return {
        "encontrado": True,
        "resumen": {
            "doc_id": doc_id,
            "titulo_documento": meta.title,
            "categoria": meta.category,
            "secciones": secciones,
        },
    }
