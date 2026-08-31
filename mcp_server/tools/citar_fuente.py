"""
tools/citar_fuente.py

Dado un chunk_id (el que devuelve buscar_politica), recupera el
fragmento exacto tal como está indexado, sin pasar por una nueva
búsqueda semántica. Esto es lo que permite verificar una cita después
del hecho: "¿este chunk_id realmente dice lo que el agente afirmó que
decía?" — clave para el chequeo de faithfulness en eval/.

*** SUPUESTO EXPLÍCITO ***
VectorStore (tal como me lo pasaste) no expone un método público de
"traer por id", solo `.query()` (búsqueda semántica) y `.add_chunks()`.
Para no inventar un cambio de comportamiento, este tool accede a
`vectorstore._collection` (el cliente de Chroma interno) directamente.
Es un acceso a un atributo "privado" por convención de nombre, así que
lo ideal a mediano plazo es que agregues un método público
`VectorStore.get_by_id(chunk_id)` — lo dejo marcado para que sea tu
decisión de diseño, no una que tome yo por debajo sin que la veas.
"""

from __future__ import annotations

from ingestion.vectorstore import VectorStore

from mcp_server.doc_registry import lookup
from mcp_server.guardrails import check_category_access, log_tool_call


def citar_fuente(vectorstore: VectorStore, chunk_id: str, role: str | None = None) -> dict:
    """
    Recupera el fragmento exacto indexado bajo `chunk_id`.

    Returns:
        dict con "encontrado": bool y, si es True, "cita" con
        doc_id, titulo_documento, section_title, texto, char_start, char_end.
    """
    # TODO(usuario): reemplazar por vectorstore.get_by_id(chunk_id) si lo agregas.
    raw = vectorstore._collection.get(ids=[chunk_id])  # noqa: SLF001

    if not raw["ids"]:
        log_tool_call(
            "citar_fuente", {"chunk_id": chunk_id}, role, result_summary="chunk_id no encontrado"
        )
        return {"encontrado": False, "cita": None}

    metadata = raw["metadatas"][0]
    meta = lookup(metadata["doc_id"])

    try:
        check_category_access(meta.category, role)
    except Exception as e:  # PermissionDenied
        log_tool_call(
            "citar_fuente",
            {"chunk_id": chunk_id},
            role,
            result_summary=f"denegado: {e}",
            authorized=False,
        )
        return {"encontrado": False, "cita": None, "motivo": str(e)}

    cita = {
        "doc_id": metadata["doc_id"],
        "titulo_documento": meta.title,
        "section_title": metadata["section_title"],
        "texto": raw["documents"][0],
        "char_start": metadata["char_start"],
        "char_end": metadata["char_end"],
    }
    log_tool_call(
        "citar_fuente", {"chunk_id": chunk_id}, role, result_summary=f"doc={meta.doc_id}"
    )
    return {"encontrado": True, "cita": cita}
