"""
tools/buscar_politica.py

Tool principal de búsqueda. La regla de negoción del README es:
"la respuesta debe resolver a un documento+fragmento exacto, o el
sistema dice que no sabe" — así que este tool nunca redacta una
respuesta en lenguaje natural por sí mismo; devuelve los fragmentos
citables y deja que el agente consumidor (agent_client/) construya
la respuesta final SOBRE esos fragmentos. Esto es lo que hace que
"faithfulness" sea medible en eval/: el tool es determinista y
auditable, la generación de lenguaje natural queda en otra capa.
"""

from __future__ import annotations

from ingestion.vectorstore import VectorStore

from mcp_server import config
from mcp_server.doc_registry import lookup
from mcp_server.guardrails import (
    PermissionDenied,
    check_category_access,
    filter_allowed_categories,
    log_tool_call,
)


def buscar_politica(
    vectorstore: VectorStore,
    query: str,
    top_k: int = 3,
    categoria: str | None = None,
    role: str | None = None,
) -> dict:
    """
    Busca fragmentos relevantes en la base de conocimiento.

    Args:
        query: pregunta en lenguaje natural del usuario.
        top_k: cuántos fragmentos candidatos traer del índice antes de filtrar.
        categoria: si se especifica, solo se consideran documentos de esa
            categoría (ver doc_registry.CATEGORIES). Si es None, se buscan
            todas las categorías permitidas para `role`.
        role: rol del solicitante, para permission scoping (ver guardrails.py).

    Returns:
        dict con:
          - "encontrado": bool
          - "fragmentos": lista de citas (doc_id, titulo, section_title,
            texto, chunk_id, distancia) ordenadas por relevancia
          - "motivo_sin_resultado": string explicando por qué no se
            encontró nada citable, si "encontrado" es False
    """
    allowed_categories = filter_allowed_categories(role)

    if categoria is not None:
        try:
            check_category_access(categoria, role)
        except PermissionDenied as e:
            log_tool_call(
                "buscar_politica",
                {"query": query, "categoria": categoria, "top_k": top_k},
                role,
                result_summary=f"denegado: {e}",
                authorized=False,
            )
            return {
                "encontrado": False,
                "fragmentos": [],
                "motivo_sin_resultado": str(e),
            }

    # Traemos más candidatos de los pedidos porque vamos a filtrar por
    # categoría/permiso después — así top_k sigue siendo "resultados
    # útiles", no "resultados antes de filtrar".
    raw_results = vectorstore.query(query, top_k=max(top_k * 3, top_k))

    fragmentos = []
    for r in raw_results:
        meta = lookup(r.doc_id)
        if categoria is not None and meta.category != categoria:
            continue
        if meta.category not in allowed_categories:
            continue
        if r.distance > config.MAX_ACCEPTABLE_DISTANCE:
            # Demasiado lejos semánticamente para ser una cita confiable:
            # se descarta en vez de forzarlo, aunque sea el "mejor" resultado.
            continue
        fragmentos.append(
            {
                "doc_id": r.doc_id,
                "titulo_documento": meta.title,
                "section_title": r.section_title,
                "texto": r.text,
                "chunk_id": r.chunk_id,
                "distancia": r.distance,
            }
        )
        if len(fragmentos) >= top_k:
            break

    if not fragmentos:
        motivo = (
            "No se encontró ningún fragmento suficientemente relevante "
            f"(umbral de distancia coseno = {config.MAX_ACCEPTABLE_DISTANCE}). "
            "Esto puede significar que la pregunta no está cubierta por el "
            "corpus, o que está fuera de las categorías permitidas para tu rol."
        )
        log_tool_call(
            "buscar_politica",
            {"query": query, "categoria": categoria, "top_k": top_k},
            role,
            result_summary="sin resultados citables",
        )
        return {"encontrado": False, "fragmentos": [], "motivo_sin_resultado": motivo}

    log_tool_call(
        "buscar_politica",
        {"query": query, "categoria": categoria, "top_k": top_k},
        role,
        result_summary=f"{len(fragmentos)} fragmento(s) de "
        f"{sorted({f['doc_id'] for f in fragmentos})}",
    )
    return {"encontrado": True, "fragmentos": fragmentos, "motivo_sin_resultado": None}
