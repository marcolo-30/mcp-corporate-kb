"""
tools/listar_documentos.py

Lista los documentos disponibles, opcionalmente filtrados por
categoría, y cuenta cuántos chunks tiene cada uno en el índice (útil
para detectar documentos "a medias" indexados). Cruza dos fuentes:
- doc_registry: catálogo de doc_id -> categoría/título conocido
- vectorstore: qué doc_id están REALMENTE indexados ahora mismo

Si un doc_id está en el registro pero no en el índice (o viceversa),
se reporta explícitamente en vez de esconder la discrepancia — es una
señal útil de que build_index.py no corrió sobre el corpus completo.
"""

from __future__ import annotations

from ingestion.vectorstore import VectorStore

from mcp_server.doc_registry import all_known_doc_ids, lookup
from mcp_server.guardrails import filter_allowed_categories, log_tool_call


def listar_documentos(
    vectorstore: VectorStore, categoria: str | None = None, role: str | None = None
) -> dict:
    """
    Returns:
        dict con "documentos": lista de {doc_id, titulo, categoria,
        chunks_indexados} y "advertencias": discrepancias entre el
        catálogo y el índice real, si las hay.
    """
    allowed = filter_allowed_categories(role)

    # TODO(usuario): si VectorStore expone un método público de listar
    # doc_ids indexados, úsalo aquí en vez de `_collection.get()`.
    raw = vectorstore._collection.get()  # noqa: SLF001
    indexed_doc_ids = sorted({m["doc_id"] for m in raw["metadatas"]}) if raw["metadatas"] else []
    chunk_counts: dict[str, int] = {}
    for m in raw["metadatas"] or []:
        chunk_counts[m["doc_id"]] = chunk_counts.get(m["doc_id"], 0) + 1

    catalog_doc_ids = set(all_known_doc_ids())
    indexed_set = set(indexed_doc_ids)

    documentos = []
    for doc_id in sorted(catalog_doc_ids | indexed_set):
        meta = lookup(doc_id)
        if categoria is not None and meta.category != categoria:
            continue
        if meta.category not in allowed:
            continue
        documentos.append(
            {
                "doc_id": doc_id,
                "titulo": meta.title,
                "categoria": meta.category,
                "chunks_indexados": chunk_counts.get(doc_id, 0),
                "indexado": doc_id in indexed_set,
            }
        )

    advertencias = []
    not_indexed = (catalog_doc_ids & allowed_ids_placeholder(allowed)) - indexed_set
    unknown_in_index = indexed_set - catalog_doc_ids
    if not_indexed:
        advertencias.append(
            f"En el catálogo pero sin indexar todavía: {sorted(not_indexed)}"
        )
    if unknown_in_index:
        advertencias.append(
            f"Indexados pero sin entrada en doc_registry (categoría desconocida): "
            f"{sorted(unknown_in_index)}"
        )

    log_tool_call(
        "listar_documentos",
        {"categoria": categoria},
        role,
        result_summary=f"{len(documentos)} documento(s) listado(s)",
    )
    return {"documentos": documentos, "advertencias": advertencias}


def allowed_ids_placeholder(allowed_categories: set[str]) -> set[str]:
    """Filtra el catálogo completo a los doc_id cuya categoría está permitida."""
    return {doc_id for doc_id in all_known_doc_ids() if lookup(doc_id).category in allowed_categories}
