"""
tools/search_policy.py

Semantic search over the corporate knowledge base. Returns candidate
chunks with their source document and section — never a bare answer
without a traceable origin.
"""

from __future__ import annotations

from ingestion.vectorstore import VectorStore
from mcp_server.guardrails import get_category, is_allowed


def search_policy(
    store: VectorStore,
    query: str,
    top_k: int = 3,
    allowed_categories: list[str] | None = None,
) -> list[dict]:
    """Searches the knowledge base and returns up to `top_k` candidate
    passages, each with its source document, section title, and a
    relevance distance (lower = more relevant).

    Results outside `allowed_categories` (if provided) are filtered out
    before being returned — this is the access-scoping guardrail.
    """
    raw_results = store.query(query, top_k=top_k)

    filtered = [
        r for r in raw_results if is_allowed(r.doc_id, allowed_categories)
    ]

    return [
        {
            "chunk_id": r.chunk_id,
            "doc_id": r.doc_id,
            "category": get_category(r.doc_id),
            "section_title": r.section_title,
            "text": r.text,
            "relevance_distance": round(r.distance, 4),
        }
        for r in filtered
    ]
