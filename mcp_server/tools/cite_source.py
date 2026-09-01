"""
tools/cite_source.py

The anti-hallucination tool: given a question, returns ONE citation
(document + exact fragment) if — and only if — retrieval confidence
clears a threshold. Below that threshold, it explicitly returns
"not found" instead of guessing. This is the tool responsible for
correctly handling the two trick questions in golden_set_esp.json.
"""

from __future__ import annotations

from ingestion.vectorstore import VectorStore
from mcp_server.guardrails import get_category, is_allowed

# Cosine distance threshold above which a match is considered "not
# confident enough" to cite as a real answer. Tune this against the
# golden set's trick questions during evaluation (Phase 5) — this
# starting value is a reasonable default, not a proven number yet.
DEFAULT_CONFIDENCE_THRESHOLD = 0.35


def cite_source(
    store: VectorStore,
    query: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    allowed_categories: list[str] | None = None,
) -> dict:
    """Returns the single best-matching citation for `query`, or an
    explicit "not found" result if nothing clears the confidence bar.

    Return shape (found):
        {"found": True, "doc_id": ..., "category": ..., "section_title": ...,
         "fragment": ..., "confidence_distance": ...}

    Return shape (not found):
        {"found": False, "reason": "..."}
    """
    results = store.query(query, top_k=5)
    allowed_results = [r for r in results if is_allowed(r.doc_id, allowed_categories)]

    if not allowed_results:
        return {
            "found": False,
            "reason": "No matching document found in the accessible knowledge base.",
        }

    best = allowed_results[0]

    if best.distance > confidence_threshold:
        return {
            "found": False,
            "reason": (
                "No passage in the knowledge base is confident enough to "
                "answer this question. Best candidate distance "
                f"({best.distance:.4f}) exceeds the threshold "
                f"({confidence_threshold})."
            ),
        }

    return {
        "found": True,
        "doc_id": best.doc_id,
        "category": get_category(best.doc_id),
        "section_title": best.section_title,
        "fragment": best.text,
        "confidence_distance": round(best.distance, 4),
    }
