"""
tools/list_documents.py

Lists available documents in the corpus, optionally filtered by
category. Useful for a client/agent to discover what's queryable
before searching.
"""

from __future__ import annotations

from pathlib import Path

from mcp_server.guardrails import DOC_CATEGORY_MAP, get_category, is_allowed


def list_documents(
    corpus_dir: str,
    category: str | None = None,
    allowed_categories: list[str] | None = None,
) -> list[dict]:
    """Returns metadata (doc_id, category, title) for every document in
    the corpus, optionally filtered to a single `category`.

    `allowed_categories` (if provided) additionally restricts results to
    only categories the caller is scoped to see — same guardrail used by
    the other tools.
    """
    corpus_path = Path(corpus_dir)
    results: list[dict] = []

    for md_file in sorted(corpus_path.glob("*.md")):
        doc_id = md_file.name
        doc_category = get_category(doc_id)

        if category is not None and doc_category != category:
            continue
        if not is_allowed(doc_id, allowed_categories):
            continue

        first_line = md_file.read_text(encoding="utf-8").splitlines()[0]
        title = first_line.lstrip("#").strip()

        results.append(
            {
                "doc_id": doc_id,
                "category": doc_category,
                "title": title,
            }
        )

    return results


def list_categories() -> list[str]:
    """Convenience helper: every known category, for building UIs/CLIs."""
    return sorted(set(DOC_CATEGORY_MAP.values()))
