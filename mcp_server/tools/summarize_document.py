"""
tools/summarize_document.py

Summarizes a full source document by its doc_id.

Pluggable summarization backend, same pattern as ingestion/embeddings.py:
- If ANTHROPIC_API_KEY is set, calls the Anthropic API for a real
  abstractive summary.
- Otherwise, falls back to a deterministic extractive summary (section
  headings + first sentence of each section's body) — no external
  dependency, no network call, fully offline-safe.

This mirrors the same design decision made in ingestion/embeddings.py:
the tool must be fully testable and runnable without external services,
while still supporting a real LLM backend in production.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _extractive_summary(full_text: str, max_sentences_per_section: int = 1) -> str:
    """Offline fallback: keeps section headings, plus the first N
    sentences of each section's body. No API key or network access
    required."""
    lines = full_text.splitlines()
    summary_lines: list[str] = []
    current_section_body: list[str] = []

    def flush_section():
        if current_section_body:
            body_text = " ".join(current_section_body)
            sentences = _SENTENCE_SPLIT_RE.split(body_text.strip())
            kept = " ".join(sentences[:max_sentences_per_section]).strip()
            if kept:
                summary_lines.append(kept)

    for line in lines:
        if line.startswith("#"):
            flush_section()
            current_section_body = []
            summary_lines.append(line)
        else:
            current_section_body.append(line.strip())

    flush_section()
    return "\n".join(summary_lines).strip()


def _llm_summary(full_text: str, doc_id: str) -> str:
    """Real abstractive summary via the Anthropic API. Lazy import so
    the module doesn't require the `anthropic` package unless this path
    is actually used."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Summarize the following corporate document ({doc_id}) "
                    "in 4-6 concise bullet points, preserving any specific "
                    "numbers, deadlines, or thresholds exactly as written. "
                    "Do not add information that is not in the text.\n\n"
                    f"{full_text}"
                ),
            }
        ],
    )
    text_blocks = [block.text for block in response.content if block.type == "text"]
    return "\n".join(text_blocks).strip()


def summarize_document(
    corpus_dir: str,
    doc_id: str,
    backend: str = "auto",
) -> dict:
    """Summarizes the document identified by `doc_id`.

    backend:
        "auto"        - use the LLM if ANTHROPIC_API_KEY is set, else extractive
        "extractive"  - force the offline fallback
        "llm"         - force the LLM call (raises if no API key is configured)
    """
    file_path = Path(corpus_dir) / doc_id
    if not file_path.exists():
        return {"found": False, "reason": f"Document '{doc_id}' not found in corpus."}

    full_text = file_path.read_text(encoding="utf-8")

    use_llm = backend == "llm" or (backend == "auto" and bool(os.environ.get("ANTHROPIC_API_KEY")))

    if use_llm:
        summary = _llm_summary(full_text, doc_id)
        method = "llm"
    else:
        summary = _extractive_summary(full_text)
        method = "extractive"

    return {
        "found": True,
        "doc_id": doc_id,
        "summary": summary,
        "method": method,
    }
