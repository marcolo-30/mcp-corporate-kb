"""
eval/metrics.py

Metric functions used by run_eval.py. Split into two groups:

1. Retrieval-only metrics — computed directly from vector store results,
   no LLM call needed. Always available.
2. LLM-judged metrics (answer correctness, faithfulness) — require an
   actual generated answer and an LLM judge. Only run when
   ANTHROPIC_API_KEY is set; otherwise run_eval.py reports them as
   "skipped (no LLM backend configured)" rather than faking a number.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass


# --- Retrieval-only metrics (no LLM required) ----------------------------


def retrieval_hit_at_k(search_results: list[dict], expected_doc_id: str | None) -> bool | None:
    """True if expected_doc_id appears anywhere in the top-k search
    results. Returns None (not applicable) for trick questions that have
    no expected document."""
    if expected_doc_id is None:
        return None
    return any(r["doc_id"] == expected_doc_id for r in search_results)


def citation_correct(cite_result: dict, expected_doc_id: str | None) -> bool:
    """For factual questions: True if cite_source cited the correct
    document. For trick questions (expected_doc_id is None): True if
    cite_source correctly abstained (found=False) instead of citing
    something that doesn't exist."""
    if expected_doc_id is None:
        return cite_result.get("found") is False
    return cite_result.get("found") is True and cite_result.get("doc_id") == expected_doc_id


def is_hallucination(cite_result: dict, expected_doc_id: str | None) -> bool:
    """A hallucination here means: the system claimed to find an answer
    (found=True) on a trick question that has no real answer in the
    corpus. This is the metric that matters most for the trick
    questions in golden_set.json."""
    if expected_doc_id is not None:
        return False  # not applicable to factual questions
    return cite_result.get("found") is True


@dataclass
class Timer:
    elapsed_ms: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


# --- LLM-judged metrics (optional, require ANTHROPIC_API_KEY) -----------


def llm_backend_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def judge_answer_correctness(question: str, generated_answer: str, expected_answer: str) -> float:
    """Returns 1.0 if the generated answer is factually consistent with
    the expected answer, 0.0 otherwise. Uses an LLM-as-judge call.
    Raises if no API key is configured — callers should check
    llm_backend_available() first."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": (
                    "Question: " + question + "\n"
                    "Expected answer: " + expected_answer + "\n"
                    "Generated answer: " + generated_answer + "\n\n"
                    "Does the generated answer convey the same key facts as the "
                    "expected answer (numbers, deadlines, thresholds must match "
                    "exactly)? Reply with only '1' or '0'."
                ),
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return 1.0 if text.startswith("1") else 0.0


def judge_faithfulness(generated_answer: str, retrieved_context: str) -> float:
    """Returns 1.0 if every claim in generated_answer is supported by
    retrieved_context, 0.0 if it contains any unsupported claim."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[
            {
                "role": "user",
                "content": (
                    "Context:\n" + retrieved_context + "\n\n"
                    "Answer:\n" + generated_answer + "\n\n"
                    "Is every factual claim in the answer directly supported by "
                    "the context above (no invented numbers, dates, or clauses)? "
                    "Reply with only '1' or '0'."
                ),
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return 1.0 if text.startswith("1") else 0.0