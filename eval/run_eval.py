"""
eval/run_eval.py

Runs the golden set against two system configurations (v0 baseline vs
v1 improved) and reports the metrics defined in eval/metrics.py.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --v1-backend sentence-transformers
    python -m eval.run_eval --golden-set golden_set.json --top-k 3

Notes:
- v0 and v1 differ only in the embeddings backend by default (hashing
  vs sentence-transformers). If/when hybrid retrieval + reranking are
  added to the pipeline, wire them in here as v1's retrieval step
  instead of just swapping the embedder.
- Answer-correctness and faithfulness require an LLM call. If
  ANTHROPIC_API_KEY isn't set, those two metrics are reported as
  "skipped" rather than a fabricated number — retrieval-based metrics
  (hit rate, citation accuracy, hallucination rate, latency) never
  require an LLM and always run.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from eval.metrics import (
    Timer,
    citation_correct,
    is_hallucination,
    judge_answer_correctness,
    judge_faithfulness,
    llm_backend_available,
    retrieval_hit_at_k,
)
from ingestion.build_index import build_index
from ingestion.embeddings import get_embedder
from ingestion.vectorstore import VectorStore
from mcp_server.tools.cite_source import cite_source
from mcp_server.tools.search_policy import search_policy


def _generate_answer(question: str, context_text: str) -> str:
    """Minimal answer synthesis for evaluation purposes, independent of
    agent_client/ (which is the interactive consumer built in a later
    phase). Only called when an LLM backend is available."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer the question using ONLY the context below. If the "
                    "context doesn't contain the answer, say so explicitly — "
                    "do not guess.\n\n"
                    f"Context:\n{context_text}\n\nQuestion: {question}"
                ),
            }
        ],
    )
    return "".join(b.text for b in response.content if b.type == "text").strip()


def _run_one_config(
    label: str,
    backend: str,
    corpus_dir: str,
    persist_path: str,
    golden_items: list[dict],
    top_k: int,
    use_llm: bool,
) -> dict:
    print(f"\n--- Building index for {label} (backend={backend}) ---")
    build_index(corpus_dir=corpus_dir, persist_path=persist_path, backend=backend)

    embedder = get_embedder(backend)
    store = VectorStore(embedder=embedder, persist_path=persist_path)

    per_item_records = []
    factual_items = [i for i in golden_items if i["type"] == "factual"]
    trick_items = [i for i in golden_items if i["type"] == "trick_no_answer"]

    for item in golden_items:
        expected_doc = item["expected_source_doc"]

        with Timer() as search_timer:
            search_results = search_policy(store, item["question"], top_k=top_k)
        with Timer() as cite_timer:
            cite_result = cite_source(store, item["question"])

        record = {
            "id": item["id"],
            "type": item["type"],
            "retrieval_hit": retrieval_hit_at_k(search_results, expected_doc),
            "citation_correct": citation_correct(cite_result, expected_doc),
            "hallucination": is_hallucination(cite_result, expected_doc),
            "search_latency_ms": round(search_timer.elapsed_ms, 2),
            "cite_latency_ms": round(cite_timer.elapsed_ms, 2),
        }

        if use_llm and search_results:
            context_text = "\n\n".join(r["text"] for r in search_results)
            generated = _generate_answer(item["question"], context_text)
            record["answer_correctness"] = judge_answer_correctness(
                item["question"], generated, item["expected_answer"]
            )
            record["faithfulness"] = judge_faithfulness(generated, context_text)

        per_item_records.append(record)

    # --- Aggregate ---
    hits = [r["retrieval_hit"] for r in per_item_records if r["retrieval_hit"] is not None]
    retrieval_hit_rate = sum(hits) / len(hits) if hits else None

    citation_accuracy = sum(r["citation_correct"] for r in per_item_records) / len(per_item_records)

    trick_records = [r for r in per_item_records if r["type"] == "trick_no_answer"]
    hallucination_rate = (
        sum(r["hallucination"] for r in trick_records) / len(trick_records)
        if trick_records
        else None
    )

    all_search_latencies = sorted(r["search_latency_ms"] for r in per_item_records)
    p95_latency = (
        all_search_latencies[int(len(all_search_latencies) * 0.95) - 1]
        if all_search_latencies
        else None
    )

    summary = {
        "label": label,
        "backend": backend,
        "n_items": len(per_item_records),
        "n_factual": len(factual_items),
        "n_trick": len(trick_items),
        "retrieval_hit_rate_at_k": round(retrieval_hit_rate, 4) if retrieval_hit_rate is not None else None,
        "citation_accuracy": round(citation_accuracy, 4),
        "hallucination_rate": round(hallucination_rate, 4) if hallucination_rate is not None else None,
        "search_latency_p50_ms": round(statistics.median(all_search_latencies), 2) if all_search_latencies else None,
        "search_latency_p95_ms": round(p95_latency, 2) if p95_latency is not None else None,
    }

    if use_llm:
        correctness_scores = [r["answer_correctness"] for r in per_item_records if "answer_correctness" in r]
        faithfulness_scores = [r["faithfulness"] for r in per_item_records if "faithfulness" in r]
        summary["answer_correctness"] = round(sum(correctness_scores) / len(correctness_scores), 4) if correctness_scores else None
        summary["faithfulness"] = round(sum(faithfulness_scores) / len(faithfulness_scores), 4) if faithfulness_scores else None
    else:
        summary["answer_correctness"] = "skipped (no LLM backend configured)"
        summary["faithfulness"] = "skipped (no LLM backend configured)"

    return {"summary": summary, "per_item": per_item_records}


def _print_comparison_table(v0: dict, v1: dict) -> None:
    def fmt(value):
        if value is None:
            return "n/a"
        if isinstance(value, str):
            return value
        if isinstance(value, float) and value <= 1.0:
            return f"{value:.1%}"
        return str(value)

    rows = [
        ("Retrieval Hit Rate@k", "retrieval_hit_rate_at_k"),
        ("Citation Accuracy", "citation_accuracy"),
        ("Hallucination Rate (trick Qs)", "hallucination_rate"),
        ("Answer Correctness", "answer_correctness"),
        ("Faithfulness", "faithfulness"),
        ("Search Latency p50 (ms)", "search_latency_p50_ms"),
        ("Search Latency p95 (ms)", "search_latency_p95_ms"),
    ]

    print("\n=== v0 vs v1 comparison ===")
    print(f"{'Metric':<32}{'v0 (' + v0['backend'] + ')':<22}{'v1 (' + v1['backend'] + ')':<22}")
    for label, key in rows:
        print(f"{label:<32}{fmt(v0[key]):<22}{fmt(v1[key]):<22}")


def main():
    parser = argparse.ArgumentParser(description="Runs v0 vs v1 against the golden set.")
    parser.add_argument("--golden-set", default="golden_set.json")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--v0-backend", default="hashing")
    parser.add_argument("--v1-backend", default="sentence-transformers")
    parser.add_argument("--v0-index-path", default=".chroma_v0")
    parser.add_argument("--v1-index-path", default=".chroma_v1")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument(
        "--skip-v1",
        action="store_true",
        help="Only run v0 (useful when the v1 backend isn't installed/available yet).",
    )
    args = parser.parse_args()

    with open(args.golden_set, encoding="utf-8") as f:
        golden_data = json.load(f)
    golden_items = golden_data["items"]

    use_llm = llm_backend_available()
    if not use_llm:
        print(
            "\n⚠️  ANTHROPIC_API_KEY not set — answer_correctness and "
            "faithfulness will be reported as 'skipped'. Retrieval-based "
            "metrics (hit rate, citation accuracy, hallucination rate, "
            "latency) do not require an LLM and will still run."
        )

    v0_result = _run_one_config(
        "v0", args.v0_backend, args.corpus_dir, args.v0_index_path, golden_items, args.top_k, use_llm
    )

    if args.skip_v1:
        print("\n--skip-v1 passed — skipping v1 run.")
        results = {"v0": v0_result, "v1": None}
    else:
        v1_result = _run_one_config(
            "v1", args.v1_backend, args.corpus_dir, args.v1_index_path, golden_items, args.top_k, use_llm
        )
        _print_comparison_table(v0_result["summary"], v1_result["summary"])
        results = {"v0": v0_result, "v1": v1_result}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{int(time.time())}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()