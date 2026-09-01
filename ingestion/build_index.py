"""
build_index.py

Orchestrates the full ingestion pipeline: reads corpus/, applies
chunking, computes embeddings, and loads everything into the vector
store.

Usage:
    python -m ingestion.build_index
    python -m ingestion.build_index --backend sentence-transformers
    python -m ingestion.build_index --corpus-dir corpus --persist-path .chroma_index
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ingestion.chunking import chunk_markdown_file
from ingestion.embeddings import get_embedder
from ingestion.vectorstore import VectorStore


def build_index(
    corpus_dir: str = "corpus",
    persist_path: str = ".chroma_index",
    backend: str = "hashing",
    reset: bool = True,
) -> dict:
    start = time.perf_counter()

    corpus_path = Path(corpus_dir)
    md_files = sorted(corpus_path.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in '{corpus_dir}'. "
            "Did you run this from the repo root?"
        )

    embedder = get_embedder(backend)
    store = VectorStore(embedder=embedder, persist_path=persist_path)

    if reset:
        store.reset()

    total_chunks = 0
    per_doc_chunks: dict[str, int] = {}

    for md_file in md_files:
        doc_id = md_file.name
        chunks = chunk_markdown_file(str(md_file), doc_id=doc_id)
        store.add_chunks(chunks)
        per_doc_chunks[doc_id] = len(chunks)
        total_chunks += len(chunks)

    elapsed = time.perf_counter() - start

    summary = {
        "backend": backend,
        "documents_indexed": len(md_files),
        "total_chunks": total_chunks,
        "chunks_per_document": per_doc_chunks,
        "elapsed_seconds": round(elapsed, 3),
        "vectors_in_store": store.count(),
    }
    return summary


def _print_summary(summary: dict) -> None:
    print("=== Ingestion summary ===")
    print(f"Embedding backend    : {summary['backend']}")
    print(f"Documents indexed    : {summary['documents_indexed']}")
    print(f"Total chunks         : {summary['total_chunks']}")
    print(f"Vectors in store     : {summary['vectors_in_store']}")
    print(f"Total time           : {summary['elapsed_seconds']}s")
    print("\nChunks per document:")
    for doc_id, n_chunks in summary["chunks_per_document"].items():
        print(f"  - {doc_id}: {n_chunks}")

    # Phase 1 measurable goal: 100% of documents indexed and ingestion
    # time under 2 minutes for the full corpus.
    if summary["elapsed_seconds"] < 120:
        print("\n✅ Phase 1 goal met: full ingestion in under 2 minutes.")
    else:
        print("\n⚠️  Ingestion exceeded 2 minutes — investigate before CI.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds the vector index for the corpus.")
    parser.add_argument("--corpus-dir", default="corpus")
    parser.add_argument("--persist-path", default=".chroma_index")
    parser.add_argument(
        "--backend",
        default="hashing",
        choices=["hashing", "sentence-transformers"],
        help="'hashing' = offline/deterministic (default). "
        "'sentence-transformers' = real quality, requires model download.",
    )
    args = parser.parse_args()

    result = build_index(
        corpus_dir=args.corpus_dir,
        persist_path=args.persist_path,
        backend=args.backend,
    )
    _print_summary(result)
