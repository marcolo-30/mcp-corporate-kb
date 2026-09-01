import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.chunking import chunk_markdown_file
from ingestion.embeddings import HashingEmbedder


def test_chunking_splits_by_header(tmp_path):
    content = (
        "# Title\n\n"
        "## Section A\n\ncontent of A\n\n"
        "## Section B\n\ncontent of B\n"
    )
    file_path = tmp_path / "doc.md"
    file_path.write_text(content, encoding="utf-8")

    chunks = chunk_markdown_file(str(file_path), doc_id="doc.md")

    titles = [c.section_title for c in chunks]
    assert "Section A" in titles
    assert "Section B" in titles
    assert all(c.doc_id == "doc.md" for c in chunks)


def test_chunking_all_corpus_docs_produce_chunks():
    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "corpus")
    md_files = [f for f in os.listdir(corpus_dir) if f.endswith(".md")]
    assert len(md_files) == 8, "Expected 8 documents in the corpus"

    for filename in md_files:
        chunks = chunk_markdown_file(os.path.join(corpus_dir, filename), doc_id=filename)
        assert len(chunks) > 0, f"{filename} produced no chunks"


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dimension=64)
    text = "vacation policy"
    v1 = embedder.embed_one(text)
    v2 = embedder.embed_one(text)
    assert v1 == v2


def test_hashing_embedder_dimension():
    embedder = HashingEmbedder(dimension=64)
    vectors = embedder.embed(["first text", "second text"])
    assert all(len(v) == 64 for v in vectors)
