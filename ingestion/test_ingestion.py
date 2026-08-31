import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.chunking import chunk_markdown_file
from ingestion.embeddings import HashingEmbedder


def test_chunking_splits_by_header(tmp_path):
    content = (
        "# Título\n\n"
        "## Sección A\n\ncontenido de A\n\n"
        "## Sección B\n\ncontenido de B\n"
    )
    file_path = tmp_path / "doc.md"
    file_path.write_text(content, encoding="utf-8")

    chunks = chunk_markdown_file(str(file_path), doc_id="doc.md")

    titles = [c.section_title for c in chunks]
    assert "Sección A" in titles
    assert "Sección B" in titles
    assert all(c.doc_id == "doc.md" for c in chunks)


def test_chunking_all_corpus_docs_produce_chunks():
    corpus_dir = os.path.join(os.path.dirname(__file__), "..", "corpus")
    md_files = [f for f in os.listdir(corpus_dir) if f.endswith(".md")]
    assert len(md_files) == 8, "Se esperaban 8 documentos en el corpus"

    for filename in md_files:
        chunks = chunk_markdown_file(os.path.join(corpus_dir, filename), doc_id=filename)
        assert len(chunks) > 0, f"{filename} no produjo ningún chunk"


def test_hashing_embedder_is_deterministic():
    embedder = HashingEmbedder(dimension=64)
    text = "política de vacaciones"
    v1 = embedder.embed_one(text)
    v2 = embedder.embed_one(text)
    assert v1 == v2


def test_hashing_embedder_dimension():
    embedder = HashingEmbedder(dimension=64)
    vectors = embedder.embed(["texto uno", "texto dos"])
    assert all(len(v) == 64 for v in vectors)
