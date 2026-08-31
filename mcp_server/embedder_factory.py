"""
embedder_factory.py

build_index.py exposes `get_embedder(backend)` with two backends:
"hashing" (offline/deterministic, its default) and
"sentence-transformers" (real quality, needs a model download).

*** IMPORTANTE ***
Whichever backend built the persisted Chroma index at
config.CHROMA_PERSIST_PATH is the ONLY one that can query it
meaningfully — cosine distances from a "hashing" embedder and a
"sentence-transformers" embedder are not comparable, and mixing them
silently produces nonsense results (or a dimension-mismatch error from
Chroma) instead of a clear error message.

This is controlled by GROUNDEDKB_EMBEDDER_BACKEND so you set it once,
matching whatever `--backend` flag you passed to
`python -m ingestion.build_index`, instead of hardcoding it here.
Defaults to "hashing" to match build_index.py's own default.
"""

from __future__ import annotations

from ingestion.embeddings import Embedder, get_embedder

from mcp_server import config


def build_embedder() -> Embedder:
    return get_embedder(config.EMBEDDER_BACKEND)
