"""
embeddings.py

Interchangeable embeddings interface (Strategy pattern) with two
implementations:

- `SentenceTransformerEmbedder`: real embeddings (all-MiniLM-L6-v2),
  for use in an environment with internet/HuggingFace Hub access.
- `HashingEmbedder`: deterministic embedding based on feature hashing,
  no external dependencies or model downloads. Serves as a reproducible
  fallback for CI, tests, and sandboxed environments without HF Hub
  access.

Why the fallback exists: an ingestion pipeline that only works with an
internet connection isn't deterministically testable in CI. The
HashingEmbedder isn't competitive in semantic quality against a real
model, but it lets the ENTIRE pipeline (chunking → embedding →
vector store → retrieval) be validated end to end without depending on
an external download — which is exactly the constraint this very
deliverable was built under.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class Embedder(ABC):
    """Common interface. Any implementation must return vectors of the
    same dimension for all texts within a given instance."""

    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Deterministic embedding with no external dependencies: tokenizes,
    applies feature hashing into a fixed-size vector, and L2-normalizes.

    Doesn't capture real semantics (it doesn't know that "vacation" and
    "PTO" are related) — it's a reproducible stand-in, not a production
    replacement. See SentenceTransformerEmbedder for that case.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            tokens = _TOKEN_RE.findall(text.lower())
            for token in tokens:
                idx = self._hash_token(token)
                vector[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


class SentenceTransformerEmbedder(Embedder):
    """Wrapper around sentence-transformers. Requires the package to be
    installed and network access to the HuggingFace Hub the first time
    (to download the model). Import is lazy on purpose: if the package
    isn't installed, it only fails when this class is actually used,
    not when the embeddings.py module is imported.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(backend: str = "hashing") -> Embedder:
    """Factory. backend='hashing' (default, offline) or
    'sentence-transformers' (real quality, requires the package + model
    download)."""
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embeddings backend: {backend}")
