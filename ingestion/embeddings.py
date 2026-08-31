"""
embeddings.py

Interfaz de embeddings intercambiable (Strategy pattern) con dos
implementaciones:

- `SentenceTransformerEmbedder`: embeddings reales (all-MiniLM-L6-v2),
  para uso en un entorno con acceso a internet/HuggingFace Hub.
- `HashingEmbedder`: embedding determinista basado en feature hashing,
  sin dependencias externas ni descargas de modelos. Sirve como fallback
  reproducible para CI, tests y entornos sandboxed sin acceso a HF Hub.

Por qué existe el fallback: un pipeline de ingesta que solo funciona con
conexión a internet no es testeable de forma determinista en CI. El
HashingEmbedder no es competitivo en calidad semántica frente a un modelo
real, pero permite validar TODO el pipeline (chunking → embedding →
vector store → retrieval) de punta a punta sin depender de una descarga
externa — que es exactamente la limitación bajo la que se construyó este
mismo entregable.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod

_TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


class Embedder(ABC):
    """Interfaz común. Cualquier implementación debe devolver vectores
    de la misma dimensión para todos los textos de una misma instancia."""

    dimension: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashingEmbedder(Embedder):
    """Embedding determinista sin dependencias externas: tokeniza,
    aplica feature hashing a un vector de tamaño fijo, y normaliza L2.

    No captura semántica real (no sabe que "vacaciones" y "PTO" son
    similares) — es un sustituto reproducible, no un reemplazo de
    producción. Ver SentenceTransformerEmbedder para ese caso.
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
    """Wrapper sobre sentence-transformers. Requiere el paquete instalado
    y acceso de red a HuggingFace Hub la primera vez (descarga el modelo).
    Import perezoso a propósito: si el paquete no está instalado, solo
    falla cuando realmente se intenta usar esta clase, no al importar
    el módulo embeddings.py completo.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers no está instalado. "
                "Instálalo con: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


def get_embedder(backend: str = "hashing") -> Embedder:
    """Factory. backend='hashing' (default, offline) o 'sentence-transformers'
    (calidad real, requiere el paquete + descarga del modelo)."""
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Backend de embeddings desconocido: {backend}")
