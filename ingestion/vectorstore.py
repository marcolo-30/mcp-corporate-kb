"""
vectorstore.py

Thin wrapper around ChromaDB. Isolates the rest of the pipeline (and the
mcp_server later on) from the details of the concrete library — if you
switch to pgvector or Qdrant tomorrow, only this file needs rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection

from ingestion.chunking import Chunk
from ingestion.embeddings import Embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    section_title: str
    text: str
    distance: float


class VectorStore:
    def __init__(
        self,
        embedder: Embedder,
        persist_path: str = ".chroma_index",
        collection_name: str = "corporate_kb",
    ):
        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=persist_path)
        # embedding_function=None: we pass in vectors we've already
        # computed with our own Embedder, instead of letting Chroma
        # download its default model (avoids an uncontrolled network
        # dependency).
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "section_title": c.section_title,
                    "char_start": c.char_start,
                    "char_end": c.char_end,
                }
                for c in chunks
            ],
        )

    def count(self) -> int:
        return self._collection.count()

    def query(self, query_text: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed([query_text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved: list[RetrievedChunk] = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(ids)):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=ids[i],
                    doc_id=metadatas[i]["doc_id"],
                    section_title=metadatas[i]["section_title"],
                    text=documents[i],
                    distance=distances[i],
                )
            )
        return retrieved
