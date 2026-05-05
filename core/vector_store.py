"""Small ChromaDB wrapper used by the RAG pipeline."""

from pathlib import Path
from typing import Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = ".rag_index", collection_name: str = "edubot_chunks"):
        base = Path(__file__).resolve().parent.parent
        self.persist_dir = base / persist_dir
        self.persist_dir.mkdir(exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self, where: dict[str, Any] | None = None) -> int:
        if where is None:
            return self.collection.count()

        result = self.collection.get(where=where, include=[])
        return len(result.get("ids", []))

    def delete_where(self, where: dict[str, Any]) -> None:
        self.collection.delete(where=where)

    def metadata_counts(self, key: str) -> dict[str, int]:
        result = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in result.get("metadatas", []):
            if not metadata:
                continue
            value = metadata.get(key)
            if value is None:
                continue
            counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

    def get_chunks(self, where: dict[str, Any] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        result = self.collection.get(where=where, limit=limit, include=["documents", "metadatas"])
        ids = result.get("ids", [])
        docs = result.get("documents", [])
        metadatas = result.get("metadatas", [])

        chunks = []
        for chunk_id, text, metadata in zip(ids, docs, metadatas):
            chunks.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata or {},
                    "score": 0.0,
                }
            )
        return chunks

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        self.collection.upsert(
            ids=[chunk["id"] for chunk in chunks],
            documents=[chunk["text"] for chunk in chunks],
            metadatas=[chunk["metadata"] for chunk in chunks],
            embeddings=embeddings,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        hits = []
        for text, metadata, distance in zip(docs, metadatas, distances):
            hits.append(
                {
                    "text": text,
                    "metadata": metadata or {},
                    "score": 1 - float(distance),
                }
            )
        return hits
