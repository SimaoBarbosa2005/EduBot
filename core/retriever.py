"""Embedding and retrieval helpers for Ollama + ChromaDB."""

from typing import Any

import requests

from core.vector_store import VectorStore


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 5


class OllamaEmbedder:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = EMBED_MODEL, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.model, "prompt": text}
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding")
        if not embedding:
            raise RuntimeError(f"Ollama nao devolveu embedding para o modelo '{self.model}'.")
        return embedding

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: OllamaEmbedder, top_k: int = DEFAULT_TOP_K):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(self, question: str, top_k: int | None = None) -> list[dict[str, Any]]:
        query_embedding = self.embedder.embed(question)
        return self.vector_store.query(query_embedding, top_k=top_k or self.top_k)
