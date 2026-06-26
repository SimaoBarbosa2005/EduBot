"""Embedding and retrieval helpers for Ollama + ChromaDB."""

import os
from typing import Any
from functools import lru_cache
import hashlib
import re
import unicodedata

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.vector_store import VectorStore


OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_TOP_K = 5
EMBED_TIMEOUT = 30
EMBED_BATCH_SIZE = 10  # Process embeddings in batches for efficiency


class OllamaEmbedder:
    def __init__(self, base_url: str = OLLAMA_URL, model: str = EMBED_MODEL, timeout: int = EMBED_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._session = self._create_session()
        self._embedding_cache = {}

    def _create_session(self) -> requests.Session:
        """Create a persistent session with connection pooling."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=5,
            pool_maxsize=5
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text embedding."""
        return hashlib.md5(text.encode()).hexdigest()

    def embed(self, text: str) -> list[float]:
        """Get embedding for text, with caching."""
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        payload = {"model": self.model, "prompt": text}
        try:
            response = self._session.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if not embedding:
                raise RuntimeError(f"Ollama nao devolveu embedding para o modelo '{self.model}'.")
            
            # Cache the result
            self._embedding_cache[cache_key] = embedding
            return embedding
        except requests.Timeout:
            raise RuntimeError(f"Timeout ao gerar embedding após {self.timeout}s")
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar embedding: {e}")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts efficiently with batching."""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            for text in batch:
                embeddings.append(self.embed(text))
        
        return embeddings

    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()

    def __del__(self):
        """Clean up session on deletion."""
        if hasattr(self, '_session'):
            self._session.close()


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: OllamaEmbedder, top_k: int = DEFAULT_TOP_K):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        subject: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for a question."""
        limit = top_k or self.top_k
        query_embedding = self.embedder.embed(question)
        where = {"subject": subject} if subject else None
        hits = self.vector_store.query(query_embedding, top_k=max(limit * 4, limit), where=where)
        hits.extend(self._introductory_web_hits(question, subject))
        return self._rerank(question, hits)[:limit]

    def _introductory_web_hits(self, question: str, subject: str | None) -> list[dict[str, Any]]:
        query = self._normalize(question)
        broad_markers = ("fala me", "explica", "o que e", "quem foi", "resumo")
        if not any(marker in query for marker in broad_markers):
            return []

        tokens = {token for token in query.split() if len(token) >= 5}
        where = {"source_type": "web"}
        if subject:
            where = {"$and": [{"source_type": "web"}, {"subject": subject}]}

        chunks = self.vector_store.get_chunks(where=where, limit=200)
        intro_hits = []
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if int(metadata.get("chunk_index", 9999)) > 1:
                continue
            source_text = self._normalize(f"{metadata.get('source', '')} {metadata.get('title', '')}")
            if tokens and not any(token in source_text for token in tokens):
                continue
            chunk["score"] = 0.75
            intro_hits.append(chunk)
        return intro_hits

    def _rerank(self, question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query = self._normalize(question)
        tokens = {token for token in query.split() if len(token) >= 4}

        for hit in hits:
            metadata = hit.get("metadata", {})
            boost = 0.0
            source_text = self._normalize(f"{metadata.get('source', '')} {metadata.get('title', '')}")
            body_text = self._normalize(hit.get("text", "")[:500])

            if tokens and any(token in source_text for token in tokens):
                boost += 0.08
            if tokens and any(token in body_text for token in tokens):
                boost += 0.03
            if metadata.get("source_type") == "web":
                chunk_index = int(metadata.get("chunk_index", 0))
                boost += max(0.0, 0.08 - (chunk_index * 0.004))

            hit["score"] = float(hit.get("score", 0.0)) + boost

        return sorted(hits, key=lambda item: item.get("score", 0.0), reverse=True)

    @staticmethod
    def _normalize(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
        return text.casefold().strip()
