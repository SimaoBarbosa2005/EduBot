"""Builds a real RAG index from loaded documents."""

import hashlib
from collections.abc import Callable
from typing import Any

from core.document_loader import DocumentLoader, DocumentSection
from core.retriever import OllamaEmbedder
from core.vector_store import VectorStore


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
EMBEDDING_BATCH_SIZE = 10  # Process embeddings in batches for faster indexing


class RAGIndexer:
    def __init__(
        self,
        loader: DocumentLoader,
        vector_store: VectorStore,
        embedder: OllamaEmbedder,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.loader = loader
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def rebuild(self, progress: Callable[[str], None] | None = None) -> int:
        self._progress(progress, "A ler documentos...")
        sections = self.loader.load_all_sections()
        self._progress(progress, f"{len(sections)} seccao(oes) extraida(s). A criar chunks...")
        return self.index_sections(sections, reset_where={"source_type": "document"}, progress=progress)

    def index_sections(
        self,
        sections: list[DocumentSection],
        reset_where: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> int:
        chunks = self._chunk_sections(sections)

        if not chunks:
            if reset_where:
                self.vector_store.delete_where(reset_where)
            return 0

        if extra_metadata:
            for chunk in chunks:
                chunk["metadata"].update(extra_metadata)

        # Batch process embeddings for better performance
        embeddings = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
            batch_end = min(i + EMBEDDING_BATCH_SIZE, len(chunks))
            self._progress(progress, f"A gerar embeddings: {batch_end}/{len(chunks)} chunks...")
            
            batch_embeddings = self.embedder.embed_many([chunk["text"] for chunk in batch])
            embeddings.extend(batch_embeddings)

        self._progress(progress, "A guardar indice vetorial...")
        if reset_where:
            self.vector_store.delete_where(reset_where)
        self.vector_store.add_chunks(chunks, embeddings)
        return len(chunks)

    def _chunk_sections(self, sections: list[DocumentSection]) -> list[dict[str, Any]]:
        chunks = []
        for section in sections:
            text = DocumentLoader.clean_text(section.text)
            if not text:
                continue

            start = 0
            chunk_index = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    metadata = {
                        "source": section.source,
                        "source_type": "document",
                        "chunk_index": chunk_index,
                    }
                    if section.page is not None:
                        metadata["page"] = section.page
                    if section.slide is not None:
                        metadata["slide"] = section.slide
                    metadata.update(section.metadata)

                    chunks.append(
                        {
                            "id": self._chunk_id(section, chunk_index, chunk_text),
                            "text": chunk_text,
                            "metadata": metadata,
                        }
                    )

                chunk_index += 1
                if end == len(text):
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks

    @staticmethod
    def _chunk_id(section: DocumentSection, chunk_index: int, text: str) -> str:
        location = section.page if section.page is not None else section.slide
        url = section.metadata.get("url", "")
        raw = f"{section.source}:{url}:{location}:{chunk_index}:{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _progress(progress: Callable[[str], None] | None, message: str) -> None:
        if progress:
            progress(message)
