"""Builds a real RAG index from loaded documents."""

import hashlib
from collections.abc import Callable
from typing import Any

from core.document_loader import DocumentLoader, DocumentSection
from core.retriever import OllamaEmbedder
from core.vector_store import VectorStore


CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100


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
        chunks = self._chunk_sections(sections)

        if not chunks:
            self.vector_store.reset()
            return 0

        embeddings = []
        for i, chunk in enumerate(chunks, 1):
            if i == 1 or i % 10 == 0 or i == len(chunks):
                self._progress(progress, f"A gerar embeddings: {i}/{len(chunks)} chunks...")
            embeddings.append(self.embedder.embed(chunk["text"]))

        self._progress(progress, "A guardar indice vetorial...")
        self.vector_store.reset()
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
                        "chunk_index": chunk_index,
                    }
                    if section.page is not None:
                        metadata["page"] = section.page
                    if section.slide is not None:
                        metadata["slide"] = section.slide

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
        raw = f"{section.source}:{location}:{chunk_index}:{text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _progress(progress: Callable[[str], None] | None, message: str) -> None:
        if progress:
            progress(message)
