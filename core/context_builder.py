"""Prompt construction for retrieved RAG context."""

from typing import Any


class ContextBuilder:
    def __init__(self):
        self._documents: dict[str, str] = {}
        self._retrieved_chunks: list[dict[str, Any]] = []

    def update_documents(self, docs: dict[str, str]) -> None:
        # Kept for compatibility with older CLI/web code. RAG uses chunks.
        self._documents = docs

    def set_retrieved_chunks(self, chunks: list[dict[str, Any]]) -> None:
        self._retrieved_chunks = chunks

    def clear(self) -> None:
        self._documents = {}
        self._retrieved_chunks = []

    @property
    def has_documents(self) -> bool:
        return bool(self._documents) or bool(self._retrieved_chunks)

    @property
    def document_names(self) -> list[str]:
        names = set(self._documents.keys())
        for chunk in self._retrieved_chunks:
            source = chunk.get("metadata", {}).get("source")
            if source:
                names.add(source)
        return sorted(names)

    def build_system_prompt(self) -> str:
        if not self._retrieved_chunks:
            return self._no_context_prompt()

        chunks_block = self._build_chunks_block()
        if not chunks_block:
            return self._no_context_prompt()

        return f"""Es um assistente pedagogico especializado.

REGRAS IMPORTANTES:
- Responde APENAS com base nos excertos recuperados em CONTEXTO.
- Se a informacao nao estiver EXPLICITAMENTE presente nos excertos do CONTEXTO, NAO respondas a pergunta.
- NUNCA uses conhecimento externo, mesmo que saibas a resposta.
- NUNCA inventes ou assumas uma fonte.
- Se nao houver evidencia direta nos excertos, responde apenas: "Nao encontrei informacao suficiente nos materiais fornecidos."
- Usa apenas informacao EXPLICITA nos excertos.
- NAO facas inferencias, extrapolacoes ou interpretacoes alem do que esta escrito.
- Nao uses expressoes como "podemos inferir", "provavelmente", "sugere que", "parece que" ou equivalentes.
- Cada frase relevante da resposta deve indicar a fonte correspondente.
- Usa SEMPRE referencias aos documentos quando responderes.
- O formato da referencia deve ser exatamente: (nome_do_ficheiro, pagina X) ou (nome_do_ficheiro, slide X).
- Nao uses identificadores como "Excerto 1"; usa apenas as referencias reais dos documentos.
- Nao escrevas qualquer referencia se nao estiver associada a um excerto real do CONTEXTO.
- Todas as referencias devem corresponder diretamente a excertos fornecidos.
- Se os excertos tiverem informacao parcial, diz apenas o que esta explicito e indica que nao ha mais detalhe no contexto.
- Se varios excertos forem relevantes, combina-os apenas quando houver suporte explicito em cada excerto citado.
- Se direto, claro e evita respostas longas sem necessidade.

FORMATO:
- Responde em frases curtas.
- Coloca a citacao no fim da frase que ela suporta.
- Nao apresentes afirmacoes sem citacao.

CONTEXTO:
{chunks_block}
"""

    def _build_chunks_block(self) -> str:
        blocks = []
        for chunk in self._retrieved_chunks:
            metadata = chunk.get("metadata", {})
            ref = self._format_reference(metadata)
            if ref is None:
                continue
            score = chunk.get("score")
            score_text = f" | score={score:.3f}" if isinstance(score, float) else ""
            blocks.append(f"[{ref}{score_text}]\n{chunk.get('text', '').strip()}")
        return "\n\n".join(blocks)

    @staticmethod
    def _format_reference(metadata: dict[str, Any]) -> str | None:
        source = metadata.get("source")
        if not source:
            return None
        if metadata.get("page") is not None:
            return f"{source}, pagina {metadata['page']}"
        if metadata.get("slide") is not None:
            return f"{source}, slide {metadata['slide']}"
        return None

    @staticmethod
    def _no_context_prompt() -> str:
        return """Es um assistente pedagogico especializado.

Nao existe contexto recuperado dos documentos para esta pergunta.
Responde apenas: "Nao encontrei informacao suficiente nos materiais fornecidos."
"""

    def build_messages(self, history: list[dict]) -> list[dict]:
        messages = [{"role": "system", "content": self.build_system_prompt()}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        return messages
