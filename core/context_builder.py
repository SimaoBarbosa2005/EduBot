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
- Usa os excertos recuperados em CONTEXTO como fonte principal.
- Quando usares informacao retirada do CONTEXTO, cita a fonte de forma natural.
- Se varias frases seguidas vierem da mesma fonte/pagina, coloca uma unica citacao no fim do paragrafo.
- Evita repetir a mesma citacao em frases consecutivas.
- O formato da referencia deve ser: (nome_do_ficheiro, pagina X), (nome_do_ficheiro, slide X) ou (titulo_da_pagina, disciplina, URL).
- Nao inventes referencias e nao cites documentos que nao estejam no CONTEXTO.
- Se o CONTEXTO for parcial, responde ao que ele permite e completa com uma explicacao geral util.
- Se a pergunta nao estiver bem coberta pelo CONTEXTO, diz isso de forma natural e depois ajuda com conhecimento geral.
- Quando usares conhecimento geral fora dos materiais, assinala com uma frase curta, por exemplo: "Fora dos materiais recuperados, em termos gerais..."
- Nao respondas apenas "Nao encontrei informacao suficiente nos materiais fornecidos" se conseguires dar uma explicacao pedagogica util.
- Se direto, claro e evita respostas longas sem necessidade.

FORMATO:
- Responde em portugues europeu.
- Prefere 2 a 5 paragrafos curtos.
- Agrupa ideias da mesma fonte no mesmo paragrafo para evitar citacoes repetidas.
- Quando fizeres listas, coloca cada ponto numa linha separada, começando por "- ".
- Usa listas apenas quando ajudarem a estudar.

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
        if metadata.get("url"):
            subject = metadata.get("subject")
            if subject:
                return f"{source}, {subject}, {metadata['url']}"
            return f"{source}, {metadata['url']}"
        return None

    @staticmethod
    def _no_context_prompt() -> str:
        return """Es um assistente pedagogico especializado.

Nao existe contexto recuperado dos documentos para esta pergunta.
Explica isso de forma breve e, se a pergunta for geral, responde com conhecimento geral.
Nao inventes referencias a documentos.
Quando a resposta nao vier dos materiais, assinala isso com uma frase curta.
Quando fizeres listas, coloca cada ponto numa linha separada, começando por "- ".
"""

    def build_messages(self, history: list[dict]) -> list[dict]:
        messages = [{"role": "system", "content": self.build_system_prompt()}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        return messages
