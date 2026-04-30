"""
Módulo de construção de contexto.
"""

from typing import Optional

MAX_CONTEXT_CHARS = 120_000

class ContextBuilder:
    def __init__(self):
        self._documents: dict[str, str] = {}

    # ── gestão de documentos ────────────────────────────────────────────────

    def update_documents(self, docs: dict[str, str]) -> None:
        self._documents = docs

    def clear(self) -> None:
        self._documents = {}

    @property
    def has_documents(self) -> bool:
        return bool(self._documents)

    @property
    def document_names(self) -> list[str]:
        return list(self._documents.keys())

    # ── construção do system prompt ─────────────────────────────────────────

    def build_system_prompt(self) -> str:
        if not self._documents:
            return self._no_docs_prompt()

        docs_block = self._build_documents_block()

        return f"""És um assistente pedagógico especializado.

REGRAS IMPORTANTES:
- Responde APENAS com base nos documentos fornecidos.
- NÃO inventes informação.
- Se não encontrares a resposta, diz: "Essa informação não está nos documentos."
- Sê direto e claro.
- Evita respostas longas sem necessidade.
- Usa listas quando fizer sentido.

FORMATO DE RESPOSTA:
- Explicação clara
- (Opcional) exemplos
- (Opcional) resumo final

DOCUMENTOS:
{docs_block}
"""

    def _build_documents_block(self) -> str:
        """Constrói contexto de forma mais eficiente para modelos locais."""
        sections = []
        total_chars = 0

        for name, text in self._documents.items():
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break

            # 🔥 NOVO: corta de forma mais inteligente
            snippet = text[:remaining]

            # remove espaços excessivos
            snippet = snippet.strip()

            total_chars += len(snippet)

            sections.append(
                f"[{name}]\n{snippet}"
            )

        return "\n\n".join(sections)

    # ── fallback sem documentos ─────────────────────────────────────────────

    @staticmethod
    def _no_docs_prompt() -> str:
        return """És um assistente pedagógico.

Não existem documentos carregados.

Informa o utilizador para adicionar ficheiros à pasta 'documentos/'.

Podes responder a perguntas gerais."""
    
    # ── mensagens ───────────────────────────────────────────────────────────

    def build_messages(self, history: list[dict]) -> list[dict]:
        """Inclui system prompt como primeira mensagem (melhor para Ollama)."""
        system_prompt = self.build_system_prompt()

        messages = [{"role": "system", "content": system_prompt}]

        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        return messages