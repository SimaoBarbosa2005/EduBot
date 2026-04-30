"""
Interface de terminal do EduBot.
Gere a interacção com o utilizador, comandos e apresentação das respostas.
"""

import os
import sys
import textwrap
from pathlib import Path

from core.document_loader import DocumentLoader
from core.context_builder import ContextBuilder
from core.chat_engine import ChatEngine


# ── helpers de formatação ───────────────────────────────────────────────────

WIDTH = 72

def hr(char="─"):
    print(char * WIDTH)

def header(text: str):
    print()
    hr("═")
    print(f"  {text}")
    hr("═")
    print()

def section(text: str):
    print()
    hr()
    print(f"  {text}")
    hr()

def bot_prefix():
    print("\n🎓 EduBot › ", end="", flush=True)

def wrap(text: str) -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=WIDTH))
    return "\n".join(lines)


# ── interface principal ─────────────────────────────────────────────────────

class EduBotInterface:

    COMMANDS = {
        "/ajuda":      "Mostra esta mensagem de ajuda",
        "/documentos": "Lista os documentos carregados",
        "/recarregar": "Recarrega os documentos da pasta",
        "/limpar":     "Limpa o histórico da conversa",
        "/pasta":      "Mostra o caminho da pasta de documentos",
        "/sair":       "Termina o EduBot",
    }

    def __init__(self):
        self.loader = DocumentLoader(docs_folder="documentos")
        self.context = ContextBuilder()
        self.engine = ChatEngine(self.context)

    # ── arranque ────────────────────────────────────────────────────────────

    def run(self):
        self._load_documents(silent=False)
        self._print_welcome()
        self._loop()

    def _print_welcome(self):
        header("EduBot — Assistente Pedagógico")
        if self.context.has_documents:
            docs = self.context.document_names
            print(f"  ✅  {len(docs)} documento(s) carregado(s):")
            for d in docs:
                print(f"       • {d}")
        else:
            print("  ⚠️  Nenhum documento carregado.")
            print(f"     → Coloca ficheiros PDF/PPTX/TXT em:")
            print(f"       {self.loader.get_docs_folder_path()}")
            print("     → Depois usa o comando  /recarregar")
        print()
        print("  Escreve a tua pergunta ou  /ajuda  para ver os comandos.")
        print()
        hr()

    # ── loop principal ──────────────────────────────────────────────────────

    def _loop(self):
        while True:
            try:
                user_input = input("\n👤 Tu › ").strip()
            except (EOFError, KeyboardInterrupt):
                self._cmd_sair()

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input.lower())
            else:
                self._ask(user_input)

    # ── comandos ────────────────────────────────────────────────────────────

    def _handle_command(self, cmd: str):
        dispatch = {
            "/ajuda":      self._cmd_ajuda,
            "/documentos": self._cmd_documentos,
            "/recarregar": self._cmd_recarregar,
            "/limpar":     self._cmd_limpar,
            "/pasta":      self._cmd_pasta,
            "/sair":       self._cmd_sair,
        }
        fn = dispatch.get(cmd)
        if fn:
            fn()
        else:
            print(f"  ❌  Comando desconhecido: '{cmd}'  (usa /ajuda)")

    def _cmd_ajuda(self):
        section("Comandos disponíveis")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:<16} {desc}")

    def _cmd_documentos(self):
        section("Documentos carregados")
        if self.context.has_documents:
            for name in self.context.document_names:
                print(f"  📄  {name}")
        else:
            print("  Nenhum documento carregado.")

    def _cmd_recarregar(self):
        print("\n  ↺  A recarregar documentos...")
        self._load_documents(silent=False)
        self.engine.clear_history()
        print("  ✅  Pronto. Histórico de conversa limpo.")

    def _cmd_limpar(self):
        self.engine.clear_history()
        print("\n  ✅  Histórico limpo.")

    def _cmd_pasta(self):
        print(f"\n  📁  {self.loader.get_docs_folder_path()}")

    def _cmd_sair(self):
        print("\n  Até breve! 👋\n")
        sys.exit(0)

    # ── carregamento de documentos ──────────────────────────────────────────

    def _load_documents(self, silent=True):
        docs = self.loader.load_all()
        # Limpa texto de cada documento
        cleaned = {
            name: DocumentLoader.clean_text(text)
            for name, text in docs.items()
        }
        self.context.update_documents(cleaned)

        if not silent:
            if cleaned:
                print(f"  → {len(cleaned)} ficheiro(s) processado(s).")
            else:
                print("  → Pasta de documentos vazia.")

    # ── pergunta / resposta ─────────────────────────────────────────────────

    def _ask(self, question: str):
        bot_prefix()
        # Streaming: imprime token a token
        full = []
        try:
            for token in self.engine.stream(question):
                print(token, end="", flush=True)
                full.append(token)
        except Exception as e:
            print(f"\n  [Erro inesperado: {e}]")
        print("\n")
