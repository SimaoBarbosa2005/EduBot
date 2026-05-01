"""Terminal interface for EduBot."""

import sys
import textwrap

from core.chat_engine import ChatEngine
from core.context_builder import ContextBuilder
from core.document_loader import DocumentLoader
from core.rag_indexer import RAGIndexer
from core.retriever import OllamaEmbedder, Retriever
from core.vector_store import VectorStore


WIDTH = 72


def hr(char="-"):
    print(char * WIDTH)


def header(text: str):
    print()
    hr("=")
    print(f"  {text}")
    hr("=")
    print()


def section(text: str):
    print()
    hr()
    print(f"  {text}")
    hr()


def bot_prefix():
    print("\nEduBot > ", end="", flush=True)


def wrap(text: str) -> str:
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(paragraph, width=WIDTH))
    return "\n".join(lines)


class EduBotInterface:
    COMMANDS = {
        "/ajuda": "Mostra esta mensagem de ajuda",
        "/documentos": "Lista os documentos carregados",
        "/recarregar": "Reconstrucao do indice RAG dos documentos",
        "/limpar": "Limpa o historico da conversa",
        "/pasta": "Mostra o caminho da pasta de documentos",
        "/sair": "Termina o EduBot",
    }

    def __init__(self):
        self.loader = DocumentLoader(docs_folder="documentos")
        self.context = ContextBuilder()
        self.vector_store = VectorStore()
        self.embedder = OllamaEmbedder()
        self.retriever = Retriever(self.vector_store, self.embedder, top_k=5)
        self.indexer = RAGIndexer(self.loader, self.vector_store, self.embedder)
        self.engine = ChatEngine(self.context, retriever=self.retriever)
        self.indexed_chunks = self.vector_store.count()

    def run(self):
        self._load_documents(silent=False, force=False)
        self._print_welcome()
        self._loop()

    def _print_welcome(self):
        header("EduBot - Assistente Pedagogico RAG")
        docs = self.loader.list_documents()
        if docs:
            print(f"  {len(docs)} documento(s) encontrado(s).")
            print(f"  {self.indexed_chunks} chunk(s) indexado(s).")
            for doc in docs:
                print(f"    - {doc.name}")
        else:
            print("  Nenhum documento carregado.")
            print("  Coloca ficheiros PDF/PPTX/TXT/MD em:")
            print(f"  {self.loader.get_docs_folder_path()}")
            print("  Depois usa o comando /recarregar")
        print()
        print("  Escreve a tua pergunta ou /ajuda para ver os comandos.")
        print()
        hr()

    def _loop(self):
        while True:
            try:
                user_input = input("\nTu > ").strip()
            except (EOFError, KeyboardInterrupt):
                self._cmd_sair()

            if not user_input:
                continue

            if user_input.startswith("/"):
                self._handle_command(user_input.lower())
            else:
                self._ask(user_input)

    def _handle_command(self, cmd: str):
        dispatch = {
            "/ajuda": self._cmd_ajuda,
            "/documentos": self._cmd_documentos,
            "/recarregar": self._cmd_recarregar,
            "/limpar": self._cmd_limpar,
            "/pasta": self._cmd_pasta,
            "/sair": self._cmd_sair,
        }
        fn = dispatch.get(cmd)
        if fn:
            fn()
        else:
            print(f"  Comando desconhecido: '{cmd}' (usa /ajuda)")

    def _cmd_ajuda(self):
        section("Comandos disponiveis")
        for cmd, desc in self.COMMANDS.items():
            print(f"  {cmd:<16} {desc}")

    def _cmd_documentos(self):
        section("Documentos carregados")
        docs = self.loader.list_documents()
        if docs:
            for doc in docs:
                print(f"  - {doc.name}")
            print(f"\n  Chunks indexados: {self.indexed_chunks}")
        else:
            print("  Nenhum documento carregado.")

    def _cmd_recarregar(self):
        print("\n  A reconstruir indice RAG...")
        self._load_documents(silent=False, force=True)
        self.engine.clear_history()
        print("  Pronto. Historico de conversa limpo.")

    def _cmd_limpar(self):
        self.engine.clear_history()
        print("\n  Historico limpo.")

    def _cmd_pasta(self):
        print(f"\n  {self.loader.get_docs_folder_path()}")

    def _cmd_sair(self):
        print("\n  Ate breve!\n")
        sys.exit(0)

    def _load_documents(self, silent=True, force=False):
        try:
            existing_chunks = self.vector_store.count()
            if existing_chunks > 0 and not force:
                self.indexed_chunks = existing_chunks
                self.context.update_documents({doc.name: "" for doc in self.loader.list_documents()})
                if not silent:
                    print(f"  -> A usar indice RAG existente ({self.indexed_chunks} chunks).")
                    print("     Usa /recarregar para reconstruir depois de mudares documentos.")
                return

            def progress(message: str) -> None:
                if not silent:
                    print(f"  -> {message}", flush=True)

            self.indexed_chunks = self.indexer.rebuild(progress=progress)
            self.context.update_documents({doc.name: "" for doc in self.loader.list_documents()})
            if not silent:
                print(f"  -> {self.indexed_chunks} chunk(s) indexado(s).")
        except Exception as e:
            self.indexed_chunks = self.vector_store.count()
            print(f"  [Erro] Nao foi possivel reconstruir o indice RAG: {e}")
            print("  Confirma que o Ollama esta ativo e que o modelo nomic-embed-text existe.")

    def _ask(self, question: str):
        bot_prefix()
        try:
            for token in self.engine.stream(question):
                print(token, end="", flush=True)
        except Exception as e:
            print(f"\n  [Erro inesperado: {e}]")
        print("\n")
