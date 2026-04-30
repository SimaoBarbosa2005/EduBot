"""
Módulo de carregamento e processamento de documentos.
Suporta ficheiros PDF e apresentações PowerPoint (.pptx).
"""

import os
import re
from pathlib import Path
from typing import Optional


# ── dependências opcionais ──────────────────────────────────────────────────

def _try_import_pdf():
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        return None

def _try_import_pptx():
    try:
        from pptx import Presentation
        return Presentation
    except ImportError:
        return None


# ── classe principal ────────────────────────────────────────────────────────

class DocumentLoader:
    """Carrega documentos da pasta 'documentos/' e extrai o texto."""

    SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".txt", ".md"}

    def __init__(self, docs_folder: str = "documentos"):
        # Resolve sempre relativo ao diretório deste ficheiro
        base = Path(__file__).resolve().parent.parent
        self.docs_folder = base / docs_folder
        self.docs_folder.mkdir(exist_ok=True)

    # ── listagem ────────────────────────────────────────────────────────────

    def list_documents(self) -> list[Path]:
        """Devolve a lista de ficheiros suportados na pasta de documentos."""
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(self.docs_folder.glob(f"*{ext}"))
        return sorted(files)

    # ── carregamento ────────────────────────────────────────────────────────

    def load_all(self) -> dict[str, str]:
        """Carrega todos os documentos e devolve {nome: texto}."""
        docs = {}
        files = self.list_documents()

        if not files:
            return docs

        for path in files:
            text = self._load_file(path)
            if text:
                docs[path.name] = text

        return docs

    def load_file(self, filename: str) -> Optional[str]:
        """Carrega um único ficheiro pelo nome."""
        path = self.docs_folder / filename
        if not path.exists():
            return None
        return self._load_file(path)

    # ── dispatchers internos ────────────────────────────────────────────────

    def _load_file(self, path: Path) -> Optional[str]:
        ext = path.suffix.lower()
        loaders = {
            ".pdf":  self._load_pdf,
            ".pptx": self._load_pptx,
            ".ppt":  self._load_pptx,
            ".txt":  self._load_text,
            ".md":   self._load_text,
        }
        loader = loaders.get(ext)
        if loader is None:
            return None
        try:
            return loader(path)
        except Exception as e:
            print(f"  [Aviso] Erro ao carregar '{path.name}': {e}")
            return None

    # ── leitores específicos ────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> Optional[str]:
        pdfplumber = _try_import_pdf()
        if pdfplumber is None:
            raise ImportError(
                "pdfplumber não está instalado. "
                "Execute: pip install pdfplumber"
            )

        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    pages.append(f"[Página {i}]\n{text.strip()}")

        return "\n\n".join(pages) if pages else None

    def _load_pptx(self, path: Path) -> Optional[str]:
        Presentation = _try_import_pptx()
        if Presentation is None:
            raise ImportError(
                "python-pptx não está instalado. "
                "Execute: pip install python-pptx"
            )

        prs = Presentation(str(path))
        slides = []
        for i, slide in enumerate(prs.slides, 1):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if line:
                            texts.append(line)
            if texts:
                slides.append(f"[Slide {i}]\n" + "\n".join(texts))

        return "\n\n".join(slides) if slides else None

    def _load_text(self, path: Path) -> Optional[str]:
        try:
            return path.read_text(encoding="utf-8").strip() or None
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1").strip() or None

    # ── utilitários ─────────────────────────────────────────────────────────

    @staticmethod
    def clean_text(text: str) -> str:
        """Remove espaços e linhas em branco excessivos."""
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    def get_docs_folder_path(self) -> str:
        return str(self.docs_folder)
