"""
EduBot - Assistente Inteligente para Apoio ao Ensino e Estudo
Ponto de entrada principal da aplicação
"""

import sys
import os

# Garante que o diretório raiz está no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.interface import EduBotInterface


def main():
    print("=" * 60)
    print("  EduBot — Assistente de Apoio Pedagógico")
    print("=" * 60)
    print()

    app = EduBotInterface()
    app.run()


if __name__ == "__main__":
    main()
