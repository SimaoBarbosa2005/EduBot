# EduBot — Assistente Inteligente de Apoio ao Ensino

EduBot é um assistente conversacional que responde a perguntas **exclusivamente com base nos documentos que carregares**, funcionando como tutor pessoal da tua disciplina.

---

## Estrutura do projecto

```
EduBot/
├── main.py               ← Ponto de entrada (executa este ficheiro)
├── requirements.txt      ← Dependências Python
├── documentos/           ← Coloca aqui PDFs, PPTXs e TXTs
├── core/
│   ├── document_loader.py   ← Lê e extrai texto dos ficheiros
│   ├── context_builder.py   ← Prepara o contexto para a API
│   └── chat_engine.py       ← Comunica com a API da Claude
└── ui/
    └── interface.py         ← Interface de terminal (chat)
```

---

## Instalação

### 1. Instala as dependências

pip install -r requirements.txt

### 2. Coloca os teus documentos

Copia os ficheiros da disciplina para a pasta `documentos/`:

```
documentos/
├── aula01_introducao.pdf
├── slides_cap2.pptx
├── resumo_teorico.txt
└── ...
```

**Formatos suportados:** `.pdf` · `.pptx` · `.ppt` · `.txt` · `.md`

### 4. Executa o EduBot

uvicorn web.web:app --reload

live service do html

---

## Comandos disponíveis no chat

| Comando         | Descrição                                      |
|-----------------|------------------------------------------------|
| `/ajuda`        | Mostra a lista de comandos                     |
| `/documentos`   | Lista os documentos actualmente carregados     |
| `/recarregar`   | Recarrega os ficheiros da pasta `documentos/`  |
| `/limpar`       | Limpa o histórico da conversa actual           |
| `/pasta`        | Mostra o caminho da pasta de documentos        |
| `/sair`         | Termina o EduBot                               |

---

## Exemplo de utilização

```
👤 Tu › O que é a normalização numa base de dados?

🎓 EduBot › Segundo o documento "aula03_bd.pdf", a normalização é o
processo de organizar os atributos e tabelas de uma base de dados
relacional de forma a reduzir a redundância de dados...
```

---

## Notas

- As respostas são geradas **apenas** com base nos documentos carregados.
- Se a informação não estiver nos documentos, o EduBot avisa-te.
- O histórico de conversa é mantido durante a sessão; usa `/limpar` para recomeçar.
- Documentos muito grandes são truncados automaticamente para respeitar os limites da API.
