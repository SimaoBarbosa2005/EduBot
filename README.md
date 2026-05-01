# EduBot - Assistente Inteligente com RAG

EduBot e um assistente conversacional para apoio ao ensino e estudo. Carrega
documentos pedagogicos, cria um indice vetorial local e responde usando apenas
os excertos recuperados desses documentos.

## Como funciona

1. Os documentos em `documentos/` sao lidos por `DocumentLoader`.
2. O `RAGIndexer` divide o texto em chunks com metadados:
   - nome do ficheiro
   - pagina de PDF, quando existir
   - slide de PPTX, quando existir
   - indice do chunk
3. O `OllamaEmbedder` gera embeddings locais com `nomic-embed-text`.
4. O `VectorStore` guarda chunks e embeddings em ChromaDB, na pasta `.rag_index/`.
5. A cada pergunta, o `Retriever` procura os chunks mais relevantes.
6. O `ChatEngine` envia ao modelo apenas os chunks recuperados, nao os documentos inteiros.

## Requisitos

- Python 3.10+
- Ollama ativo localmente
- Modelo de chat, por exemplo `llama3`
- Modelo de embeddings `nomic-embed-text`

```bash
pip install -r requirements.txt
ollama pull llama3
ollama pull nomic-embed-text
```

## Executar em terminal

```bash
python main.py
```

Ao arrancar, o EduBot constroi o indice RAG com os documentos existentes.
Tambem podes usar `/recarregar` para reconstruir o indice depois de adicionares
ou alterares ficheiros.

## Executar API web

```bash
uvicorn web.web:app --reload
```

Depois abre `web/index.html` no browser, ou usa a extensao Live Server.

## Endpoints uteis

```bash
POST /chat
POST /reindex
POST /retrieve
GET  /documents
```

Exemplo para testar retrieval:

```bash
curl -X POST http://127.0.0.1:8000/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Quais sao os deveres deontologicos do engenheiro?\",\"top_k\":5}"
```

Se o retrieval estiver a funcionar, a resposta deve listar excertos com
`metadata.source` e, quando aplicavel, `metadata.page` ou `metadata.slide`.

## Formatos suportados

- `.pdf`
- `.pptx`
- `.txt`
- `.md`

`.ppt` antigo pode nao ser suportado pela biblioteca `python-pptx`; se falhar,
converte para `.pptx`.
