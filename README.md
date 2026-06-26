# EduBot - Assistente Inteligente com RAG

EduBot é um assistente conversacional para apoio ao ensino e estudo. Carrega
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

## Executar com Docker

Antes de arrancar o container, confirma que o Ollama esta ativo no host e que
os modelos existem:

```bash
ollama serve
ollama pull llama3
ollama pull nomic-embed-text
```

Depois executa:

```bash
docker compose up --build
```

Abre `http://127.0.0.1:8000` no browser. O `docker-compose.yml` monta
`documentos/` como leitura e guarda o indice ChromaDB em `.rag_index/`, para o
indice persistir entre reinicios.

Podes alterar os modelos ou o URL do Ollama com variaveis de ambiente:

```bash
OLLAMA_BASE_URL=http://host.docker.internal:11434 docker compose up --build
OLLAMA_CHAT_MODEL=mistral OLLAMA_EMBED_MODEL=nomic-embed-text docker compose up --build
```

## Endpoints uteis

```bash
POST /chat                  # Streaming endpoint (recomendado)
POST /chat-blocking         # Blocking endpoint (espera resposta completa)
POST /reindex
POST /retrieve
GET  /documents
GET  /crawl-sources         # Lista fontes web por disciplina
POST /crawl                 # Crawla URLs de uma disciplina e guarda em ChromaDB
POST /crawl-all             # Crawla as fontes predefinidas
```

Exemplo para testar retrieval:

```bash
curl -X POST http://127.0.0.1:8000/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Quais sao os deveres deontologicos do engenheiro?\",\"top_k\":5}"
```

Se o retrieval estiver a funcionar, a resposta deve listar excertos com
`metadata.source` e, quando aplicavel, `metadata.page` ou `metadata.slide`.

## Web crawlers por disciplina

O EduBot tambem consegue recolher conteudo web educativo e guardar os chunks na
mesma base ChromaDB usada pelos PDFs. Cada chunk web fica com metadados como:

- `source_type`: `web`
- `subject`: disciplina, por exemplo `filosofia`, `matematica`, `historia`, `ciencias`
- `url`: pagina original
- `title`: titulo da pagina

Exemplo para crawlar uma pagina de Filosofia:

```bash
curl -X POST http://127.0.0.1:8000/crawl ^
  -H "Content-Type: application/json" ^
  -d "{\"subject\":\"filosofia\",\"urls\":[\"https://pt.wikipedia.org/wiki/Arist%C3%B3teles\"],\"max_pages_per_seed\":1}"
```

Exemplo para crawlar todas as fontes predefinidas:

```bash
curl -X POST "http://127.0.0.1:8000/crawl-all?max_pages_per_seed=1"
```

Depois podes filtrar retrieval/chat por disciplina:

```bash
curl -X POST http://127.0.0.1:8000/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Fala-me de Aristoteles\",\"subject\":\"filosofia\",\"top_k\":3}"
```

No frontend tambem existe um seletor de disciplina. Se escolheres uma disciplina,
o Ollama recebe contexto recuperado do ChromaDB apenas dessa area.

## Formatos suportados

- `.pdf`
- `.pptx`
- `.txt`
- `.md`

`.ppt` antigo pode nao ser suportado pela biblioteca `python-pptx`; se falhar,
converte para `.pptx`.

---

## Otimizacoes de Performance

EduBot foi otimizado para respostas mais rapidas:

### 1. **Reducao de Timeouts**
- Timeout de requests reduzido de 120s para 30s
- Feedback mais rapido em caso de erro
- Modelos configurados para respostas mais concisas

### 2. **Connection Pooling**
- Reutilizacao de conexoes HTTP persistentes
- Reduz overhead de criacao de conexoes
- 10 conexoes simultaneas no pool

### 3. **Embedding Cache**
- Cache MD5 de embeddings já calculados
- Reduz calculos desnecessarios em perguntas repetidas ou similares
- Limpa automaticamente se necessario

### 4. **Batch Processing**
- Embeddings processados em lotes de 10 chunks
- Reduz latencia durante indexacao
- Melhor eficiencia computacional

### 5. **Streaming por Defecto**
- Terminal e web API usam streaming
- Tokens aparecem em tempo real na interface
- Melhor experiencia de usuario

### 6. **Otimizacao do Modelo**
- `num_predict: 512` - limita tamanho de respostas
- `temperature: 0.3` - respostas mais focadas e rapidas
- `top_p: 0.9` - nucleus sampling para eficiencia

### 7. **Retry Automático**
- Reconexao automatica em falhas temporarias
- Backoff exponencial para evitar congestao

## Configuracoes Ajustaveis

Para personalizar a performance, edita as constantes em:

### `core/chat_engine.py`
```python
REQUEST_TIMEOUT = 30          # Tempo maximo de espera (segundos)
CONNECTION_POOL_SIZE = 10     # Conexoes simultaneas
MAX_RETRIES = 2               # Tentativas de reconexao

# Opcoes de modelo (em send() e stream())
"num_predict": 512            # Limita output do modelo
"temperature": 0.3            # Controlabilidade (0.0-1.0)
"top_p": 0.9                  # Diversidade (0.0-1.0)
```

### `core/retriever.py`
```python
EMBED_TIMEOUT = 30            # Timeout para embeddings
EMBED_BATCH_SIZE = 10         # Tamanho de lote de embeddings
```

### `core/rag_indexer.py`
```python
EMBEDDING_BATCH_SIZE = 10     # Tamanho de lote durante indexacao
```

## Dicas de Performance

### Para respostas ainda mais rapidas:
1. **Usar modelo mais leve**: Tentar `mistral` em vez de `llama3`
   ```bash
   ollama pull mistral
   ```
   Depois edita `MODEL = "mistral"` em `chat_engine.py`

2. **Reduzir num_predict**: Para respostas muito curtas
   ```python
   "num_predict": 256  # Resposta ainda mais curta
   ```

3. **Aumentar temperature para 0.5**: Mais natural, potencialmente mais rapido
   ```python
   "temperature": 0.5
   ```

4. **Reduzir top_k**: Recupera menos chunks
   ```python
   self.retriever = Retriever(self.vector_store, self.embedder, top_k=3)
   ```

### Para respostas de melhor qualidade:
1. **Aumentar temperature**: Para mais criatividade
   ```python
   "temperature": 0.7
   ```

2. **Reduzir top_p**: Para mais focar
   ```python
   "top_p": 0.8
   ```

3. **Aumentar num_predict**: Para respostas mais detalhadas
   ```python
   "num_predict": 1024
   ```

4. **Aumentar top_k**: Mais contexto disponivel
   ```python
   self.retriever = Retriever(self.vector_store, self.embedder, top_k=10)
   ```

## Benchmarks Tipicos

Com hardware moderado e modelo `llama3`:

- **Tempo medio de resposta**: 3-8 segundos
- **Tempo de embedding**: 100-500ms
- **Latencia da API**: <100ms
- **Indexacao**: ~200-500 chunks/minuto

*Tempos variam com hardware, tamanho dos chunks e quantidade de contexto recuperado.*

## Troubleshooting

### "Timeout ao contactar Ollama"
- Verifica se Ollama esta a correr: `ollama serve`
- Aumenta REQUEST_TIMEOUT se o hardware for lento
- Tenta reduzir num_predict para respostas mais rapidas

### Respostas muito lentas
- Verifica o uso de CPU/GPU em Ollama
- Reduz top_k para menos chunks
- Usa um modelo mais leve (mistral vs llama3)
- Aumenta temperatura para respostas menos elaboradas

### Cache de embeddings nao funciona
- Limpa o cache: `embedder.clear_cache()` no codigo
- Verifica se perguntas sao realmente identicas

### Indexacao lenta
- Reduz EMBEDDING_BATCH_SIZE para usar menos memoria
- Aumenta EMBEDDING_BATCH_SIZE se houver memoria disponivel
- Tenta com modelo embeddings mais leve (se disponivel)

