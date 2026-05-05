from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.chat_engine import ChatEngine
from core.context_builder import ContextBuilder
from core.document_loader import DocumentLoader
from core.rag_indexer import RAGIndexer
from core.retriever import OllamaEmbedder, Retriever
from core.vector_store import VectorStore
from core.web_crawler import DEFAULT_CRAWL_SOURCES, CrawlSource, WebCrawler, default_sources

app = FastAPI(title="EduBot RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse("web/index.html")

context = ContextBuilder()
loader = DocumentLoader("documentos")
vector_store = VectorStore()
embedder = OllamaEmbedder()
retriever = Retriever(vector_store, embedder, top_k=5)
indexer = RAGIndexer(loader, vector_store, embedder)
crawler = WebCrawler(indexer)
engine = ChatEngine(context, retriever=retriever)


class ChatRequest(BaseModel):
    message: str
    subject: str | None = None


class RetrieveRequest(BaseModel):
    message: str
    top_k: int = 5
    subject: str | None = None


class CrawlRequest(BaseModel):
    subject: str
    urls: list[str]
    max_pages_per_seed: int = 2


@app.on_event("startup")
def startup_rebuild_index():
    try:
        if vector_store.count() > 0:
            context.update_documents({doc.name: "" for doc in loader.list_documents()})
            print(f"[EduBot] A usar indice RAG existente com {vector_store.count()} chunk(s).")
            return

        chunks = indexer.rebuild(progress=lambda message: print(f"[EduBot] {message}", flush=True))
        context.update_documents({doc.name: "" for doc in loader.list_documents()})
        print(f"[EduBot] Indice RAG reconstruido com {chunks} chunk(s).")
    except Exception as e:
        print(f"[EduBot] Nao foi possivel reconstruir o indice RAG: {e}")
        print("[EduBot] Confirma que o Ollama esta ativo e que existe o modelo nomic-embed-text.")


@app.post("/chat")
def chat(req: ChatRequest):
    """Send a message and get a streaming response."""
    def generate_response():
        try:
            for token in engine.stream(req.message, subject=req.subject):
                yield token
        except Exception as e:
            yield f"[Erro: {e}]"
    
    return StreamingResponse(generate_response(), media_type="text/plain")


@app.post("/chat-blocking")
def chat_blocking(req: ChatRequest):
    """Send a message and wait for complete response."""
    return {"response": engine.send(req.message, subject=req.subject)}


@app.post("/reindex")
def reindex():
    chunks = indexer.rebuild(progress=lambda message: print(f"[EduBot] {message}", flush=True))
    context.update_documents({doc.name: "" for doc in loader.list_documents()})
    engine.clear_history()
    return {"chunks": chunks, "documents": [doc.name for doc in loader.list_documents()]}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    hits = retriever.retrieve(req.message, top_k=req.top_k, subject=req.subject)
    return {"results": hits}


@app.get("/documents")
def documents():
    return {
        "documents": [doc.name for doc in loader.list_documents()],
        "chunks": vector_store.count(),
        "by_source_type": vector_store.metadata_counts("source_type"),
        "by_subject": vector_store.metadata_counts("subject"),
    }


@app.get("/crawl-sources")
def crawl_sources():
    return {"sources": DEFAULT_CRAWL_SOURCES}


@app.post("/crawl")
def crawl(req: CrawlRequest):
    result = crawler.crawl_subject(
        req.subject,
        req.urls,
        max_pages_per_seed=req.max_pages_per_seed,
        progress=lambda message: print(f"[EduBot crawler] {message}", flush=True),
    )
    engine.clear_history()
    return result


@app.post("/crawl-all")
def crawl_all(max_pages_per_seed: int = 2):
    sources = default_sources(max_pages=max_pages_per_seed)
    results = crawler.crawl_sources(
        sources,
        progress=lambda message: print(f"[EduBot crawler] {message}", flush=True),
    )
    engine.clear_history()
    return {"results": results, "chunks": vector_store.count()}
