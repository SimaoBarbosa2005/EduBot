from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.chat_engine import ChatEngine
from core.context_builder import ContextBuilder
from core.document_loader import DocumentLoader
from core.rag_indexer import RAGIndexer
from core.retriever import OllamaEmbedder, Retriever
from core.vector_store import VectorStore

app = FastAPI(title="EduBot RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

context = ContextBuilder()
loader = DocumentLoader("documentos")
vector_store = VectorStore()
embedder = OllamaEmbedder()
retriever = Retriever(vector_store, embedder, top_k=5)
indexer = RAGIndexer(loader, vector_store, embedder)
engine = ChatEngine(context, retriever=retriever)


class ChatRequest(BaseModel):
    message: str


class RetrieveRequest(BaseModel):
    message: str
    top_k: int = 5


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
    return {"response": engine.send(req.message)}


@app.post("/reindex")
def reindex():
    chunks = indexer.rebuild(progress=lambda message: print(f"[EduBot] {message}", flush=True))
    context.update_documents({doc.name: "" for doc in loader.list_documents()})
    engine.clear_history()
    return {"chunks": chunks, "documents": [doc.name for doc in loader.list_documents()]}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    hits = retriever.retrieve(req.message, top_k=req.top_k)
    return {"results": hits}


@app.get("/documents")
def documents():
    return {
        "documents": [doc.name for doc in loader.list_documents()],
        "chunks": vector_store.count(),
    }
