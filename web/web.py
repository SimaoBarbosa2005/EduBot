from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.context_builder import ContextBuilder
from core.chat_engine import ChatEngine
from core.document_loader import DocumentLoader

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

context = ContextBuilder()
engine = ChatEngine(context)
loader = DocumentLoader("documentos")

docs = loader.load_all()
context.update_documents({
    name: loader.clean_text(text)
    for name, text in docs.items()
})

class ChatRequest(BaseModel):
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    return {"response": engine.send(req.message)}