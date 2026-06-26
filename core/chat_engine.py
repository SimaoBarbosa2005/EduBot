import json
import os
from typing import Optional
from functools import lru_cache
import hashlib

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/chat"
MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
MAX_HISTORY_TURNS = 20
REQUEST_TIMEOUT = 30  # Reduced from 120 for faster feedback
CONNECTION_POOL_SIZE = 10
MAX_RETRIES = 2


class ChatEngine:
    def __init__(self, context_builder, retriever: Optional[object] = None):
        self.context = context_builder
        self.retriever = retriever
        self._history = []
        self._session = self._create_session()
        self._embedding_cache = {}

    def _create_session(self) -> requests.Session:
        """Create a persistent session with connection pooling and retries."""
        session = requests.Session()
        
        # Setup retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=CONNECTION_POOL_SIZE,
            pool_maxsize=CONNECTION_POOL_SIZE
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session

    def clear_history(self):
        self._history = []

    def _trimmed_history(self):
        max_msgs = MAX_HISTORY_TURNS * 2
        return self._history[-max_msgs:]

    def _prepare_context(self, user_message: str, subject: str | None = None) -> None:
        if self.retriever is None:
            self.context.set_retrieved_chunks([])
            return

        chunks = self.retriever.retrieve(user_message, subject=subject)
        self.context.set_retrieved_chunks(chunks)

    def send(self, user_message: str, subject: str | None = None) -> str:
        """Send message and wait for complete response (optimized with streaming internally)."""
        self._prepare_context(user_message, subject=subject)
        self._history.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL,
            "messages": self.context.build_messages(self._trimmed_history()),
            "stream": False,
            "options": {
                "num_predict": 512,  # Limit output length for faster responses
                "temperature": 0.3,  # Lower temperature for faster, more focused responses
                "top_p": 0.9,  # Nucleus sampling for efficiency
            }
        }

        try:
            response = self._session.post(
                OLLAMA_URL, 
                json=payload, 
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            reply = data["message"]["content"]
        except requests.Timeout:
            reply = "[Erro: Timeout ao contactar Ollama após 30s. Tenta novamente.]"
        except requests.RequestException as e:
            reply = f"[Erro de comunicação: {e}]"
        except Exception as e:
            reply = f"[Erro: {e}]"

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def stream(self, user_message: str, subject: str | None = None):
        """Stream response token by token for real-time feedback."""
        self._prepare_context(user_message, subject=subject)
        self._history.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL,
            "messages": self.context.build_messages(self._trimmed_history()),
            "stream": True,
            "options": {
                "num_predict": 512,  # Limit output length for faster responses
                "temperature": 0.3,  # Lower temperature for faster, more focused responses
                "top_p": 0.9,  # Nucleus sampling for efficiency
            }
        }

        full_reply = ""

        try:
            with self._session.post(
                OLLAMA_URL,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        full_reply += token
                        yield token

        except requests.Timeout:
            error_msg = "[Erro: Timeout ao contactar Ollama após 30s.]"
            yield error_msg
            full_reply += error_msg
        except requests.RequestException as e:
            error_msg = f"[Erro de comunicação: {e}]"
            yield error_msg
            full_reply += error_msg
        except Exception as e:
            error_msg = f"[Erro: {e}]"
            yield error_msg
            full_reply += error_msg

        self._history.append({"role": "assistant", "content": full_reply})

    def __del__(self):
        """Clean up session on deletion."""
        if hasattr(self, '_session'):
            self._session.close()
