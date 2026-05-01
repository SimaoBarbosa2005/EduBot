import json
from typing import Optional

import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3"
MAX_HISTORY_TURNS = 20
REQUEST_TIMEOUT = 120


class ChatEngine:
    def __init__(self, context_builder, retriever: Optional[object] = None):
        self.context = context_builder
        self.retriever = retriever
        self._history = []

    def clear_history(self):
        self._history = []

    def _trimmed_history(self):
        max_msgs = MAX_HISTORY_TURNS * 2
        return self._history[-max_msgs:]

    def _prepare_context(self, user_message: str) -> None:
        if self.retriever is None:
            self.context.set_retrieved_chunks([])
            return

        chunks = self.retriever.retrieve(user_message)
        self.context.set_retrieved_chunks(chunks)

    def send(self, user_message: str) -> str:
        self._prepare_context(user_message)
        self._history.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL,
            "messages": self.context.build_messages(self._trimmed_history()),
            "stream": False,
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            reply = data["message"]["content"]
        except Exception as e:
            reply = f"[Erro: {e}]"

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def stream(self, user_message: str):
        self._prepare_context(user_message)
        self._history.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL,
            "messages": self.context.build_messages(self._trimmed_history()),
            "stream": True,
        }

        full_reply = ""

        try:
            with requests.post(
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

        except Exception as e:
            error_msg = f"[Erro: {e}]"
            yield error_msg
            full_reply += error_msg

        self._history.append({"role": "assistant", "content": full_reply})
