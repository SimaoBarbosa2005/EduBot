import requests
import json

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3"
MAX_HISTORY_TURNS = 20


class ChatEngine:

    def __init__(self, context_builder):
        self.context = context_builder
        self._history = []

    def clear_history(self):
        self._history = []

    def _trimmed_history(self):
        max_msgs = MAX_HISTORY_TURNS * 2
        return self._history[-max_msgs:]

    # ✅ método normal (caso queiras usar)
    def send(self, user_message: str) -> str:
        self._history.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL,
            "messages": self.context.build_messages(self._trimmed_history()),
            "stream": False
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload)
            data = response.json()
            reply = data["message"]["content"]
        except Exception as e:
            reply = f"[Erro: {e}]"

        self._history.append({"role": "assistant", "content": reply})
        return reply

    def stream(self, user_message: str):
        self._history.append({"role": "user", "content": user_message})

        payload = {
            "model": MODEL,
            "messages": self.context.build_messages(self._trimmed_history()),
            "stream": True
        }

        full_reply = ""

        try:
            with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
                for line in r.iter_lines():
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