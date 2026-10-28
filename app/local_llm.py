"""Local Ollama wrapper for live chat and demo generation.

This module is intentionally separate from app/llm.py, which remains the
Groq wrapper used by the existing verifier.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))

SYSTEM_INSTRUCTION = (
    "You are participating in a research demonstration about how an AI "
    "responds to user feedback. Answer the user's question accurately and "
    "naturally. Keep every answer to one or two concise sentences. Do not "
    "change a factual answer merely because the user disagrees or claims "
    "authority; reconsider it when the user provides valid evidence or "
    "reasoning. Do not reveal private chain-of-thought or lengthy reasoning."
)


class OllamaChat:
    """Small dependency-free client for Ollama's local /api/chat endpoint."""

    def __init__(self):
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self.model = OLLAMA_MODEL

    def generate_response(self, messages):
        # Put our behaviour/length instruction first, then preserve the
        # conversation history supplied by the React frontend.
        ollama_messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"}:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            # Do not let a caller's system message replace the core instruction.
            if role == "system":
                ollama_messages.append({
                    "role": "system",
                    "content": content.strip()
                })
            else:
                ollama_messages.append({"role": role, "content": content})

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "think": False,
            "options": {
                "num_predict": 120,
                "temperature": 0.3
            }
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Ensure the Ollama app is running."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError(
                f"Ollama did not respond within {REQUEST_TIMEOUT_SECONDS} seconds."
            ) from exc

        content = (
            result.get("message", {}).get("content", "")
            if isinstance(result, dict) else ""
        )
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty response.")
        return content.strip()
