"""
Groq API wrapper.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH, override=True)


GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)


class GroqChat:

    def __init__(self):

        self.api_key = os.getenv("GROQ_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                f"GROQ_API_KEY not found.\n"
                f"Expected .env location: {ENV_PATH}"
            )

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1"
        )

        self.model = GROQ_MODEL

        print(f"[Groq] Using model: {self.model}")


    def generate_response(self, messages):

        response = self.client.responses.create(
            model=self.model,
            input=messages,
        )

        if not response.output_text:
            raise RuntimeError(
                "Groq returned an empty response."
            )

        return response.output_text


    @staticmethod
    def is_configured():

        return bool(
            os.getenv("GROQ_API_KEY")
        )