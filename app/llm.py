"""
Gemini API wrapper for the Sycophancy Classifier project.

This file is the SINGLE source of truth for Gemini configuration.
Every module (main.py, verifier.py, tests, etc.) imports GeminiChat
from here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

# ============================================================
# LOAD PROJECT .ENV
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# Explicitly load the .env from the project root.
# override=True ensures values are refreshed if needed.
load_dotenv(ENV_PATH, override=True)

# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)


class GeminiChat:
    """
    Handles communication with the Gemini API.
    """

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                f"GEMINI_API_KEY not found.\n"
                f"Expected .env location: {ENV_PATH}"
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

        self.model = GEMINI_MODEL

    def generate_response(self, messages):
        """
        Generate a response from Gemini using a conversation history.

        Parameters
        ----------
        messages : list[dict]
            [
                {"role":"system","content":"..."},
                {"role":"user","content":"..."},
                {"role":"assistant","content":"..."}
            ]
        """

        contents = []

        for message in messages:

            role = message["role"]

            # Gemini accepts user/model roles.
            gemini_role = (
                "model"
                if role == "assistant"
                else "user"
            )

            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[
                        types.Part(
                            text=message["content"]
                        )
                    ],
                )
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return response.text

    @staticmethod
    def is_configured():
        """Useful for tests and Streamlit UI."""

        return bool(os.getenv("GEMINI_API_KEY"))