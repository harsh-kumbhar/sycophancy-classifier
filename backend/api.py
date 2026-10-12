"""
FastAPI bridge for the Sycophancy Classifier.

Run from the project root:
    uvicorn backend.api:app --reload

The existing Streamlit app and Python research modules remain unchanged.
"""

from __future__ import annotations

import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "synthetic_validated.jsonl"
)
DATASET_PATH = Path(
    os.getenv("VALIDATED_DATASET_PATH", str(DEFAULT_DATASET))
)

# Existing modules use imports such as `from model import ...`.
# Adding app/ to sys.path preserves those imports without editing them.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

LABELS = {
    "CONSISTENT",
    "JUSTIFIED_CHANGE",
    "SYCOPHANTIC_SHIFT",
}

app = FastAPI(
    title="Sycophancy Classifier API",
    version="1.0.0",
    description=(
        "A thin HTTP API around the existing Python sycophancy "
        "classification and live-LLM modules."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=30000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    initial_answer: str = Field(min_length=1, max_length=30000)
    user_feedback: str = Field(min_length=1, max_length=20000)
    revised_answer: str = Field(min_length=1, max_length=30000)


class FeedbackRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)
    previous_assistant_response: str = ""
    original_question: str = ""


class DemoSimulationRequest(BaseModel):
    example: dict[str, Any]


@lru_cache(maxsize=1)
def _get_llm():
    try:
        from llm import GroqChat
        return GroqChat()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The Groq LLM could not be initialized. Check that "
                "the project's .env file contains a valid GROQ_API_KEY."
            ),
        ) from exc


def _get_classifier_function():
    try:
        from classifier import classify_sycophancy
        return classify_sycophancy
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "The classifier could not be loaded. Confirm that the "
                "local model checkpoint and tokenizer exist under "
                "model/final_model, and that dependencies are installed."
            ),
        ) from exc


def _load_examples() -> list[dict[str, Any]]:
    if not DATASET_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "The validated dataset was not found. Expected it at "
                f"{DATASET_PATH}. Set VALIDATED_DATASET_PATH if it is "
                "stored elsewhere."
            ),
        )

    examples: list[dict[str, Any]] = []

    try:
        with DATASET_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if not record.get("accepted", False):
                    continue

                example = record.get("example")
                if not isinstance(example, dict):
                    continue

                if example.get("target_label") not in LABELS:
                    continue

                required = (
                    "question",
                    "initial_answer",
                    "user_feedback",
                    "revised_answer",
                )
                if not all(
                    isinstance(example.get(key), str)
                    and example[key].strip()
                    for key in required
                ):
                    continue

                examples.append(example)

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read the validated dataset: {exc}",
        ) from exc

    if not examples:
        raise HTTPException(
            status_code=503,
            detail=(
                "The dataset was found, but no accepted examples with "
                "the expected schema were loaded."
            ),
        )

    return examples


@app.get("/api/health")
def health():
    """Lightweight check that does not load the model or call Groq."""
    return {
        "status": "ok",
        "service": "sycophancy-classifier-api",
        "dataset_configured": DATASET_PATH.is_file(),
        "model_configured": (
            PROJECT_ROOT / "model" / "final_model" / "config.json"
        ).is_file(),
    }


@app.get("/api/examples/random")
def random_example():
    """Choose randomly across all accepted labels, without preselecting a class."""
    examples = _load_examples()
    example = random.choice(examples)
    return {
        "example": example,
        "accepted_example_count": len(examples),
    }


@app.post("/api/chat/respond")
def chat_respond(request: ChatRequest):
    """Generate one assistant response using the same GroqChat wrapper."""
    llm = _get_llm()
    messages = [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]

    try:
        response = llm.generate_response(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"The live LLM request failed: {exc}",
        ) from exc

    return {"response": response}


@app.post("/api/feedback/detect")
def detect_feedback(request: FeedbackRequest):
    """Run the project's existing hybrid feedback detector."""
    try:
        from feedback_detector import analyze_feedback

        result = analyze_feedback(
            message=request.message,
            previous_assistant_response=(
                request.previous_assistant_response
            ),
            original_question=request.original_question,
            use_semantic_fallback=True,
        )
        return result
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback detection failed: {exc}",
        ) from exc


@app.post("/api/analyze")
def analyze(request: AnalyzeRequest):
    """Run the existing DeBERTa + evidence + verifier + decision pipeline."""
    classify_sycophancy = _get_classifier_function()

    try:
        result = classify_sycophancy(
            question=request.question,
            initial_answer=request.initial_answer,
            user_feedback=request.user_feedback,
            revised_answer=request.revised_answer,
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Conversation analysis failed: {exc}",
        ) from exc


@app.post("/api/demo/simulate")
def simulate_demo(request: DemoSimulationRequest):
    """
    Generate a fresh initial answer and revised answer for a validated
    example. The original target label is returned separately and is
    never sent to the classifier as an input.
    """
    example = request.example
    required = (
        "question",
        "user_feedback",
        "target_label",
    )

    if not all(key in example for key in required):
        raise HTTPException(
            status_code=422,
            detail=(
                "Example must contain question, user_feedback, "
                "and target_label."
            ),
        )

    if example["target_label"] not in LABELS:
        raise HTTPException(
            status_code=422,
            detail="Example has an unsupported target_label.",
        )

    question = example["question"]
    feedback = example["user_feedback"]

    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=422, detail="Invalid question.")
    if not isinstance(feedback, str) or not feedback.strip():
        raise HTTPException(status_code=422, detail="Invalid user feedback.")

    llm = _get_llm()
    messages = [{"role": "user", "content": question}]

    try:
        initial_answer = llm.generate_response(messages)

        messages.append(
            {"role": "assistant", "content": initial_answer}
        )
        messages.append(
            {"role": "user", "content": feedback}
        )

        revised_answer = llm.generate_response(messages)

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Live demo simulation failed: {exc}",
        ) from exc

    return {
        "conversation": {
            "question": question,
            "initial_answer": initial_answer,
            "user_feedback": feedback,
            "revised_answer": revised_answer,
        },
        "target_label": example["target_label"],
    }
