from model import SycophancyModel
from analysis import extract_evidence
from verifier import verify_conversation
from decision_engine import make_decision


# Load the classifier once when this module is imported.
_classifier = SycophancyModel()


def build_conversation(
    question,
    initial_answer,
    user_feedback,
    revised_answer,
):
    """Construct the conversation in the format used during model training."""
    return f"""USER:
{question}

ASSISTANT (INITIAL RESPONSE):
{initial_answer}

USER (FOLLOW-UP / FEEDBACK):
{user_feedback}

ASSISTANT (REVISED RESPONSE):
{revised_answer}"""


def classify_sycophancy(
    question,
    initial_answer,
    user_feedback,
    revised_answer,
):
    """
    Run the hybrid sycophancy detection pipeline.

    If LLM verification fails, return the DeBERTa prediction as an
    explicitly unverified fallback. Errors from the classifier and evidence
    extractor are intentionally not swallowed by the verifier fallback.
    """

    # 1. Build the conversation.
    conversation = build_conversation(
        question=question,
        initial_answer=initial_answer,
        user_feedback=user_feedback,
        revised_answer=revised_answer,
    )

    # 2. Run DeBERTa classification.
    classifier_result = _classifier.predict(conversation)

    # 3. Extract deterministic evidence.
    evidence = extract_evidence(
        question=question,
        initial_answer=initial_answer,
        user_feedback=user_feedback,
        revised_answer=revised_answer,
    )

    # 4. Verify with the LLM, then make the normal hybrid decision.
    try:
        verification = verify_conversation(
            question=question,
            initial_answer=initial_answer,
            user_feedback=user_feedback,
            revised_answer=revised_answer,
        )
    except Exception as exc:
        # Keep analysis available if Groq/verifier is unavailable.
        print(f"[Verifier unavailable; using DeBERTa fallback] {exc}")

        decision = {
            "prediction": classifier_result["prediction"],
            "confidence": classifier_result["confidence"],
            "decision_source": "DEBERTA_FALLBACK_VERIFIER_UNAVAILABLE",
            "decision_reason": (
                "LLM verification was unavailable. "
                "This prediction is based on DeBERTa only and is unverified."
            ),
            "classifier_prediction": classifier_result["prediction"],
            "classifier_confidence": classifier_result["confidence"],
            "classifier_probabilities": classifier_result.get("probabilities", {}),
            "verification": None,
            "evidence": evidence,
        }
    else:
        decision = make_decision(
            classifier_result=classifier_result,
            evidence=evidence,
            verification=verification,
        )

    # 5. Return the result in the existing response shape.
    return {
        **decision,
        "conversation": conversation,
    }
