from model import SycophancyModel
from analysis import extract_evidence
from verifier import verify_conversation
from decision_engine import make_decision


# Load model once
_classifier = SycophancyModel()


def build_conversation(
    question,
    initial_answer,
    user_feedback,
    revised_answer,
):
    """
    Construct the conversation in the same
    format used during model training.
    """

    conversation = f"""USER:
{question}

ASSISTANT (INITIAL RESPONSE):
{initial_answer}

USER (FOLLOW-UP / FEEDBACK):
{user_feedback}

ASSISTANT (REVISED RESPONSE):
{revised_answer}"""

    return conversation


def classify_sycophancy(
    question,
    initial_answer,
    user_feedback,
    revised_answer,
):
    """
    Run the complete hybrid sycophancy detection pipeline.

    Pipeline:
        1. DeBERTa classification
        2. Deterministic evidence extraction
        3. LLM verification
        4. Decision engine
    """

    # ---------------------------------------------------------
    # 1. Build conversation
    # ---------------------------------------------------------

    conversation = build_conversation(
        question=question,
        initial_answer=initial_answer,
        user_feedback=user_feedback,
        revised_answer=revised_answer,
    )

    # ---------------------------------------------------------
    # 2. DeBERTa classifier
    # ---------------------------------------------------------

    classifier_result = _classifier.predict(
        conversation
    )

    # ---------------------------------------------------------
    # 3. Deterministic evidence extraction
    # ---------------------------------------------------------

    evidence = extract_evidence(
        question=question,
        initial_answer=initial_answer,
        user_feedback=user_feedback,
        revised_answer=revised_answer,
    )

    # ---------------------------------------------------------
    # 4. LLM verification
    # ---------------------------------------------------------

    verification = verify_conversation(
        question=question,
        initial_answer=initial_answer,
        user_feedback=user_feedback,
        revised_answer=revised_answer,
    )

    # ---------------------------------------------------------
    # 5. Final hybrid decision
    # ---------------------------------------------------------

    decision = make_decision(
        classifier_result=classifier_result,
        evidence=evidence,
        verification=verification,
    )

    # ---------------------------------------------------------
    # 6. Return complete result
    # ---------------------------------------------------------

    result = {
        **decision,
        "conversation": conversation,
    }

    return result