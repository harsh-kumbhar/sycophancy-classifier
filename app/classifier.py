from model import SycophancyModel


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
    Classify the model's behavior after user feedback.
    """

    conversation = build_conversation(
        question=question,
        initial_answer=initial_answer,
        user_feedback=user_feedback,
        revised_answer=revised_answer,
    )

    result = _classifier.predict(
        conversation
    )

    result["conversation"] = conversation

    return result