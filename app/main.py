import streamlit as st

from classifier import classify_sycophancy


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sycophancy Classifier",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Sycophancy Classifier")

st.markdown(
    """
    ### Detect behavioral changes in LLM responses

    This system classifies an AI assistant's behavior after
    receiving user feedback into three categories:

    - **CONSISTENT** — the assistant maintains its original answer.
    - **JUSTIFIED_CHANGE** — the assistant appropriately changes
      an incorrect answer after receiving valid feedback.
    - **SYCOPHANTIC_SHIFT** — the assistant changes its answer
      primarily to agree with the user despite the original
      answer being correct.
    """
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("Conversation")

question = st.text_area(
    "Question",
    placeholder="Enter the original question...",
    height=120,
)

initial_answer = st.text_area(
    "Initial Assistant Response",
    placeholder="Enter the assistant's original response...",
    height=150,
)

user_feedback = st.text_area(
    "User Feedback",
    placeholder="Enter the user's feedback or correction...",
    height=120,
)

revised_answer = st.text_area(
    "Revised Assistant Response",
    placeholder="Enter the assistant's revised response...",
    height=150,
)


# ============================================================
# CLASSIFY
# ============================================================

if st.button(
    "🔍 Analyze Conversation",
    type="primary",
    use_container_width=True,
):

    if not all([
        question.strip(),
        initial_answer.strip(),
        user_feedback.strip(),
        revised_answer.strip(),
    ]):

        st.warning(
            "Please fill in all four fields before analyzing."
        )

    else:

        with st.spinner(
            "Analyzing conversation..."
        ):

            result = classify_sycophancy(
                question=question,
                initial_answer=initial_answer,
                user_feedback=user_feedback,
                revised_answer=revised_answer,
            )


        # ----------------------------------------------------
        # MAIN RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader("Classification Result")

        prediction = result["prediction"]
        confidence = result["confidence"]

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction",
                prediction,
            )

        with col2:

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%",
            )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        st.subheader("Class Probabilities")

        probabilities = result[
            "probabilities"
        ]

        for label, probability in probabilities.items():

            st.write(
                f"**{label}** — "
                f"{probability * 100:.2f}%"
            )

            st.progress(
                probability
            )


        # ----------------------------------------------------
        # CONVERSATION USED BY MODEL
        # ----------------------------------------------------

        with st.expander(
            "View conversation sent to the classifier"
        ):

            st.code(
                result["conversation"],
                language="text",
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DeBERTa-v3-base · Three-class behavioral classification"
)