import streamlit as st
from dotenv import load_dotenv

from llm import GroqChat
from classifier import classify_sycophancy
from feedback_detector import analyze_feedback
from evaluation_mode import render_evaluation_mode


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sycophancy Classifier",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# LOAD GROQ
# ============================================================

@st.cache_resource
def load_llm():
    return GroqChat()


try:
    llm = load_llm()

except Exception as e:

    st.error("Failed to initialize Groq API.")
    st.exception(e)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analyses" not in st.session_state:
    st.session_state.analyses = []

# ------------------------------------------------------------
# Current analysis thread
#
# These remain stable across multiple rounds of feedback.
#
# Example:
#
# Question
#    ↓
# Initial Answer
#    ↓
# Feedback 1 → Revised Answer 1
#    ↓
# Feedback 2 → Revised Answer 2
#    ↓
# Feedback 3 → Revised Answer 3
# ------------------------------------------------------------

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "current_answer" not in st.session_state:
    st.session_state.current_answer = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Sycophancy Classifier")

    st.divider()

    # --------------------------------------------------------
    # MODE SELECTION
    # --------------------------------------------------------

    mode = st.radio(
        "Application Mode",
        [
            "💬 Live Chat",
            "🧪 Evaluation / Demo",
        ],
    )

    st.divider()

    # --------------------------------------------------------
    # LIVE CHAT METRICS
    # --------------------------------------------------------

    if mode == "💬 Live Chat":

        st.header("Conversation Metrics")

        analyses = st.session_state.analyses

        consistent_count = sum(
            a["prediction"] == "CONSISTENT"
            for a in analyses
        )

        justified_count = sum(
            a["prediction"] == "JUSTIFIED_CHANGE"
            for a in analyses
        )

        sycophantic_count = sum(
            a["prediction"] == "SYCOPHANTIC_SHIFT"
            for a in analyses
        )

        total_analyses = len(analyses)

        st.metric(
            "Analyzed Feedback Events",
            total_analyses,
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Consistent",
                consistent_count,
            )

            st.metric(
                "Justified Change",
                justified_count,
            )

        with col2:

            st.metric(
                "Sycophantic Shift",
                sycophantic_count,
            )

        st.divider()

        if total_analyses > 0:

            sycophancy_rate = (
                sycophantic_count
                / total_analyses
            ) * 100

        else:

            sycophancy_rate = 0.0

        st.metric(
            "Sycophantic Rate",
            f"{sycophancy_rate:.1f}%",
        )

        st.divider()

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.messages = []
            st.session_state.analyses = []

            # Reset the active analysis thread.
            st.session_state.current_question = None
            st.session_state.current_answer = None

            st.rerun()


# ============================================================
# PAGE HEADER
# ============================================================

if mode == "💬 Live Chat":

    st.title("💬 Live Chat")

    st.caption(
        "Chat with the LLM and analyze how its responses "
        "change after user feedback."
    )

else:

    st.title("🧪 Evaluation / Demo")

    st.caption(
        "Evaluate the trained DeBERTa classifier using "
        "validated behavioral examples."
    )


# ============================================================
# EVALUATION MODE
# ============================================================

if mode == "🧪 Evaluation / Demo":

    render_evaluation_mode()

    st.stop()


# ============================================================
# LIVE CHAT
# ============================================================


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# LATEST ANALYSIS
# ============================================================

st.divider()

st.subheader("Latest Analysis")


if not st.session_state.analyses:

    st.info(
        "No feedback event has been analyzed yet. "
        "Challenge or correct the assistant's previous "
        "answer to trigger analysis."
    )

else:

    latest = st.session_state.analyses[-1]

    prediction = latest["prediction"]

    confidence = latest["confidence"]

    probabilities = latest[
        "classifier_probabilities"
    ]

    # --------------------------------------------------------
    # Result icon
    # --------------------------------------------------------

    icons = {
        "CONSISTENT": "🟢",
        "JUSTIFIED_CHANGE": "🟡",
        "SYCOPHANTIC_SHIFT": "🔴",
    }

    icon = icons.get(
        prediction,
        "🔎",
    )

    st.markdown(
        f"### {icon} {prediction}"
    )

    st.progress(
        float(confidence),
        text=f"Confidence: {confidence:.2%}",
    )

    st.write(
        "Class probabilities"
    )

    for label, probability in probabilities.items():

        st.write(
            f"**{label}**: "
            f"{probability:.2%}"
        )

    # --------------------------------------------------------
    # Feedback detection information
    # --------------------------------------------------------

    if "feedback_type" in latest:

        st.divider()

        st.write(
            "**Feedback Detection**"
        )

        st.write(
            f"Type: `{latest['feedback_type']}`"
        )

        st.write(
            f"Confidence: "
            f"{latest['feedback_confidence']:.2%}"
        )

        st.write(
            f"Detector: `{latest['feedback_detector']}`"
        )

        if latest.get("feedback_reason"):

            st.caption(
                latest["feedback_reason"]
            )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Type your message..."
)


# ============================================================
# PROCESS USER MESSAGE
# ============================================================

if user_input:

    # ========================================================
    # 1. DETERMINE WHETHER THIS IS FEEDBACK
    # ========================================================
    #
    # IMPORTANT:
    #
    # We no longer call:
    #
    #     detect_feedback(user_input)
    #
    # because that has no context.
    #
    # Instead, the detector receives:
    #
    #     current message
    #     previous assistant answer
    #     original question
    #
    # This allows cases such as:
    #
    #     Assistant: Ink.
    #     User: Paper
    #
    # to be detected as feedback.
    # ========================================================

    feedback_result = analyze_feedback(
        message=user_input,

        previous_assistant_response=(
            st.session_state.current_answer
            or ""
        ),

        original_question=(
            st.session_state.current_question
            or ""
        ),

        use_semantic_fallback=True,
    )

    is_feedback = feedback_result[
        "is_feedback"
    ]


    # ========================================================
    # 2. START A NEW QUESTION IF THIS IS NOT FEEDBACK
    # ========================================================
    #
    # If the user is not responding to the current answer,
    # treat this as a new question/conversation thread.
    #
    # Example:
    #
    #     Previous topic: analogy
    #
    #     User: What is Python?
    #
    # This starts a new analysis thread.
    # ========================================================

    if not is_feedback:

        st.session_state.current_question = (
            user_input
        )

        st.session_state.current_answer = None


    # ========================================================
    # 3. ADD USER MESSAGE TO CHAT
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):

        st.markdown(
            user_input
        )


    # ========================================================
    # 4. GENERATE GROQ RESPONSE
    # ========================================================

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                response = (
                    llm.generate_response(
                        st.session_state.messages
                    )
                )

            except Exception as e:

                st.error(
                    "Failed to generate Groq response."
                )

                st.exception(e)

                st.stop()

        st.markdown(
            response
        )


    # ========================================================
    # 5. SAVE ASSISTANT RESPONSE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )


    # ========================================================
    # 6. CLASSIFY FEEDBACK
    # ========================================================
    #
    # The important point here is:
    #
    #     current_question
    #
    # remains the ORIGINAL question.
    #
    #     current_answer
    #
    # is the answer immediately being challenged.
    #
    # Therefore:
    #
    # Question
    #    ↓
    # Answer 1
    #    ↓
    # Feedback 1
    #    ↓
    # Answer 2
    #    ↓
    # Feedback 2
    #
    # Both feedback events remain tied to the same
    # original question.
    # ========================================================

    if (
        is_feedback
        and st.session_state.current_question
        and st.session_state.current_answer
    ):

        with st.spinner(
            "Analyzing response behavior..."
        ):

            try:

                result = classify_sycophancy(

                    question=(
                        st.session_state.current_question
                    ),

                    initial_answer=(
                        st.session_state.current_answer
                    ),

                    user_feedback=user_input,

                    revised_answer=response,
                )


                # ====================================================
                # STORE FEEDBACK DETECTION METADATA
                # ====================================================

                result["feedback_type"] = (
                    feedback_result[
                        "feedback_type"
                    ]
                )

                result["feedback_confidence"] = (
                    feedback_result[
                        "confidence"
                    ]
                )

                result["feedback_detector"] = (
                    feedback_result[
                        "detector"
                    ]
                )

                result["feedback_reason"] = (
                    feedback_result[
                        "reason"
                    ]
                )

                result["feedback_candidate"] = (
                    feedback_result.get(
                        "candidate_answer"
                    )
                )


                # ====================================================
                # STORE ANALYSIS
                # ====================================================

                st.session_state.analyses.append(
                    result
                )


                # ====================================================
                # IMPORTANT:
                #
                # The newly generated answer becomes the answer
                # that the NEXT feedback event will challenge.
                # ====================================================

                st.session_state.current_answer = (
                    response
                )


                st.rerun()


            except Exception as e:

                st.error(
                    "Classification failed."
                )

                st.exception(e)


    # ========================================================
    # 7. FIRST ANSWER / NORMAL MESSAGE
    # ========================================================
    #
    # If this was a normal user question, the generated
    # response becomes the first answer in the new thread.
    # ========================================================

    elif not is_feedback:

        st.session_state.current_answer = (
            response
        )


    # ========================================================
    # 8. FEEDBACK WITHOUT A COMPLETE ACTIVE THREAD
    # ========================================================

    elif is_feedback:

        st.info(
            "Feedback detected, but there isn't a "
            "complete previous exchange to analyze."
        )