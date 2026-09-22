import streamlit as st
from dotenv import load_dotenv

from llm import GeminiChat
from classifier import classify_sycophancy
from feedback_detector import detect_feedback
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
# LOAD GEMINI
# ============================================================

@st.cache_resource
def load_gemini():
    return GeminiChat()


try:
    gemini = load_gemini()

except Exception as e:
    st.error("Failed to initialize Gemini API.")
    st.exception(e)
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analyses" not in st.session_state:
    st.session_state.analyses = []


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
                sycophantic_count /
                total_analyses
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

            st.rerun()


# ============================================================
# PAGE HEADER
# ============================================================

if mode == "💬 Live Chat":

    st.title("💬 Live Chat")

    st.caption(
        "Chat with Gemini and analyze how its responses "
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

    probabilities = latest["classifier_probabilities"]

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

    st.write("Class probabilities")

    for label, probability in probabilities.items():

        st.write(
            f"**{label}**: {probability:.2%}"
        )


# ============================================================
# CHAT INPUT
# ============================================================

user_input = st.chat_input(
    "Type your message..."
)


if user_input:

    # --------------------------------------------------------
    # Detect whether this is feedback
    # --------------------------------------------------------

    is_feedback = detect_feedback(
        user_input
    )

    # --------------------------------------------------------
    # Identify previous question + answer
    # --------------------------------------------------------

    previous_user = None

    previous_assistant = None

    if (
        is_feedback
        and len(st.session_state.messages) >= 2
    ):

        # The previous message must be the assistant's
        # response.

        if (
            st.session_state.messages[-1]["role"]
            == "assistant"
        ):

            previous_assistant = (
                st.session_state.messages[-1]["content"]
            )

            # The message before that must be the
            # corresponding user question.

            if (
                st.session_state.messages[-2]["role"]
                == "user"
            ):

                previous_user = (
                    st.session_state.messages[-2]["content"]
                )

    # --------------------------------------------------------
    # Add user message
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Generate Gemini response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Thinking..."
        ):

            try:

                response = (
                    gemini.generate_response(
                        st.session_state.messages
                    )
                )

            except Exception as e:

                st.error(
                    "Failed to generate Gemini response."
                )

                st.exception(e)

                st.stop()

        st.markdown(
            response
        )

    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    # --------------------------------------------------------
    # CLASSIFY ONLY IF FEEDBACK WAS DETECTED
    # --------------------------------------------------------

    if (
        is_feedback
        and previous_user is not None
        and previous_assistant is not None
    ):

        with st.spinner(
            "Analyzing response behavior..."
        ):

            try:

                result = classify_sycophancy(
                    question=previous_user,
                    initial_answer=previous_assistant,
                    user_feedback=user_input,
                    revised_answer=response,
                )

                st.session_state.analyses.append(
                    result
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "Classification failed."
                )

                st.exception(e)

    elif is_feedback:

        st.info(
            "Feedback detected, but there isn't a "
            "complete previous exchange to analyze."
        )