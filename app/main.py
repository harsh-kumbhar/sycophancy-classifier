import re

import streamlit as st
from dotenv import load_dotenv

from llm import GroqChat
from classifier import classify_sycophancy
from feedback_detector import analyze_feedback
from evaluation_mode import render_evaluation_mode


# ============================================================
# LLM MARKDOWN / LATEX RENDERING
# ============================================================

def render_llm_response(content: str) -> str:
    """
    Normalize common LaTeX delimiters produced by LLMs so
    Streamlit renders them as math instead of showing raw
    delimiter syntax.
    """

    if not isinstance(content, str):
        return content

    content = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        lambda match: f"\n$$\n{match.group(1).strip()}\n$$\n",
        content,
        flags=re.DOTALL,
    )

    content = re.sub(
        r"\\\(\s*(.*?)\s*\\\)",
        lambda match: f"${match.group(1).strip()}$",
        content,
        flags=re.DOTALL,
    )

    return content



# ============================================================
# UI HELPERS
# ============================================================

LABEL_ICONS = {
    "CONSISTENT": "🟢",
    "JUSTIFIED_CHANGE": "🟡",
    "SYCOPHANTIC_SHIFT": "🔴",
}

LABEL_ACCENTS = {
    "CONSISTENT": "#16a34a",
    "JUSTIFIED_CHANGE": "#d97706",
    "SYCOPHANTIC_SHIFT": "#dc2626",
}


def inject_live_css():
    st.markdown(
        """
        <style>
        .live-hero {
            padding: 22px 26px;
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                rgba(120,120,120,.08),
                rgba(120,120,120,.025)
            );
            margin-bottom: 20px;
        }

        .live-hero-title {
            font-size: 30px;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.5px;
        }

        .live-hero-sub {
            margin-top: 7px;
            color: rgba(128,128,128,.95);
            font-size: 15px;
        }

        .section-label {
            font-size: 13px;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: rgba(128,128,128,.9);
            margin: 20px 0 9px;
        }

        .analysis-card {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 20px;
            padding: 22px;
            background: rgba(128,128,128,.035);
            margin: 10px 0 18px;
        }

        .analysis-kicker {
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .08em;
            font-weight: 750;
            color: rgba(128,128,128,.9);
        }

        .analysis-label {
            font-size: 28px;
            font-weight: 850;
            margin: 5px 0 8px;
        }

        .analysis-meta {
            color: rgba(128,128,128,.95);
            font-size: 14px;
        }

        .info-card {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 15px;
            padding: 16px 18px;
            background: rgba(128,128,128,.025);
            height: 100%;
        }

        .info-title {
            font-weight: 750;
            margin-bottom: 6px;
        }

        .info-body {
            color: rgba(128,128,128,.95);
            line-height: 1.5;
            font-size: 14px;
        }

        .feedback-pill {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 750;
            background: rgba(120,120,120,.10);
            border: 1px solid rgba(120,120,120,.20);
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(120,120,120,.14);
            border-radius: 13px;
            padding: 12px 14px;
            background: rgba(128,128,128,.025);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_analysis(result):
    """Render the latest behavioral analysis using the current result schema."""

    prediction = result.get("prediction", "UNKNOWN")
    confidence = float(result.get("confidence", 0.0))
    probabilities = result.get("classifier_probabilities", {})
    accent = LABEL_ACCENTS.get(prediction, "#64748b")
    icon = LABEL_ICONS.get(prediction, "🔎")

    st.markdown(
        '<div class="section-label">Latest Behavioral Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="analysis-card"
             style="border-left: 5px solid {accent};">
            <div class="analysis-kicker">Final classification</div>
            <div class="analysis-label">
                {icon} {prediction.replace("_", " ")}
            </div>
            <div class="analysis-meta">
                Confidence: <strong>{confidence:.2%}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(
        confidence,
        text=f"Model confidence: {confidence:.2%}",
    )

    st.markdown(
        '<div class="section-label">Class Probabilities</div>',
        unsafe_allow_html=True,
    )

    for label in [
        "CONSISTENT",
        "JUSTIFIED_CHANGE",
        "SYCOPHANTIC_SHIFT",
    ]:
        probability = float(
            probabilities.get(label, 0.0)
        )

        st.write(
            f"{LABEL_ICONS[label]} **{label.replace('_', ' ')}** — "
            f"{probability:.2%}"
        )
        st.progress(probability)

    decision_source = result.get(
        "decision_source",
        "N/A",
    )

    decision_reason = result.get(
        "decision_reason",
        "",
    )

    left, right = st.columns(2)

    with left:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">⚙️ Decision Source</div>
                <div class="info-body">
                    {decision_source}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">💡 Why this classification?</div>
                <div class="info-body">
                    {decision_reason or "No decision explanation returned."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    feedback_type = result.get("feedback_type")

    if feedback_type:
        st.markdown(
            '<div class="section-label">Feedback Detection</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <span class="feedback-pill">
                Feedback: {feedback_type}
            </span>
            """,
            unsafe_allow_html=True,
        )

        feedback_confidence = result.get(
            "feedback_confidence"
        )

        feedback_detector = result.get(
            "feedback_detector"
        )

        if feedback_confidence is not None:
            st.caption(
                f"Detector confidence: {float(feedback_confidence):.2%}"
                + (
                    f" · Detector: `{feedback_detector}`"
                    if feedback_detector
                    else ""
                )
            )

    with st.expander("🔎 Evidence Extraction"):
        evidence = result.get("evidence", {})
        if evidence:
            st.json(evidence)
        else:
            st.info("No evidence details returned.")

    with st.expander("🤖 LLM Verification"):
        verification = result.get("verification", {})
        if verification:
            st.json(verification)
        else:
            st.info("No verification details returned.")



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

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "current_answer" not in st.session_state:
    st.session_state.current_answer = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Sycophancy Classifier")
    st.caption("Detect behavioral shifts in multi-turn LLM conversations.")

    st.divider()

    mode = st.radio(
        "Application Mode",
        [
            "💬 Live Chat",
            "🧪 Evaluation / Demo",
        ],
    )

    st.divider()

    if mode == "💬 Live Chat":

        st.header("Conversation Metrics")

        analyses = st.session_state.analyses

        consistent_count = sum(
            a.get("prediction") == "CONSISTENT"
            for a in analyses
        )

        justified_count = sum(
            a.get("prediction") == "JUSTIFIED_CHANGE"
            for a in analyses
        )

        sycophantic_count = sum(
            a.get("prediction") == "SYCOPHANTIC_SHIFT"
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

        sycophancy_rate = (
            (sycophantic_count / total_analyses) * 100
            if total_analyses
            else 0.0
        )

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
            st.session_state.current_question = None
            st.session_state.current_answer = None
            st.rerun()


# ============================================================
# PAGE HEADER
# ============================================================

if mode == "💬 Live Chat":

    inject_live_css()

    st.markdown(
        """
        <div class="live-hero">
            <div class="live-hero-title">💬 Live Chat</div>
            <div class="live-hero-sub">
                Chat with the LLM and analyze how its responses
                change after user feedback.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.title("🧪 Evaluation / Demo")
    st.caption(
        "Load validated examples and reproduce their behavioral pattern "
        "with a live LLM."
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

# ------------------------------------------------------------
# DISPLAY CHAT HISTORY
# ------------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            render_llm_response(message["content"])
        )


# ------------------------------------------------------------
# LATEST ANALYSIS
# ------------------------------------------------------------

if st.session_state.analyses:

    render_analysis(
        st.session_state.analyses[-1]
    )

else:

    st.markdown(
        '<div class="section-label">Analysis</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "No feedback event has been analyzed yet. "
        "Ask the assistant a question first, then challenge "
        "or correct its answer."
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

    # --------------------------------------------------------
    # 1. DETERMINE WHETHER THIS IS FEEDBACK
    # --------------------------------------------------------

    feedback_result = analyze_feedback(
        message=user_input,
        previous_assistant_response=(
            st.session_state.current_answer or ""
        ),
        original_question=(
            st.session_state.current_question or ""
        ),
        use_semantic_fallback=True,
    )

    is_feedback = feedback_result["is_feedback"]

    # --------------------------------------------------------
    # 2. START A NEW QUESTION IF THIS IS NOT FEEDBACK
    # --------------------------------------------------------

    if not is_feedback:
        st.session_state.current_question = user_input
        st.session_state.current_answer = None

    # --------------------------------------------------------
    # 3. ADD USER MESSAGE TO CHAT
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------
    # 4. GENERATE GROQ RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                response = llm.generate_response(
                    st.session_state.messages
                )

            except Exception as e:
                st.error(
                    "Failed to generate Groq response."
                )
                st.exception(e)
                st.stop()

        st.markdown(
            render_llm_response(response)
        )

    # --------------------------------------------------------
    # 5. SAVE ASSISTANT RESPONSE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    # --------------------------------------------------------
    # 6. CLASSIFY FEEDBACK
    # --------------------------------------------------------

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

                result["feedback_type"] = (
                    feedback_result["feedback_type"]
                )

                result["feedback_confidence"] = (
                    feedback_result["confidence"]
                )

                result["feedback_detector"] = (
                    feedback_result["detector"]
                )

                result["feedback_reason"] = (
                    feedback_result["reason"]
                )

                result["feedback_candidate"] = (
                    feedback_result.get("candidate_answer")
                )

                st.session_state.analyses.append(
                    result
                )

                # The newly generated response becomes the answer
                # challenged by the next feedback event.
                st.session_state.current_answer = response

                st.rerun()

            except Exception as e:

                st.error(
                    "Classification failed."
                )
                st.exception(e)

    # --------------------------------------------------------
    # 7. FIRST ANSWER / NORMAL MESSAGE
    # --------------------------------------------------------

    elif not is_feedback:

        st.session_state.current_answer = response

    # --------------------------------------------------------
    # 8. FEEDBACK WITHOUT A COMPLETE ACTIVE THREAD
    # --------------------------------------------------------

    elif is_feedback:

        st.info(
            "Feedback detected, but there isn't a complete "
            "previous exchange to analyze."
        )
