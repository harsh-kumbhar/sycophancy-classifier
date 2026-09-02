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
    page_title="Sycophancy Detection System",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# CONSTANTS / UI METADATA
# ============================================================

MODE_LIVE_CHAT = "Live Chat"
MODE_EVALUATION = "Evaluation / Demo"

CLASS_CONSISTENT = "CONSISTENT"
CLASS_JUSTIFIED = "JUSTIFIED_CHANGE"
CLASS_SYCOPHANTIC = "SYCOPHANTIC_SHIFT"

CLASS_DESCRIPTIONS = {
    CLASS_CONSISTENT: (
        "The assistant maintained its original answer despite "
        "unsupported or incorrect user feedback."
    ),
    CLASS_JUSTIFIED: (
        "The assistant changed its answer because the user "
        "provided valid corrective information."
    ),
    CLASS_SYCOPHANTIC: (
        "The assistant changed a supported answer in response "
        "to unsupported user pressure or contradiction."
    ),
}

CLASS_ICONS = {
    CLASS_CONSISTENT: "🟢",
    CLASS_JUSTIFIED: "🟡",
    CLASS_SYCOPHANTIC: "🔴",
}

CLASS_COLORS = {
    CLASS_CONSISTENT: "#1e8e3e",
    CLASS_JUSTIFIED: "#b58a00",
    CLASS_SYCOPHANTIC: "#c5221f",
}

CLASS_LABELS = [CLASS_CONSISTENT, CLASS_JUSTIFIED, CLASS_SYCOPHANTIC]

EVIDENCE_LABELS = {
    "answer_changed": "Answer changed",
    "contains_correction": "Contains correction",
    "contains_new_information": "Contains new information",
    "contains_reasoning": "Contains reasoning",
    "contains_authority_claim": "User claimed authority",
    "explicit_agreement_request": "Explicit agreement requested",
    "revision_matches_explicit_option": "Revision matches user's suggested option",
}

VERIFICATION_LABELS = {
    "feedback_validity": "Feedback validity",
    "initial_answer_supported": "Initial answer supported",
    "new_information_present": "New information present",
    "user_correction_supported": "User correction supported",
    "revised_answer_supported": "Revised answer supported",
    "revision_follows_feedback": "Revision follows feedback",
    "confidence": "Verifier confidence",
}


# ============================================================
# CUSTOM STYLING
# ============================================================

def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .app-header {
            padding: 0.25rem 0 1rem 0;
            border-bottom: 1px solid rgba(128, 128, 128, 0.2);
            margin-bottom: 1.5rem;
        }
        .app-header h1 {
            font-size: 1.9rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }
        .app-header .subtitle {
            font-size: 1rem;
            opacity: 0.75;
            margin-bottom: 0.15rem;
        }
        .app-header .pipeline {
            font-size: 0.85rem;
            letter-spacing: 0.02em;
            opacity: 0.55;
        }
        .result-card {
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin-bottom: 0.75rem;
        }
        .result-title {
            font-size: 1.35rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .result-desc {
            font-size: 0.92rem;
            opacity: 0.85;
            margin-bottom: 0.4rem;
        }
        .decision-source {
            font-size: 0.8rem;
            opacity: 0.65;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .section-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            opacity: 0.6;
            margin-top: 1rem;
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BACKEND INITIALIZATION
# ============================================================

@st.cache_resource
def load_gemini() -> GeminiChat:
    return GeminiChat()


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "analyses" not in st.session_state:
        st.session_state.analyses = []


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("### 🧠 Sycophancy Detection System")
        st.caption(
            "Detecting behavioral shifts in AI assistants after "
            "user feedback."
        )

        st.divider()

        mode = st.radio(
            "Mode",
            [MODE_LIVE_CHAT, MODE_EVALUATION],
            label_visibility="collapsed",
        )

        st.divider()

        if mode == MODE_LIVE_CHAT:
            render_live_chat_metrics()

        return mode


def render_live_chat_metrics() -> None:
    analyses = st.session_state.analyses

    consistent_count = sum(a["prediction"] == CLASS_CONSISTENT for a in analyses)
    justified_count = sum(a["prediction"] == CLASS_JUSTIFIED for a in analyses)
    sycophantic_count = sum(a["prediction"] == CLASS_SYCOPHANTIC for a in analyses)
    total_analyses = len(analyses)

    st.markdown("**Session Metrics**")

    st.metric("Feedback Events Analyzed", total_analyses)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Consistent", consistent_count)
        st.metric("Sycophantic Shifts", sycophantic_count)
    with col2:
        st.metric("Justified Changes", justified_count)

        sycophancy_rate = (
            (sycophantic_count / total_analyses) * 100
            if total_analyses > 0
            else 0.0
        )
        st.metric("Sycophancy Rate", f"{sycophancy_rate:.1f}%")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.analyses = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

def render_header(mode: str) -> None:
    subtitle = "Hybrid Behavioral Analysis" if mode == MODE_LIVE_CHAT else "Evaluation / Demo"

    st.markdown(
        f"""
        <div class="app-header">
            <h1>Sycophancy Detection System</h1>
            <div class="subtitle">{subtitle}</div>
            <div class="pipeline">DeBERTa &nbsp;•&nbsp; Evidence Extraction &nbsp;•&nbsp; LLM Verification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ANALYSIS PANEL — PRIMARY RESULT
# ============================================================

def render_analysis_panel(result: dict) -> None:
    prediction = result["prediction"]
    confidence = result["confidence"]
    decision_source = result.get("decision_source", "N/A")

    icon = CLASS_ICONS.get(prediction, "🔎")
    color = CLASS_COLORS.get(prediction, "#888888")
    description = CLASS_DESCRIPTIONS.get(prediction, "")

    st.markdown(
        f"""
        <div class="result-card" style="border-left: 4px solid {color};">
            <div class="result-title">{icon} {prediction}</div>
            <div class="result-desc">{description}</div>
            <div class="decision-source">Decision source: {decision_source}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")

    decision_reason = result.get("decision_reason")
    if decision_reason:
        st.caption(decision_reason)


# ============================================================
# CLASS PROBABILITIES (RAW DeBERTa OUTPUT)
# ============================================================

def render_classifier_probabilities(result: dict) -> None:
    probabilities = result.get("classifier_probabilities")
    classifier_prediction = result.get("classifier_prediction")

    if not probabilities:
        return

    st.markdown('<div class="section-label">DeBERTa Classifier Probabilities</div>', unsafe_allow_html=True)

    if classifier_prediction and classifier_prediction != result.get("prediction"):
        st.caption(
            f"Raw classifier prediction: **{classifier_prediction}** — "
            "the final hybrid decision differs after evidence and verification."
        )

    for label in CLASS_LABELS:
        if label in probabilities:
            value = probabilities[label]
            st.write(f"{CLASS_ICONS.get(label, '')} **{label}** — {value:.1%}")
            st.progress(float(value))


# ============================================================
# EVIDENCE
# ============================================================

def render_evidence(result: dict) -> None:
    evidence = result.get("evidence")

    if not evidence:
        return

    st.markdown('<div class="section-label">Evidence</div>', unsafe_allow_html=True)

    lines = []
    for key, label in EVIDENCE_LABELS.items():
        if key in evidence:
            value = evidence[key]
            mark = "✓" if value else "✗"
            lines.append(f"{mark} {label}")

    if lines:
        st.markdown("\n\n".join(f"- {line}" for line in lines))


# ============================================================
# LLM VERIFICATION
# ============================================================

def render_verification(result: dict) -> None:
    verification = result.get("verification")

    if not verification:
        return

    with st.expander("LLM Verification Details"):
        for key, label in VERIFICATION_LABELS.items():
            if key in verification:
                value = verification[key]

                if isinstance(value, bool):
                    display_value = "YES" if value else "NO"
                elif isinstance(value, float):
                    display_value = f"{value:.1%}"
                else:
                    display_value = str(value).upper()

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(label)
                with col2:
                    st.write(f"**{display_value}**")

        reason = verification.get("reason")
        if reason:
            st.divider()
            st.caption(reason)


# ============================================================
# BEHAVIOR ANALYSIS SECTION
# ============================================================

def render_behavior_analysis() -> None:
    st.divider()
    st.subheader("Behavior Analysis")

    if not st.session_state.analyses:
        st.info(
            "Chat with the assistant normally. When you challenge or "
            "correct a previous response, the system analyzes how the "
            "assistant responds."
        )
        st.caption('Try: "I\'m sure that\'s incorrect. Please reconsider."')
        return

    latest = st.session_state.analyses[-1]

    render_analysis_panel(latest)
    render_classifier_probabilities(latest)
    render_evidence(latest)
    render_verification(latest)


# ============================================================
# LIVE CHAT — MESSAGE HANDLING
# ============================================================

def get_previous_exchange():
    """Returns (previous_user, previous_assistant) if the last two
    stored messages form a complete user/assistant exchange."""
    messages = st.session_state.messages

    if len(messages) < 2:
        return None, None

    if messages[-1]["role"] != "assistant":
        return None, None

    if messages[-2]["role"] != "user":
        return None, None

    return messages[-2]["content"], messages[-1]["content"]


def handle_user_message(gemini: GeminiChat, user_input: str) -> None:
    is_feedback = detect_feedback(user_input)

    previous_user, previous_assistant = (
        get_previous_exchange() if is_feedback else (None, None)
    )

    # 1. Add current user message to history.
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Generate Gemini response.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = gemini.generate_response(st.session_state.messages)
            except Exception:
                st.error("Unable to generate a response right now. Please try again.")
                st.stop()

        # 3. Display Gemini response.
        st.markdown(response)

    # 4. Save Gemini response to chat history.
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 5. Run classifier only if feedback was detected and a full
    #    previous exchange was captured.
    if is_feedback and previous_user is not None and previous_assistant is not None:
        with st.spinner("Analyzing response behavior..."):
            try:
                result = classify_sycophancy(
                    question=previous_user,
                    initial_answer=previous_assistant,
                    user_feedback=user_input,
                    revised_answer=response,
                )
                st.session_state.analyses.append(result)
                st.rerun()
            except Exception:
                st.error("Behavioral analysis failed for this exchange.")

    elif is_feedback:
        st.info(
            "Feedback detected, but there isn't a complete previous "
            "exchange to analyze."
        )


def render_chat_history() -> None:
    if not st.session_state.messages:
        st.info(
            "Chat with the assistant normally. When you challenge or "
            "correct a previous response, the system analyzes how the "
            "assistant responds."
        )
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_live_chat(gemini: GeminiChat) -> None:
    render_chat_history()
    render_behavior_analysis()

    user_input = st.chat_input("Type your message...")
    if user_input:
        handle_user_message(gemini, user_input)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    inject_custom_css()
    initialize_session_state()

    mode = render_sidebar()
    render_header(mode)

    if mode == MODE_EVALUATION:
        render_evaluation_mode()
        st.stop()

    try:
        gemini = load_gemini()
    except Exception:
        st.error("Unable to initialize the Gemini API.")
        st.stop()

    handle_live_chat(gemini)


if __name__ == "__main__":
    main()