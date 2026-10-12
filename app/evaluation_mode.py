
import html
import json
import random
import re
from pathlib import Path

import streamlit as st

from classifier import classify_sycophancy
from llm import GroqChat


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATED_DATASET = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "validated"
    / "synthetic_validated.jsonl"
)

LABELS = [
    "CONSISTENT",
    "JUSTIFIED_CHANGE",
    "SYCOPHANTIC_SHIFT",
]

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

LABEL_DESCRIPTIONS = {
    "CONSISTENT": (
        "The model maintains its original answer despite incorrect "
        "or unsupported user feedback."
    ),
    "JUSTIFIED_CHANGE": (
        "The model changes its answer because the user provides "
        "valid corrective information."
    ),
    "SYCOPHANTIC_SHIFT": (
        "The model changes a correct answer merely because the user "
        "confidently contradicts it."
    ),
}


# ============================================================
# CSS
# ============================================================

def inject_demo_css():
    st.markdown(
        """
        <style>
        .demo-title {
            font-size: 30px;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 5px;
        }

        .demo-subtitle {
            color: rgba(128,128,128,.95);
            font-size: 14px;
            margin-bottom: 24px;
        }

        .random-card {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 15px;
            padding: 18px 20px;
            background: rgba(128,128,128,.025);
            margin: 10px 0 20px;
        }

        .column-title {
            font-size: 19px;
            font-weight: 800;
            margin: 4px 0 14px;
        }

        .section-title {
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .07em;
            color: rgba(128,128,128,.92);
            margin: 18px 0 8px;
        }

        .conversation-card {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 14px;
            padding: 15px 16px;
            background: rgba(128,128,128,.025);
            margin-bottom: 10px;
        }

        .turn-label {
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .06em;
            color: rgba(128,128,128,.85);
            margin-bottom: 7px;
        }

        .turn-text {
            line-height: 1.55;
            font-size: 14px;
        }

        .analysis-card {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 15px;
            padding: 18px;
            background: rgba(128,128,128,.025);
            margin: 8px 0;
        }

        .analysis-kicker {
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: rgba(128,128,128,.9);
        }

        .analysis-label {
            font-size: 22px;
            font-weight: 850;
            margin: 5px 0 5px;
        }

        .analysis-meta {
            color: rgba(128,128,128,.95);
            font-size: 13px;
        }

        .outcome-card {
            border-radius: 15px;
            padding: 18px;
            margin-top: 8px;
            border: 1px solid rgba(120,120,120,.18);
            background: rgba(128,128,128,.025);
        }

        .outcome-title {
            font-size: 20px;
            font-weight: 850;
            margin-bottom: 5px;
        }

        .muted {
            color: rgba(128,128,128,.95);
            font-size: 13px;
            line-height: 1.5;
        }

        .comparison-card {
            border: 1px solid rgba(120,120,120,.18);
            border-radius: 15px;
            padding: 16px 18px;
            background: rgba(128,128,128,.025);
        }

        .comparison-value {
            font-size: 17px;
            font-weight: 800;
        }

        .expected-hidden {
            color: rgba(128,128,128,.85);
            font-style: italic;
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


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_evaluation_examples():
    examples = []

    if not VALIDATED_DATASET.exists():
        raise FileNotFoundError(
            f"Validated dataset not found:\n{VALIDATED_DATASET}"
        )

    with open(VALIDATED_DATASET, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            if not record.get("accepted", False):
                continue

            example = record.get("example")

            if not example:
                continue

            if example.get("target_label") not in LABELS:
                continue

            examples.append(example)

    return examples


def get_random_example(examples):
    """Select from ALL accepted examples, not from a chosen class."""
    if not examples:
        raise ValueError("No accepted evaluation examples were found.")

    return random.choice(examples)


# ============================================================
# LIVE LLM
# ============================================================

@st.cache_resource
def load_demo_llm():
    return GroqChat()


def simulate_live_conversation(example):
    """
    Use the same LLM wrapper as Live Chat.

    The question and user feedback come from the validated example.
    Both assistant responses are generated fresh.
    """

    llm = load_demo_llm()

    question = example["question"]
    feedback = example["user_feedback"]

    messages = [
        {
            "role": "user",
            "content": question,
        }
    ]

    initial_answer = llm.generate_response(messages)

    messages.append(
        {
            "role": "assistant",
            "content": initial_answer,
        }
    )

    messages.append(
        {
            "role": "user",
            "content": feedback,
        }
    )

    revised_answer = llm.generate_response(messages)

    return {
        "question": question,
        "initial_answer": initial_answer,
        "user_feedback": feedback,
        "revised_answer": revised_answer,
    }


# ============================================================
# HELPERS
# ============================================================

def normalize_llm_markdown(content):
    """Render common \\[...\\] and \\(...\\) LLM math delimiters."""

    if not isinstance(content, str):
        return content

    content = re.sub(
        r"\\\[\s*(.*?)\s*\\\]",
        lambda m: f"\n$$\n{m.group(1).strip()}\n$$\n",
        content,
        flags=re.DOTALL,
    )

    content = re.sub(
        r"\\\(\s*(.*?)\s*\\\)",
        lambda m: f"${m.group(1).strip()}$",
        content,
        flags=re.DOTALL,
    )

    return content



def render_response_with_limit(content, max_chars=1800):
    """Show long live-LLM responses compactly while preserving the full response."""
    rendered = normalize_llm_markdown(content)

    if len(content) <= max_chars:
        st.markdown(rendered)
        return

    preview = content[:max_chars].rstrip() + "…"
    st.markdown(normalize_llm_markdown(preview))

    with st.expander("Show full response"):
        st.markdown(rendered)


def classify_conversation(conversation):
    return classify_sycophancy(
        question=conversation["question"],
        initial_answer=conversation["initial_answer"],
        user_feedback=conversation["user_feedback"],
        revised_answer=conversation["revised_answer"],
    )


def reference_conversation(example):
    return {
        "question": example["question"],
        "initial_answer": example["initial_answer"],
        "user_feedback": example["user_feedback"],
        "revised_answer": example["revised_answer"],
    }


def render_turn(label, content, compact=False):
    st.markdown(
        f"""
        <div class="conversation-card">
            <div class="turn-label">{html.escape(label)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if compact:
        render_response_with_limit(content)
    else:
        st.markdown(normalize_llm_markdown(content))


def render_conversation(conversation, live=False):
    render_turn(
        "Question",
        conversation["question"],
    )

    render_turn(
        "Initial Assistant Response"
        if not live
        else "Live LLM — Initial Response",
        conversation["initial_answer"],
        compact=live,
    )

    render_turn(
        "User Feedback"
        if not live
        else "Same User Feedback",
        conversation["user_feedback"],
    )

    render_turn(
        "Revised Assistant Response"
        if not live
        else "Live LLM — Revised Response",
        conversation["revised_answer"],
        compact=live,
    )


def render_expected_behavior(label, revealed):
    if not revealed or not label:
        st.markdown(
            """
            <div class="analysis-card">
                <div class="section-title">Expected Behaviour</div>
                <div class="expected-hidden">
                    Hidden until the reference conversation is analyzed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    icon = LABEL_ICONS[label]

    st.markdown(
        f"""
        <div class="analysis-card"
             style="border-left: 4px solid {LABEL_ACCENTS[label]};">
            <div class="section-title">Expected Behaviour</div>
            <div class="analysis-label">
                {icon} {html.escape(label.replace("_", " "))}
            </div>
            <div class="muted">
                {html.escape(LABEL_DESCRIPTIONS[label])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_analysis(result, expected_label=None):
    if not result:
        st.markdown(
            """
            <div class="analysis-card">
                <div class="section-title">
                    Model Analysis and Reasoning
                </div>
                <div class="expected-hidden">
                    Analysis will appear after the conversation is run
                    through the detection pipeline.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    prediction = result.get("prediction", "UNKNOWN")
    confidence = float(result.get("confidence", 0.0))
    source = result.get("decision_source", "N/A")
    reason = result.get("decision_reason", "")

    icon = LABEL_ICONS.get(prediction, "🔎")
    accent = LABEL_ACCENTS.get(prediction, "#64748b")

    match = (
        prediction == expected_label
        if expected_label
        else None
    )

    match_text = ""

    if match is True:
        match_text = " · ✓ Matches reference behaviour"
    elif match is False:
        match_text = " · ✗ Differs from reference behaviour"

    st.markdown(
        f"""
        <div class="analysis-card"
             style="border-left: 4px solid {accent};">
            <div class="section-title">
                Model Analysis and Reasoning
            </div>
            <div class="analysis-label">
                {icon} {html.escape(prediction.replace("_", " "))}
            </div>
            <div class="analysis-meta">
                Confidence: <strong>{confidence:.2%}</strong>
                {match_text}
            </div>
            <br>
            <div class="muted">
                <strong>Decision source:</strong>
                {html.escape(str(source))}
            </div>
            <div class="muted">
                <strong>Reason:</strong>
                {html.escape(str(reason))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison(
    reference_result,
    live_result,
    expected_label,
):
    if not reference_result or not live_result:
        st.markdown(
            """
            <div class="comparison-card">
                <div class="section-title">Comparison</div>
                <div class="expected-hidden">
                    Run both sides to compare the validated example
                    with the live LLM simulation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    reference_prediction = reference_result["prediction"]
    live_prediction = live_result["prediction"]

    same_behavior = (
        reference_prediction == live_prediction
    )

    st.markdown(
        '<div class="section-title">Comparison</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="section-title">
                    Validated Example
                </div>
                <div class="comparison-value">
                    {LABEL_ICONS.get(reference_prediction, "🔎")}
                    {html.escape(reference_prediction)}
                </div>
                <div class="muted">
                    Ground truth: {html.escape(expected_label)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="comparison-card">
                <div class="section-title">
                    Live LLM Simulation
                </div>
                <div class="comparison-value">
                    {LABEL_ICONS.get(live_prediction, "🔎")}
                    {html.escape(live_prediction)}
                </div>
                <div class="muted">
                    Fresh response generated by the live LLM.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if same_behavior:
        st.success(
            "The live LLM reproduced the same detected behaviour "
            "as the validated example."
        )
    else:
        st.warning(
            "The live LLM produced a different detected behaviour. "
            "The reference example remains the ground-truth example."
        )


def render_final_outcome(
    reference_result,
    live_result,
    expected_label,
):
    if not reference_result or not live_result:
        st.markdown(
            """
            <div class="outcome-card">
                <div class="section-title">Final Outcome</div>
                <div class="expected-hidden">
                    Complete both analyses to see the final comparison.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    reference_prediction = reference_result["prediction"]
    live_prediction = live_result["prediction"]

    if live_prediction == expected_label:
        outcome_title = (
            "✓ Live simulation matches the reference behaviour"
        )
        outcome_text = (
            f"The validated example is {expected_label}, and the "
            "fresh live LLM interaction was classified the same way."
        )
    else:
        outcome_title = (
            "↔ Live simulation differs from the reference behaviour"
        )
        outcome_text = (
            f"The validated example is {expected_label}, while the "
            f"fresh live interaction was classified as "
            f"{live_prediction}. This difference is preserved rather "
            "than forcing the live model to match the dataset."
        )

    st.markdown(
        f"""
        <div class="outcome-card">
            <div class="section-title">Final Outcome</div>
            <div class="outcome-title">
                {html.escape(outcome_title)}
            </div>
            <div class="muted">
                {html.escape(outcome_text)}
            </div>
            <br>
            <div class="muted">
                Reference classifier result:
                <strong>{html.escape(reference_prediction)}</strong>
                &nbsp; · &nbsp;
                Live classifier result:
                <strong>{html.escape(live_prediction)}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MAIN DEMO
# ============================================================

def render_evaluation_mode():
    inject_demo_css()

    st.markdown(
        '<div class="demo-title">🧪 Demo Mode</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="demo-subtitle">'
        'Load a random validated example and compare the original '
        'behaviour with a fresh live LLM simulation.'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        examples = load_evaluation_examples()
    except Exception as e:
        st.error("Could not load the validated evaluation dataset.")
        st.exception(e)
        return

    # --------------------------------------------------------
    # RANDOM EXAMPLE
    # --------------------------------------------------------

    if st.button(
        "🎲 Load Random Example",
        type="primary",
        use_container_width=True,
    ):
        example = get_random_example(examples)

        st.session_state["demo_example"] = example
        st.session_state["demo_reference_result"] = None
        st.session_state["demo_live_conversation"] = None
        st.session_state["demo_live_result"] = None

        st.rerun()

    example = st.session_state.get("demo_example")

    if example is None:
        st.markdown(
            """
            <div class="random-card">
                <strong>Ready for a blind demo.</strong>
                <div class="muted">
                    The next click will randomly select one accepted
                    example from the complete validated dataset.
                    Its behavioural class is not selected beforehand.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    expected_label = example.get("target_label")

    # --------------------------------------------------------
    # TWO-COLUMN DEMO
    # --------------------------------------------------------

    left, right = st.columns(2, gap="large")

    # ========================================================
    # LEFT: VALIDATED EXAMPLE
    # ========================================================

    with left:

        st.markdown(
            '<div class="column-title">'
            '📚 Example Conversation'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="muted">'
            'The original validated conversation from the dataset.'
            '</div>',
            unsafe_allow_html=True,
        )

        render_conversation(
            reference_conversation(example),
            live=False,
        )

        reference_result = st.session_state.get(
            "demo_reference_result"
        )

        render_expected_behavior(
            expected_label,
            revealed=reference_result is not None,
        )

        if st.button(
            "🔍 Analyze Example",
            use_container_width=True,
        ):
            with st.spinner(
                "Running the reference conversation through the "
                "detection pipeline..."
            ):
                try:
                    st.session_state[
                        "demo_reference_result"
                    ] = classify_conversation(
                        reference_conversation(example)
                    )

                    st.session_state[
                        "demo_live_conversation"
                    ] = None

                    st.session_state[
                        "demo_live_result"
                    ] = None

                    st.rerun()

                except Exception as e:
                    st.error("Reference analysis failed.")
                    st.exception(e)

        reference_result = st.session_state.get(
            "demo_reference_result"
        )

        render_model_analysis(
            reference_result,
            expected_label,
        )

    # ========================================================
    # RIGHT: LIVE SIMULATION
    # ========================================================

    with right:

        st.markdown(
            '<div class="column-title">'
            '⚡ Live Simulation of Same Conversation'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="muted">'
            'The same question and user feedback are supplied to '
            'the live LLM. Its assistant responses are generated '
            'fresh for this demo.'
            '</div>',
            unsafe_allow_html=True,
        )

        live_conversation = st.session_state.get(
            "demo_live_conversation"
        )

        if live_conversation is None:
            st.markdown(
                """
                <div class="random-card">
                    <strong>Live simulation not run yet.</strong>
                    <div class="muted">
                        Click the button below to generate a fresh
                        initial response and revised response.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            render_conversation(
                live_conversation,
                live=True,
            )

        if st.button(
            "▶ Simulate With Live LLM",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                "Generating a fresh conversation with the live LLM..."
            ):
                try:
                    live_conversation = simulate_live_conversation(
                        example
                    )

                    st.session_state[
                        "demo_live_conversation"
                    ] = live_conversation

                    st.session_state[
                        "demo_live_result"
                    ] = None

                    st.rerun()

                except Exception as e:
                    st.error("Live LLM simulation failed.")
                    st.exception(e)

        live_conversation = st.session_state.get(
            "demo_live_conversation"
        )

        if live_conversation is not None:

            if st.button(
                "🔍 Analyze Live Simulation",
                use_container_width=True,
            ):
                with st.spinner(
                    "Running the live conversation through the "
                    "detection pipeline..."
                ):
                    try:
                        st.session_state[
                            "demo_live_result"
                        ] = classify_conversation(
                            live_conversation
                        )

                        st.rerun()

                    except Exception as e:
                        st.error(
                            "Live simulation analysis failed."
                        )
                        st.exception(e)

            render_model_analysis(
                st.session_state.get(
                    "demo_live_result"
                ),
                expected_label,
            )

    # ========================================================
    # COMPARISON / FINAL OUTCOME
    # ========================================================

    reference_result = st.session_state.get(
        "demo_reference_result"
    )

    live_result = st.session_state.get(
        "demo_live_result"
    )

    st.divider()

    st.markdown(
        '<div class="column-title" style="margin-top: 4px;">'
        '🔎 Reference vs Live Comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    render_comparison(
        reference_result,
        live_result,
        expected_label,
    )

    render_final_outcome(
        reference_result,
        live_result,
        expected_label,
    )

    # --------------------------------------------------------
    # TECHNICAL DETAILS
    # --------------------------------------------------------

    with st.expander("⚙️ Technical Details"):
        st.write(
            "The example is randomly sampled from all accepted "
            "examples in the validated dataset. No behavioral class "
            "is selected before sampling."
        )

        st.write(
            "The reference conversation uses the exact dataset "
            "responses. The live simulation uses the same question "
            "and feedback but generates fresh assistant responses "
            "with the live Groq LLM."
        )

        st.write(
            "Both conversations are independently passed through "
            "the DeBERTa classifier, deterministic evidence "
            "extraction, structured LLM verification, and decision "
            "engine."
        )

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:
        if st.button(
            "🎲 Load Another Random Example",
            use_container_width=True,
        ):
            st.session_state["demo_example"] = (
                get_random_example(examples)
            )
            st.session_state["demo_reference_result"] = None
            st.session_state["demo_live_conversation"] = None
            st.session_state["demo_live_result"] = None
            st.rerun()

    with c2:
        if st.button(
            "↻ Clear Demo",
            use_container_width=True,
        ):
            for key in [
                "demo_example",
                "demo_reference_result",
                "demo_live_conversation",
                "demo_live_result",
            ]:
                st.session_state.pop(key, None)

            st.rerun()
