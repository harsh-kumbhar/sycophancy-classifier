import json
import random
from pathlib import Path

import streamlit as st

from classifier import classify_sycophancy


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


LABEL_DESCRIPTIONS = {
    "CONSISTENT": (
        "The model maintains its original answer "
        "despite incorrect or unsupported user feedback."
    ),

    "JUSTIFIED_CHANGE": (
        "The model changes its answer because the "
        "user provides valid corrective information."
    ),

    "SYCOPHANTIC_SHIFT": (
        "The model changes a correct answer merely "
        "because the user confidently contradicts it."
    ),
}


LABEL_ICONS = {
    "CONSISTENT": "🟢",
    "JUSTIFIED_CHANGE": "🟡",
    "SYCOPHANTIC_SHIFT": "🔴",
}


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_evaluation_examples():
    """
    Load accepted synthetic examples from the validated
    dataset and organize them by target label.
    """

    if not VALIDATED_DATASET.exists():

        raise FileNotFoundError(
            f"Validated dataset not found:\n"
            f"{VALIDATED_DATASET}"
        )

    examples_by_label = {
        label: []
        for label in LABELS
    }

    with open(
        VALIDATED_DATASET,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            # Only use examples that passed validation.
            if not record.get("accepted", False):
                continue

            example = record.get("example")

            if not example:
                continue

            label = example.get("target_label")

            if label not in LABELS:
                continue

            examples_by_label[label].append(
                example
            )

    return examples_by_label


# ============================================================
# SINGLE EXAMPLE
# ============================================================

def get_random_example(
    examples_by_label,
    label
):
    """
    Return one random example for the requested label.
    """

    examples = examples_by_label.get(
        label,
        []
    )

    if not examples:

        raise ValueError(
            f"No accepted examples found "
            f"for label: {label}"
        )

    return random.choice(examples)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_demo_example(example):
    """
    Run the real DeBERTa classifier on one controlled
    evaluation example.
    """

    result = classify_sycophancy(
        question=example["question"],
        initial_answer=example["initial_answer"],
        user_feedback=example["user_feedback"],
        revised_answer=example["revised_answer"],
    )

    return result


# ============================================================
# RESULT DISPLAY
# ============================================================

def display_classification_result(
    result,
    expected_label
):
    """
    Display the classifier prediction and compare it
    against the expected synthetic label.
    """

    prediction = result["prediction"]

    confidence = result["confidence"]

    probabilities = result["probabilities"]

    correct = (
        prediction == expected_label
    )

    st.divider()

    st.subheader(
        "DeBERTa Classification"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if correct:

        st.success(
            f"✓ Prediction matches expected label: "
            f"{prediction}"
        )

    else:

        st.error(
            f"✗ Prediction: {prediction} "
            f"| Expected: {expected_label}"
        )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    st.markdown(
        f"### {LABEL_ICONS.get(prediction, '🔎')} "
        f"{prediction}"
    )

    st.progress(
        float(confidence),
        text=f"Confidence: {confidence:.2%}"
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    st.markdown(
        "**Class probabilities**"
    )

    for label in LABELS:

        probability = probabilities.get(
            label,
            0.0
        )

        st.write(
            f"**{label}**: "
            f"{probability:.2%}"
        )

    return correct


# ============================================================
# MAIN EVALUATION UI
# ============================================================

def render_evaluation_mode():
    """
    Render the controlled Evaluation / Demo interface.
    """

    st.header(
        "🧪 Evaluation / Demo Mode"
    )

    st.caption(
        "Run controlled examples through the trained "
        "DeBERTa classifier."
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    try:

        examples_by_label = (
            load_evaluation_examples()
        )

    except Exception as e:

        st.error(
            "Could not load the evaluation dataset."
        )

        st.exception(e)

        return

    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    st.markdown(
        "### Validated Evaluation Dataset"
    )

    cols = st.columns(3)

    for index, label in enumerate(LABELS):

        with cols[index]:

            st.metric(
                label.replace(
                    "_",
                    " "
                ),
                len(
                    examples_by_label[label]
                )
            )

    st.divider()

    # --------------------------------------------------------
    # Label selection
    # --------------------------------------------------------

    selected_label = st.selectbox(
        "Select behavior to demonstrate",
        LABELS,
        format_func=lambda label: (
            f"{LABEL_ICONS[label]} "
            f"{label.replace('_', ' ')}"
        ),
    )

    st.info(
        LABEL_DESCRIPTIONS[
            selected_label
        ]
    )

    # --------------------------------------------------------
    # Generate example
    # --------------------------------------------------------

    if st.button(
        "🎲 Load Random Example",
        use_container_width=True
    ):

        example = get_random_example(
            examples_by_label,
            selected_label
        )

        st.session_state[
            "evaluation_example"
        ] = example

        st.session_state[
            "evaluation_expected_label"
        ] = selected_label

        st.session_state[
            "evaluation_result"
        ] = None

    # --------------------------------------------------------
    # Current example
    # --------------------------------------------------------

    example = st.session_state.get(
        "evaluation_example"
    )

    expected_label = st.session_state.get(
        "evaluation_expected_label"
    )

    if example is None:

        st.info(
            "Select a behavior and click "
            "'Load Random Example' to begin."
        )

        return

    # --------------------------------------------------------
    # Display controlled conversation
    # --------------------------------------------------------

    st.markdown(
        "### Controlled Test Case"
    )

    st.markdown(
        "**Question**"
    )

    st.info(
        example["question"]
    )

    st.markdown(
        "**Initial Assistant Response**"
    )

    st.write(
        example["initial_answer"]
    )

    st.markdown(
        "**User Feedback**"
    )

    st.warning(
        example["user_feedback"]
    )

    st.markdown(
        "**Revised Assistant Response**"
    )

    st.write(
        example["revised_answer"]
    )

    # --------------------------------------------------------
    # Expected label
    # --------------------------------------------------------

    st.markdown(
        f"**Expected behavior:** "
        f"{LABEL_ICONS[expected_label]} "
        f"{expected_label}"
    )

    # --------------------------------------------------------
    # Run classifier
    # --------------------------------------------------------

    if st.button(
        "🔍 Analyze With DeBERTa",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Running DeBERTa classifier..."
        ):

            try:

                result = classify_demo_example(
                    example
                )

                st.session_state[
                    "evaluation_result"
                ] = result

            except Exception as e:

                st.error(
                    "Classification failed."
                )

                st.exception(e)

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    result = st.session_state.get(
        "evaluation_result"
    )

    if result is not None:

        display_classification_result(
            result,
            expected_label
        )