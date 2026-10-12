import sys
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

DATA_PATH = PROJECT_ROOT / "data" / "test.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_PATH = (
    OUTPUT_DIR
    / "production_baseline.json"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "production_baseline_predictions.csv"
)


# ============================================================
# IMPORT PRODUCTION CLASSIFIER
# ============================================================

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from classifier import classify_sycophancy


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "CONSISTENT",
    "JUSTIFIED_CHANGE",
    "SYCOPHANTIC_SHIFT",
]


# ============================================================
# PARSE CONVERSATION
# ============================================================

def parse_conversation(conversation):
    """
    Parse the conversation field used by data/test.csv.

    Expected format:

    USER:
    ...

    ASSISTANT (INITIAL RESPONSE):
    ...

    USER (FOLLOW-UP / FEEDBACK):
    ...

    ASSISTANT (REVISED RESPONSE):
    ...
    """

    text = str(conversation)

    initial_marker = (
        "ASSISTANT (INITIAL RESPONSE):"
    )

    feedback_marker = (
        "USER (FOLLOW-UP / FEEDBACK):"
    )

    revised_marker = (
        "ASSISTANT (REVISED RESPONSE):"
    )

    if initial_marker not in text:
        raise ValueError(
            "Could not find initial response marker."
        )

    if feedback_marker not in text:
        raise ValueError(
            "Could not find feedback marker."
        )

    if revised_marker not in text:
        raise ValueError(
            "Could not find revised response marker."
        )

    # --------------------------------------------------------
    # Question
    # --------------------------------------------------------

    question_part = text.split(
        initial_marker,
        1
    )[0]

    if question_part.startswith("USER:"):
        question = question_part[
            len("USER:")
        ].strip()
    else:
        question = question_part.strip()

    # --------------------------------------------------------
    # Initial answer
    # --------------------------------------------------------

    initial_part = text.split(
        initial_marker,
        1
    )[1]

    initial_answer = initial_part.split(
        feedback_marker,
        1
    )[0].strip()

    # --------------------------------------------------------
    # User feedback
    # --------------------------------------------------------

    feedback_part = text.split(
        feedback_marker,
        1
    )[1]

    user_feedback = feedback_part.split(
        revised_marker,
        1
    )[0].strip()

    # --------------------------------------------------------
    # Revised answer
    # --------------------------------------------------------

    revised_answer = text.split(
        revised_marker,
        1
    )[1].strip()

    return {
        "question": question,
        "initial_answer": initial_answer,
        "user_feedback": user_feedback,
        "revised_answer": revised_answer,
    }


# ============================================================
# LOAD DATASET
# ============================================================

def load_test_dataset():

    print("=" * 70)
    print("LOADING HELD-OUT TEST DATA")
    print("=" * 70)

    print(
        f"Test file: {DATA_PATH}"
    )

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Test dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"\nTest samples: {len(df)}"
    )

    print("\nColumns:")

    print(
        list(df.columns)
    )

    print("\nLabel distribution:")

    print(
        df["label"].value_counts()
    )

    return df


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    required_columns = [
        "conversation",
        "label",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    invalid_labels = (
        set(df["label"].unique())
        - set(LABELS)
    )

    if invalid_labels:

        raise ValueError(
            f"Unexpected labels: {invalid_labels}"
        )

    missing_values = df[
        required_columns
    ].isna().sum()

    if missing_values.any():

        print(
            "\nWARNING: Missing values detected:"
        )

        print(
            missing_values[
                missing_values > 0
            ]
        )


# ============================================================
# RUN PRODUCTION CLASSIFIER
# ============================================================

def evaluate_classifier(df):

    print("\n")
    print("=" * 70)
    print("RUNNING PRODUCTION CLASSIFIER")
    print("=" * 70)

    print(
        "\nUsing:"
    )

    print(
        "  app/classifier.py"
    )

    print(
        "\nParsing conversation fields..."
    )

    predictions = []

    parse_errors = 0

    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        # ----------------------------------------------------
        # Parse conversation
        # ----------------------------------------------------

        try:

            parsed = parse_conversation(
                row["conversation"]
            )

        except Exception as error:

            parse_errors += 1

            print(
                f"\nWARNING: Could not parse "
                f"test sample {position}"
            )

            print(
                f"Error: {error}"
            )

            continue

        # ----------------------------------------------------
        # Production classifier
        # ----------------------------------------------------

        result = classify_sycophancy(
            question=parsed["question"],
            initial_answer=parsed[
                "initial_answer"
            ],
            user_feedback=parsed[
                "user_feedback"
            ],
            revised_answer=parsed[
                "revised_answer"
            ],
        )

        predictions.append(
            {
                "index": int(
                    position - 1
                ),

                "sample_id": row.get(
                    "sample_id",
                    ""
                ),

                "question_id": row.get(
                    "question_id",
                    ""
                ),

                "model": row.get(
                    "model",
                    ""
                ),

                "domain": row.get(
                    "domain",
                    ""
                ),

                "difficulty": row.get(
                    "difficulty",
                    ""
                ),

                "condition": row.get(
                    "condition",
                    ""
                ),

                "true_label": str(
                    row["label"]
                ),

                "predicted_label": result[
                    "prediction"
                ],

                "confidence": float(
                    result["confidence"]
                ),

                "consistent_probability": float(
                    result[
                        "probabilities"
                    ].get(
                        "CONSISTENT",
                        0.0
                    )
                ),

                "justified_change_probability": float(
                    result[
                        "probabilities"
                    ].get(
                        "JUSTIFIED_CHANGE",
                        0.0
                    )
                ),

                "sycophantic_shift_probability": float(
                    result[
                        "probabilities"
                    ].get(
                        "SYCOPHANTIC_SHIFT",
                        0.0
                    )
                ),
            }
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if position % 100 == 0:

            print(
                f"Processed "
                f"{position}/{len(df)}"
            )

    print(
        f"\nSuccessfully classified: "
        f"{len(predictions)}"
    )

    print(
        f"Conversation parse errors: "
        f"{parse_errors}"
    )

    if not predictions:

        raise RuntimeError(
            "No test samples were successfully classified."
        )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(results_df):

    y_true = results_df[
        "true_label"
    ]

    y_pred = results_df[
        "predicted_label"
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )

    sycophantic_f1 = f1_score(
        y_true,
        y_pred,
        labels=[
            "SYCOPHANTIC_SHIFT"
        ],
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    return {
        "accuracy": float(
            accuracy
        ),

        "macro_f1": float(
            macro_f1
        ),

        "weighted_f1": float(
            weighted_f1
        ),

        "sycophantic_f1": float(
            sycophantic_f1
        ),

        "classification_report": report,

        "confusion_matrix": matrix.tolist(),
    }


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results_df,
    metrics,
):

    print("\n")
    print("=" * 70)
    print("PRODUCTION BASELINE RESULTS")
    print("=" * 70)

    print(
        f"\nTest samples       : "
        f"{len(results_df)}"
    )

    print(
        f"Accuracy           : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro F1           : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1        : "
        f"{metrics['weighted_f1']:.4f}"
    )

    print(
        f"Sycophantic F1     : "
        f"{metrics['sycophantic_f1']:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        results_df["true_label"],
        results_df["predicted_label"],
        labels=LABELS,
        target_names=LABELS,
        digits=4,
        zero_division=0,
    )

    print(
        report
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    matrix_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=LABELS,
        columns=LABELS,
    )

    print(
        matrix_df
    )

    # --------------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PREDICTION DISTRIBUTION")
    print("=" * 70)

    print(
        results_df[
            "predicted_label"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    errors = results_df[
        results_df["true_label"]
        !=
        results_df["predicted_label"]
    ]

    print("\n")
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(
        f"\nTotal errors: "
        f"{len(errors)}"
    )

    if len(errors) > 0:

        print(
            "\nErrors by true label:"
        )

        print(
            errors[
                "true_label"
            ].value_counts()
        )

        print(
            "\nErrors by predicted label:"
        )

        print(
            errors[
                "predicted_label"
            ].value_counts()
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results_df,
    metrics,
):

    print("\n")
    print("=" * 70)
    print("SAVING BASELINE RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        f"\nPredictions saved to:"
    )

    print(
        PREDICTIONS_PATH
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    output = {
        "experiment": (
            "production_baseline"
        ),

        "description": (
            "Baseline evaluation of "
            "production classifier.py "
            "against the held-out test set."
        ),

        "dataset": {
            "path": str(
                DATA_PATH
            ),

            "samples": int(
                len(results_df)
            ),
        },

        "metrics": {
            "accuracy": metrics[
                "accuracy"
            ],

            "macro_f1": metrics[
                "macro_f1"
            ],

            "weighted_f1": metrics[
                "weighted_f1"
            ],

            "sycophantic_f1": metrics[
                "sycophantic_f1"
            ],
        },

        "classification_report": metrics[
            "classification_report"
        ],

        "confusion_matrix": {
            "labels": LABELS,

            "matrix": metrics[
                "confusion_matrix"
            ],
        },
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"\nSummary saved to:"
    )

    print(
        RESULTS_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PRODUCTION CLASSIFIER BASELINE EVALUATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_test_dataset()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_dataset(
        df
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    results_df = evaluate_classifier(
        df
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results_df
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_results(
        results_df,
        metrics,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results_df,
        metrics,
    )

    # --------------------------------------------------------
    # Official comparison
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "COMPARISON WITH LOCKED COLAB RESULTS"
    )
    print("=" * 70)

    print(
        "\nOfficial Colab test results:"
    )

    print(
        "  Accuracy        : 0.9829"
    )

    print(
        "  Macro F1        : 0.9609"
    )

    print(
        "  Sycophantic F1  : 0.9616"
    )

    print(
        "\nProduction classifier results:"
    )

    print(
        f"  Accuracy        : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"  Macro F1        : "
        f"{metrics['macro_f1']:.4f}"
    )

    print(
        f"  Sycophantic F1  : "
        f"{metrics['sycophantic_f1']:.4f}"
    )

    print("\n")
    print("=" * 70)
    print(
        "PRODUCTION BASELINE COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()