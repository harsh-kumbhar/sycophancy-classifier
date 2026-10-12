"""
STEP 5 — HYBRID BACKEND EVALUATION

Evaluates the production backend using:

    DeBERTa
        +
    deterministic evidence extraction
        +
    LLM verification
        +
    decision engine

The first stage focuses on the known DeBERTa errors from
the held-out test set.

This evaluation does NOT retrain or modify DeBERTa.

IMPORTANT:
Gemini API calls are deliberately rate-limited and retried so
that the complete set of known errors can be evaluated without
losing the run because of API quota limits.
"""

import sys
import time
from pathlib import Path

import pandas as pd


# ============================================================
# RATE LIMIT CONFIGURATION
# ============================================================

# Gemini free-tier limit observed during evaluation:
# approximately 15 requests/minute.
#
# 6 seconds between successful requests keeps us safely below
# that limit.
REQUEST_INTERVAL_SECONDS = 6.0

# If Gemini still returns a 429, wait longer before retrying.
RATE_LIMIT_RETRY_SECONDS = 25.0

# Maximum number of retries for a single Gemini call.
MAX_RETRIES = 3


# ============================================================
# PATH CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from classifier import classify_sycophancy
from analysis import extract_evidence
from verifier import LLMVerifier
from decision_engine import DecisionEngine


# ============================================================
# FILES
# ============================================================

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "test.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "hybrid_error_evaluation.csv"
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def separator():
    print("=" * 70)


def section(title):
    print()
    separator()
    print(title)
    separator()


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    section(
        "LOADING HELD-OUT TEST DATA"
    )

    print(
        f"Test file: {TEST_FILE}"
    )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test file not found:\n{TEST_FILE}"
        )

    df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"Test samples: {len(df)}"
    )

    print()
    print("Columns:")
    print(
        list(df.columns)
    )

    print()
    print("Label distribution:")
    print(
        df["label"].value_counts()
    )

    return df


# ============================================================
# EXTRACT CONVERSATION FIELDS
# ============================================================

def extract_conversation_fields(row):
    """
    Extract the four logical components from the dataset's
    conversation column.

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

    conversation = str(
        row["conversation"]
    )

    question = ""
    initial_answer = ""
    user_feedback = ""
    revised_answer = ""

    # --------------------------------------------------------
    # Markers
    # --------------------------------------------------------

    initial_marker = (
        "ASSISTANT (INITIAL RESPONSE):"
    )

    feedback_marker = (
        "USER (FOLLOW-UP / FEEDBACK):"
    )

    revised_marker = (
        "ASSISTANT (REVISED RESPONSE):"
    )

    # --------------------------------------------------------
    # Initial answer
    # --------------------------------------------------------

    if initial_marker in conversation:

        before_initial, after_initial = (
            conversation.split(
                initial_marker,
                1,
            )
        )

        question = before_initial

        # ----------------------------------------------------
        # User feedback
        # ----------------------------------------------------

        if feedback_marker in after_initial:

            initial_answer, after_feedback = (
                after_initial.split(
                    feedback_marker,
                    1,
                )
            )

            # ------------------------------------------------
            # Revised answer
            # ------------------------------------------------

            if revised_marker in after_feedback:

                user_feedback, revised_answer = (
                    after_feedback.split(
                        revised_marker,
                        1,
                    )
                )

            else:

                user_feedback = after_feedback

        else:

            initial_answer = after_initial

    return {
        "question": question.strip(),
        "initial_answer": initial_answer.strip(),
        "user_feedback": user_feedback.strip(),
        "revised_answer": revised_answer.strip(),
    }


# ============================================================
# VALIDATE FIELDS
# ============================================================

def validate_fields(fields):

    required = [
        "question",
        "initial_answer",
        "user_feedback",
        "revised_answer",
    ]

    missing = []

    for field in required:

        if not fields.get(field):
            missing.append(field)

    return missing


# ============================================================
# RUN PRODUCTION CLASSIFIER
# ============================================================

def run_classifier(fields):
    """
    Use the exact same production interface used by the
    Streamlit application.
    """

    return classify_sycophancy(
        question=fields[
            "question"
        ],

        initial_answer=fields[
            "initial_answer"
        ],

        user_feedback=fields[
            "user_feedback"
        ],

        revised_answer=fields[
            "revised_answer"
        ],
    )


# ============================================================
# FIND DEBERTA ERRORS
# ============================================================

def find_deberta_errors(df):

    section(
        "IDENTIFYING DEBERTA BASELINE ERRORS"
    )

    errors = []

    processed = 0
    skipped = 0

    for index, row in df.iterrows():

        fields = extract_conversation_fields(
            row
        )

        missing = validate_fields(
            fields
        )

        if missing:

            skipped += 1

            print(
                f"Skipping row {index}: "
                f"missing {missing}"
            )

            continue

        try:

            result = run_classifier(
                fields
            )

            prediction = result[
                "prediction"
            ]

            true_label = str(
                row["label"]
            )

            processed += 1

            if prediction != true_label:

                errors.append(
                    {
                        "index": index,

                        "sample_id": row.get(
                            "sample_id",
                            index,
                        ),

                        "domain": row.get(
                            "domain",
                            "",
                        ),

                        "true_label": true_label,

                        "deberta_prediction": (
                            prediction
                        ),

                        "deberta_result": (
                            result
                        ),

                        "fields": fields,

                        "conversation": row[
                            "conversation"
                        ],
                    }
                )

        except Exception as exc:

            print()
            print(
                f"Classifier error at row "
                f"{index}:"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

    print()

    print(
        f"Rows processed : {processed}"
    )

    print(
        f"Rows skipped   : {skipped}"
    )

    print(
        f"DeBERTa errors : {len(errors)}"
    )

    print()

    if errors:

        error_counts = pd.Series(
            [
                item[
                    "true_label"
                ]
                for item in errors
            ]
        ).value_counts()

        print(
            "Errors by true label:"
        )

        print(
            error_counts
        )

        print()

        prediction_counts = pd.Series(
            [
                item[
                    "deberta_prediction"
                ]
                for item in errors
            ]
        ).value_counts()

        print(
            "Errors by prediction:"
        )

        print(
            prediction_counts
        )

    return errors


# ============================================================
# RUN HYBRID ON ONE ERROR
# ============================================================

def evaluate_error(
    error,
    verifier,
    engine,
):

    fields = error[
        "fields"
    ]

    # --------------------------------------------------------
    # STEP 1
    # Deterministic evidence
    # --------------------------------------------------------

    evidence = extract_evidence(
        question=fields[
            "question"
        ],

        initial_answer=fields[
            "initial_answer"
        ],

        user_feedback=fields[
            "user_feedback"
        ],

        revised_answer=fields[
            "revised_answer"
        ],
    )

    # --------------------------------------------------------
    # STEP 2
    # LLM verification
    # --------------------------------------------------------

    verification = verifier.verify(
        question=fields[
            "question"
        ],

        initial_answer=fields[
            "initial_answer"
        ],

        user_feedback=fields[
            "user_feedback"
        ],

        revised_answer=fields[
            "revised_answer"
        ],
    )

    # --------------------------------------------------------
    # STEP 3
    # Decision engine
    # --------------------------------------------------------

    final_result = engine.decide(
        classifier_result=error[
            "deberta_result"
        ],

        evidence=evidence,

        verification=verification,
    )

    return {
        "evidence": evidence,
        "verification": verification,
        "final": final_result,
    }


# ============================================================
# RATE-LIMIT SAFE EVALUATION
# ============================================================

def evaluate_error_with_retry(
    error,
    verifier,
    engine,
    number,
    total,
    last_call_time,
):
    """
    Evaluate one error while respecting Gemini API limits.

    Returns:

        result
        updated_last_call_time
    """

    # --------------------------------------------------------
    # Enforce spacing between Gemini requests
    # --------------------------------------------------------

    if last_call_time is not None:

        elapsed = (
            time.time()
            - last_call_time
        )

        remaining = (
            REQUEST_INTERVAL_SECONDS
            - elapsed
        )

        if remaining > 0:

            print(
                f"Waiting {remaining:.1f}s "
                f"before Gemini request..."
            )

            time.sleep(
                remaining
            )

    # --------------------------------------------------------
    # Retry loop
    # --------------------------------------------------------

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        try:

            print(
                f"Gemini request "
                f"{number}/{total} "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            request_start = time.time()

            result = evaluate_error(
                error=error,
                verifier=verifier,
                engine=engine,
            )

            return (
                result,
                request_start,
            )

        except Exception as exc:

            error_text = str(
                exc
            ).upper()

            is_rate_limit = (
                "429" in error_text
                or
                "RESOURCE_EXHAUSTED"
                in error_text
                or
                "QUOTA" in error_text
            )

            if is_rate_limit:

                if attempt < MAX_RETRIES:

                    print()
                    print(
                        "⚠ Gemini rate limit "
                        "encountered."
                    )

                    print(
                        f"Waiting "
                        f"{RATE_LIMIT_RETRY_SECONDS}s "
                        "before retry..."
                    )

                    time.sleep(
                        RATE_LIMIT_RETRY_SECONDS
                    )

                    continue

            # ------------------------------------------------
            # Other error or final failed retry
            # ------------------------------------------------

            raise


# ============================================================
# PRINT ERROR
# ============================================================

def print_error_analysis(
    error,
    result,
    number,
    total,
):

    print()
    separator()

    print(
        f"ERROR CASE {number}/{total}"
    )

    separator()

    print(
        f"Sample ID : {error['sample_id']}"
    )

    print(
        f"Domain    : {error['domain']}"
    )

    print(
        f"True      : {error['true_label']}"
    )

    print(
        f"DeBERTa   : "
        f"{error['deberta_prediction']}"
    )

    print(
        f"Hybrid    : "
        f"{result['final']['prediction']}"
    )

    print()

    print(
        "Decision source:"
    )

    print(
        result[
            "final"
        ][
            "decision_source"
        ]
    )

    print()

    print(
        "Evidence:"
    )

    evidence = result[
        "evidence"
    ]

    print(
        f"  answer_changed           : "
        f"{evidence.get('answer_changed')}"
    )

    print(
        f"  contains_new_information : "
        f"{evidence.get('contains_new_information')}"
    )

    print(
        f"  contains_correction      : "
        f"{evidence.get('contains_correction')}"
    )

    print()

    print(
        "Verification:"
    )

    verification = result[
        "verification"
    ]

    print(
        f"  feedback_validity         : "
        f"{verification['feedback_validity']}"
    )

    print(
        f"  initial_answer_supported  : "
        f"{verification['initial_answer_supported']}"
    )

    print(
        f"  new_information_present   : "
        f"{verification['new_information_present']}"
    )

    print(
        f"  user_correction_supported : "
        f"{verification['user_correction_supported']}"
    )

    print(
        f"  revised_answer_supported  : "
        f"{verification['revised_answer_supported']}"
    )

    print()

    print(
        "Verifier reason:"
    )

    print(
        verification[
            "reason"
        ]
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    if not results:
        return

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        f"Progress saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "STEP 5 — HYBRID BACKEND EVALUATION"
    )

    print(
        "Pipeline:"
    )

    print(
        "DeBERTa"
    )

    print(
        "    +"
    )

    print(
        "Deterministic Evidence"
    )

    print(
        "    +"
    )

    print(
        "LLM Verification"
    )

    print(
        "    +"
    )

    print(
        "Decision Engine"
    )

    print()

    print(
        "This stage evaluates the known DeBERTa "
        "failure cases from the held-out test set."
    )

    print()

    print(
        f"Gemini request spacing: "
        f"{REQUEST_INTERVAL_SECONDS:.1f}s"
    )

    print(
        f"Maximum retries per case: "
        f"{MAX_RETRIES}"
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    df = load_test_data()

    # --------------------------------------------------------
    # Find DeBERTa errors
    # --------------------------------------------------------

    errors = find_deberta_errors(
        df
    )

    if not errors:

        print()
        print(
            "No DeBERTa errors found."
        )

        return

    # --------------------------------------------------------
    # Initialize hybrid components
    # --------------------------------------------------------

    section(
        "INITIALIZING HYBRID BACKEND"
    )

    print(
        "Initializing LLM verifier..."
    )

    verifier = LLMVerifier()

    print(
        "Initializing decision engine..."
    )

    engine = DecisionEngine()

    print()
    print(
        "✓ Hybrid backend initialized"
    )

    # --------------------------------------------------------
    # Run hybrid evaluation
    # --------------------------------------------------------

    section(
        "RUNNING HYBRID ON KNOWN ERRORS"
    )

    print(
        f"Known DeBERTa errors: "
        f"{len(errors)}"
    )

    print()

    print(
        "Gemini calls required: "
        f"{len(errors)}"
    )

    print()

    results = []

    successful_calls = 0
    failed_calls = 0

    last_call_time = None

    # --------------------------------------------------------
    # Evaluate each known DeBERTa error
    # --------------------------------------------------------

    for number, error in enumerate(
        errors,
        start=1,
    ):

        try:

            result, last_call_time = (
                evaluate_error_with_retry(
                    error=error,

                    verifier=verifier,

                    engine=engine,

                    number=number,

                    total=len(errors),

                    last_call_time=last_call_time,
                )
            )

            successful_calls += 1

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            results.append(
                {
                    "sample_id": error[
                        "sample_id"
                    ],

                    "domain": error[
                        "domain"
                    ],

                    "true_label": error[
                        "true_label"
                    ],

                    "deberta_prediction": error[
                        "deberta_prediction"
                    ],

                    "hybrid_prediction": result[
                        "final"
                    ][
                        "prediction"
                    ],

                    "decision_source": result[
                        "final"
                    ][
                        "decision_source"
                    ],

                    "decision_reason": result[
                        "final"
                    ][
                        "decision_reason"
                    ],

                    "verification_validity": result[
                        "verification"
                    ][
                        "feedback_validity"
                    ],

                    "verification_confidence": result[
                        "verification"
                    ][
                        "confidence"
                    ],

                    "answer_changed": result[
                        "evidence"
                    ].get(
                        "answer_changed"
                    ),

                    "new_information": result[
                        "evidence"
                    ].get(
                        "contains_new_information"
                    ),

                    "contains_correction": result[
                        "evidence"
                    ].get(
                        "contains_correction"
                    ),

                    "initial_supported": result[
                        "verification"
                    ].get(
                        "initial_answer_supported"
                    ),

                    "correction_supported": result[
                        "verification"
                    ].get(
                        "user_correction_supported"
                    ),

                    "revised_supported": result[
                        "verification"
                    ].get(
                        "revised_answer_supported"
                    ),

                    "revision_follows_feedback": result[
                        "verification"
                    ].get(
                        "revision_follows_feedback"
                    ),

                    "verifier_reason": result[
                        "verification"
                    ].get(
                        "reason"
                    ),
                }
            )

            # ------------------------------------------------
            # Print case
            # ------------------------------------------------

            print_error_analysis(
                error=error,

                result=result,

                number=number,

                total=len(errors),
            )

            # ------------------------------------------------
            # Save immediately after every successful call
            #
            # This prevents losing progress if the script is
            # interrupted later.
            # ------------------------------------------------

            save_results(
                results
            )

        except Exception as exc:

            failed_calls += 1

            print()
            separator()

            print(
                f"ERROR PROCESSING CASE "
                f"{number}/{len(errors)}"
            )

            print(
                f"Sample: "
                f"{error['sample_id']}"
            )

            print(
                f"{type(exc).__name__}: {exc}"
            )

            print()

            print(
                "This case was NOT included "
                "in the quantitative recovery metrics."
            )

            separator()

    # --------------------------------------------------------
    # No successful results
    # --------------------------------------------------------

    if not results:

        print()
        print(
            "No hybrid results were produced."
        )

        return

    # --------------------------------------------------------
    # Final DataFrame
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    # ========================================================
    # EVALUATION COVERAGE
    # ========================================================

    section(
        "EVALUATION COVERAGE"
    )

    print(
        f"Known DeBERTa errors : "
        f"{len(errors)}"
    )

    print(
        f"Successfully verified: "
        f"{successful_calls}"
    )

    print(
        f"Failed API calls     : "
        f"{failed_calls}"
    )

    print(
        f"Coverage             : "
        f"{successful_calls / len(errors) * 100:.2f}%"
    )

    # ========================================================
    # ERROR RECOVERY
    # ========================================================

    section(
        "HYBRID ERROR RECOVERY"
    )

    total_evaluated = len(
        result_df
    )

    corrected = (
        (
            result_df[
                "hybrid_prediction"
            ]
            ==
            result_df[
                "true_label"
            ]
        )
        &
        (
            result_df[
                "deberta_prediction"
            ]
            !=
            result_df[
                "true_label"
            ]
        )
    ).sum()

    remaining = (
        result_df[
            "hybrid_prediction"
        ]
        !=
        result_df[
            "true_label"
        ]
    ).sum()

    print(
        f"Original DeBERTa errors evaluated : "
        f"{total_evaluated}"
    )

    print(
        f"Errors corrected                   : "
        f"{corrected}"
    )

    print(
        f"Errors remaining                   : "
        f"{remaining}"
    )

    print(
        f"Recovery rate                      : "
        f"{corrected / total_evaluated * 100:.2f}%"
    )

    # ========================================================
    # OVERRIDE ANALYSIS
    # ========================================================

    section(
        "OVERRIDE ANALYSIS"
    )

    overrides = result_df[
        result_df[
            "hybrid_prediction"
        ]
        !=
        result_df[
            "deberta_prediction"
        ]
    ]

    correct_overrides = (
        (
            overrides[
                "hybrid_prediction"
            ]
            ==
            overrides[
                "true_label"
            ]
        )
    ).sum()

    harmful_overrides = (
        (
            overrides[
                "hybrid_prediction"
            ]
            !=
            overrides[
                "true_label"
            ]
        )
    ).sum()

    print(
        f"Total overrides   : "
        f"{len(overrides)}"
    )

    print(
        f"Correct overrides  : "
        f"{correct_overrides}"
    )

    print(
        f"Harmful overrides  : "
        f"{harmful_overrides}"
    )

    if len(overrides) > 0:

        print(
            f"Override precision : "
            f"{correct_overrides / len(overrides) * 100:.2f}%"
        )

    # ========================================================
    # SYCOPHANCY RECOVERY
    # ========================================================

    section(
        "SYCOPHANCY RECOVERY"
    )

    syco_errors = result_df[
        (
            result_df[
                "true_label"
            ]
            ==
            "SYCOPHANTIC_SHIFT"
        )
        &
        (
            result_df[
                "deberta_prediction"
            ]
            !=
            "SYCOPHANTIC_SHIFT"
        )
    ]

    syco_recovered = (
        syco_errors[
            "hybrid_prediction"
        ]
        ==
        "SYCOPHANTIC_SHIFT"
    ).sum()

    print(
        f"Original sycophancy errors : "
        f"{len(syco_errors)}"
    )

    print(
        f"Sycophancy recovered       : "
        f"{syco_recovered}"
    )

    if len(syco_errors) > 0:

        print(
            f"Recovery rate              : "
            f"{syco_recovered / len(syco_errors) * 100:.2f}%"
        )

    else:

        print(
            "Recovery rate              : N/A"
        )

    # ========================================================
    # SAVE FINAL RESULTS
    # ========================================================

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()

    print(
        "Detailed results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    section(
        "STEP 5A COMPLETE"
    )

    print(
        f"DeBERTa errors found  : "
        f"{len(errors)}"
    )

    print(
        f"Successfully tested   : "
        f"{successful_calls}"
    )

    print(
        f"API failures         : "
        f"{failed_calls}"
    )

    print(
        f"Corrected by hybrid  : "
        f"{corrected}"
    )

    print(
        f"Harmful overrides    : "
        f"{harmful_overrides}"
    )

    print(
        f"Sycophancy recovered : "
        f"{syco_recovered}"
    )

    print()

    if successful_calls == len(errors):

        print(
            "✓ ALL KNOWN DEBERTA ERRORS "
            "WERE SUCCESSFULLY EVALUATED."
        )

    else:

        print(
            "⚠ Some cases could not be evaluated."
        )

        print(
            "Do NOT treat the recovery percentage "
            "as a full 23-case result."
        )

    print()

    print(
        "Next stage:"
    )

    print(
        "Run the hybrid backend against "
        "controlled and adversarial tests."
    )

    separator()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()