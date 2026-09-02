# ============================================================
# SYNTHETIC DATA GENERATOR — GOOGLE GEMINI
# ============================================================

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


from dotenv import load_dotenv
from google import genai
from google.genai import types

from synthetic_config import (
    LABELS,
    DOMAINS,
    DIFFICULTY_LEVELS,
    SYNTHETIC_SAMPLES_PER_CLASS,
    BATCH_SIZE,
    MAX_BATCH_ATTEMPTS,
    RAW_DATA_PATH,
    METADATA_PATH,
    PROMPT_VERSION,
)

from prompts import (
    SYSTEM_PROMPT,
    build_generation_prompt,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found.\n"
        "Create a .env file in the project root containing:\n"
        "GEMINI_API_KEY=your_api_key"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# GENERATOR MODEL
# ============================================================

GENERATOR_MODEL = "gemini-3.5-flash-lite"

# ============================================================
# GENERATION PARAMETERS
# ============================================================

TEMPERATURE = 0.8

MAX_OUTPUT_TOKENS = 12000


# ============================================================
# REQUIRED JSON FIELDS
# ============================================================

REQUIRED_FIELDS = {
    "domain",
    "difficulty",
    "target_label",
    "question",
    "initial_answer",
    "user_feedback",
    "revised_answer",
}

SYNTHETIC_EXAMPLE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "domain": {
                "type": "STRING"
            },
            "difficulty": {
                "type": "STRING"
            },
            "target_label": {
                "type": "STRING"
            },
            "question": {
                "type": "STRING"
            },
            "initial_answer": {
                "type": "STRING"
            },
            "user_feedback": {
                "type": "STRING"
            },
            "revised_answer": {
                "type": "STRING"
            },
        },
        "required": [
            "domain",
            "difficulty",
            "target_label",
            "question",
            "initial_answer",
            "user_feedback",
            "revised_answer",
        ],
    },
}
# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text: str):
    """
    Parse Gemini's JSON response.

    Gemini is instructed to return JSON only, but this function
    performs defensive handling of accidental markdown fences.
    """

    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

        if text.startswith("json"):
            text = text[4:].strip()

    try:
        data = json.loads(text)

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Gemini returned invalid JSON.\n"
            f"Response preview:\n{text[:1500]}"
        ) from exc

    if not isinstance(data, list):

        raise ValueError(
            "Gemini response must be a JSON array."
        )

    return data


# ============================================================
# BASIC STRUCTURAL CHECK
# ============================================================

def check_batch_structure(
    examples,
    expected_count,
    expected_label,
    expected_domain,
    expected_difficulty,
):
    """
    Perform basic structural checks.

    This is NOT the full behavioral validator.
    Full validation happens in validate.py.
    """

    if len(examples) != expected_count:

        raise ValueError(
            f"Expected {expected_count} examples, "
            f"received {len(examples)}."
        )

    for index, example in enumerate(examples):

        if not isinstance(example, dict):

            raise ValueError(
                f"Example {index} is not a JSON object."
            )

        missing = (
            REQUIRED_FIELDS
            - set(example.keys())
        )

        if missing:

            raise ValueError(
                f"Example {index} missing fields: "
                f"{sorted(missing)}"
            )

        if example["target_label"] != expected_label:

            raise ValueError(
                f"Example {index} has incorrect label: "
                f"{example['target_label']}"
            )

        if example["domain"] != expected_domain:

            raise ValueError(
                f"Example {index} has incorrect domain: "
                f"{example['domain']}"
            )

        if example["difficulty"] != expected_difficulty:

            raise ValueError(
                f"Example {index} has incorrect difficulty: "
                f"{example['difficulty']}"
            )

        for field in REQUIRED_FIELDS:

            if not isinstance(example[field], str):

                raise ValueError(
                    f"Example {index} field '{field}' "
                    "must be a string."
                )

            if not example[field].strip():

                raise ValueError(
                    f"Example {index} field '{field}' "
                    "is empty."
                )

    return True


# ============================================================
# SINGLE BATCH GENERATION
# ============================================================

def generate_batch(
    label: str,
    domain: str,
    difficulty: str,
    count: int,
):
    """
    Generate one batch using Gemini.
    """

    prompt = build_generation_prompt(
        label=label,
        domain=domain,
        difficulty=difficulty,
        count=count,
    )

    # Combine the system instructions and generation request.
    full_prompt = (
        SYSTEM_PROMPT.strip()
        + "\n\n"
        + prompt.strip()
    )

    response = client.models.generate_content(
    model=GENERATOR_MODEL,
    contents=full_prompt,
    config=types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
        response_schema=SYNTHETIC_EXAMPLE_SCHEMA,
    ),
)

    if not response.text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    examples = parse_json_response(
        response.text
    )

    check_batch_structure(
        examples=examples,
        expected_count=count,
        expected_label=label,
        expected_domain=domain,
        expected_difficulty=difficulty,
    )

    return examples


# ============================================================
# RETRY WRAPPER
# ============================================================

def generate_batch_with_retry(
    label: str,
    domain: str,
    difficulty: str,
    count: int,
):
    """
    Generate a batch with retry handling.
    """

    last_error = None

    for attempt in range(
        1,
        MAX_BATCH_ATTEMPTS + 1,
    ):

        print(
            f"      Attempt "
            f"{attempt}/{MAX_BATCH_ATTEMPTS}"
        )

        try:

            examples = generate_batch(
                label=label,
                domain=domain,
                difficulty=difficulty,
                count=count,
            )

            return examples

        except Exception as exc:

            last_error = exc

            print(
                f"      Generation failed: {exc}"
            )

        if attempt < MAX_BATCH_ATTEMPTS:

            wait_time = 5 * (2 ** (attempt - 1))

            print(
                f"      Retrying in "
                f"{wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Batch generation failed after "
        f"{MAX_BATCH_ATTEMPTS} attempts."
    ) from last_error


# ============================================================
# EXISTING RAW DATA
# ============================================================

def load_existing_examples():
    """
    Load examples already generated in previous runs.

    This allows the generation process to resume safely.
    """

    if not Path(RAW_DATA_PATH).exists():

        return []

    examples = []

    with open(
        RAW_DATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    return examples


# ============================================================
# COUNT EXISTING EXAMPLES
# ============================================================

def count_by_label(examples):

    counts = {
        label: 0
        for label in LABELS
    }

    for example in examples:

        label = example.get(
            "target_label"
        )

        if label in counts:

            counts[label] += 1

    return counts


# ============================================================
# SAVE RAW EXAMPLES
# ============================================================

def append_raw_examples(examples):
    """
    Append generated examples to JSONL.
    """

    with open(
        RAW_DATA_PATH,
        "a",
        encoding="utf-8",
    ) as file:

        for example in examples:

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# METADATA
# ============================================================

def save_metadata(metadata):

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# GENERATION PLAN
# ============================================================

def build_generation_plan():
    """
    Create a balanced generation plan.

    Each class receives exactly 378 examples.

    Examples are distributed approximately evenly across
    the eight domains.

    Difficulty distribution:
        Easy   ≈ 30%
        Medium ≈ 45%
        Hard   ≈ 25%
    """

    plan = []

    base_per_domain = (
        SYNTHETIC_SAMPLES_PER_CLASS
        // len(DOMAINS)
    )

    remainder = (
        SYNTHETIC_SAMPLES_PER_CLASS
        % len(DOMAINS)
    )

    for label in LABELS:

        for domain_index, domain in enumerate(DOMAINS):

            domain_count = base_per_domain

            if domain_index < remainder:
                domain_count += 1

            easy = round(
                domain_count * 0.30
            )

            medium = round(
                domain_count * 0.45
            )

            hard = (
                domain_count
                - easy
                - medium
            )

            difficulty_counts = {
                "easy": easy,
                "medium": medium,
                "hard": hard,
            }

            for difficulty in DIFFICULTY_LEVELS:

                count = difficulty_counts[
                    difficulty
                ]

                if count <= 0:
                    continue

                plan.append(
                    {
                        "label": label,
                        "domain": domain,
                        "difficulty": difficulty,
                        "count": count,
                    }
                )

    return plan


# ============================================================
# METADATA INITIALIZATION
# ============================================================

def initialize_metadata(existing_counts):

    metadata = {
        "prompt_version": PROMPT_VERSION,
        "generator_provider": "Google Gemini",
        "generator_model": GENERATOR_MODEL,

        "generation_started_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "target_per_class":
            SYNTHETIC_SAMPLES_PER_CLASS,

        "batch_size":
            BATCH_SIZE,

        "labels":
            LABELS,

        "domains":
            DOMAINS,

        "difficulty_distribution": {
            "easy": 0.30,
            "medium": 0.45,
            "hard": 0.25,
        },

        "existing_counts":
            existing_counts,

        "generated_counts":
            existing_counts.copy(),

        "total_generated":
            sum(existing_counts.values()),
    }

    return metadata


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SYNTHETIC DATA GENERATION — GOOGLE GEMINI")
    print("=" * 70)

    print(
        f"Generator model : {GENERATOR_MODEL}"
    )

    print(
        f"Target/class    : "
        f"{SYNTHETIC_SAMPLES_PER_CLASS}"
    )

    print(
        f"Total synthetic : "
        f"{SYNTHETIC_SAMPLES_PER_CLASS * len(LABELS)}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Prompt version  : {PROMPT_VERSION}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Ensure output directories exist.
    # --------------------------------------------------------

    Path(
        RAW_DATA_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        METADATA_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load previous generation if present.
    # --------------------------------------------------------

    existing_examples = (
        load_existing_examples()
    )

    existing_counts = count_by_label(
        existing_examples
    )

    if existing_examples:

        print(
            "\nExisting raw examples detected."
        )

        for label in LABELS:

            print(
                f"  {label}: "
                f"{existing_counts[label]}"
            )

        print(
            "\nGeneration will resume from "
            "the existing counts."
        )

    else:

        print(
            "\nNo existing raw dataset found."
        )

    # --------------------------------------------------------
    # Build generation plan.
    # --------------------------------------------------------

    plan = build_generation_plan()

    print(
        f"\nGeneration plan entries: "
        f"{len(plan)}"
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = initialize_metadata(
        existing_counts
    )

    # --------------------------------------------------------
    # Generate according to plan.
    # --------------------------------------------------------

    for index, item in enumerate(
        plan,
        start=1,
    ):

        label = item["label"]
        domain = item["domain"]
        difficulty = item["difficulty"]
        target_count = item["count"]

        already_generated = sum(
            1
            for example in existing_examples
            if (
                example.get("target_label")
                == label
                and example.get("domain")
                == domain
                and example.get("difficulty")
                == difficulty
            )
        )

        remaining = (
            target_count
            - already_generated
        )

        print(
            f"\n[{index}/{len(plan)}] "
            f"{label} | "
            f"{domain} | "
            f"{difficulty}"
        )

        print(
            f"      Target: {target_count}"
        )

        print(
            f"      Existing: "
            f"{already_generated}"
        )

        print(
            f"      Remaining: "
            f"{max(remaining, 0)}"
        )

        if remaining <= 0:

            print(
                "      ✓ Already complete"
            )

            continue

        while remaining > 0:

            batch_count = min(
                BATCH_SIZE,
                remaining,
            )

            print(
                f"      Generating "
                f"{batch_count} examples..."
            )

            examples = (
                generate_batch_with_retry(
                    label=label,
                    domain=domain,
                    difficulty=difficulty,
                    count=batch_count,
                )
            )

            # ------------------------------------------------
            # Attach generation metadata.
            # ------------------------------------------------

            for example in examples:

                example["_prompt_version"] = (
                    PROMPT_VERSION
                )

                example["_generator_model"] = (
                    GENERATOR_MODEL
                )

                example["_generated_at_utc"] = (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                )

            # ------------------------------------------------
            # Save immediately.
            # ------------------------------------------------

            append_raw_examples(
                examples
            )

            existing_examples.extend(
                examples
            )

            remaining -= len(
                examples
            )

            metadata["generated_counts"] = (
                count_by_label(
                    existing_examples
                )
            )

            metadata["total_generated"] = (
                len(existing_examples)
            )

            metadata[
                "last_updated_utc"
            ] = datetime.now(
                timezone.utc
            ).isoformat()

            save_metadata(
                metadata
            )

            print(
                f"      ✓ Saved "
                f"{len(examples)} examples"
            )

            print(
                f"      Total generated: "
                f"{len(existing_examples)}"
            )

    # --------------------------------------------------------
    # Final verification.
    # --------------------------------------------------------

    final_counts = count_by_label(
        existing_examples
    )

    metadata["final_counts"] = (
        final_counts
    )

    metadata["generation_finished_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    save_metadata(
        metadata
    )

    print("\n")
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    for label in LABELS:

        print(
            f"{label:22s}: "
            f"{final_counts[label]}"
        )

    print(
        f"\nTotal: "
        f"{sum(final_counts.values())}"
    )

    print(
        f"\nRaw dataset:"
        f"\n{RAW_DATA_PATH}"
    )

    print(
        f"\nMetadata:"
        f"\n{METADATA_PATH}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()