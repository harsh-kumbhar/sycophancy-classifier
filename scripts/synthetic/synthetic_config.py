# ============================================================
# SYNTHETIC DATA GENERATION CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# DATASET SIZE
# ------------------------------------------------------------

CURRENT_SAMPLES_PER_CLASS = 622
TARGET_SAMPLES_PER_CLASS = 1000

SYNTHETIC_SAMPLES_PER_CLASS = (
    TARGET_SAMPLES_PER_CLASS - CURRENT_SAMPLES_PER_CLASS
)

TOTAL_SYNTHETIC_SAMPLES = SYNTHETIC_SAMPLES_PER_CLASS * 3
FINAL_TRAINING_SAMPLES = TARGET_SAMPLES_PER_CLASS * 3


# ------------------------------------------------------------
# LABELS
# ------------------------------------------------------------

LABELS = [
    "CONSISTENT",
    "JUSTIFIED_CHANGE",
    "SYCOPHANTIC_SHIFT",
]


# ------------------------------------------------------------
# DOMAINS
# ------------------------------------------------------------

DOMAINS = [
    "analogies",
    "causal_reasoning",
    "common_sense",
    "word_problems",
    "logical_reasoning",
    "basic_math",
    "scientific_facts",
    "reading_comprehension",
]


# ------------------------------------------------------------
# DIFFICULTY LEVELS
# ------------------------------------------------------------

DIFFICULTY_LEVELS = [
    "easy",
    "medium",
    "hard",
]


# Approximate target proportions.
# These do not have to produce exact counts because generation
# will be controlled programmatically.

DIFFICULTY_DISTRIBUTION = {
    "easy": 0.30,
    "medium": 0.45,
    "hard": 0.25,
}


# ------------------------------------------------------------
# GENERATION
# ------------------------------------------------------------

BATCH_SIZE = 10
# Maximum number of generation attempts for a batch.
MAX_BATCH_ATTEMPTS = 3


# ------------------------------------------------------------
# OUTPUT PATHS
# ------------------------------------------------------------

RAW_DIR = "data/synthetic/raw"
VALIDATED_DIR = "data/synthetic/validated"
FINAL_DIR = "data/synthetic/final"

RAW_DATA_PATH = f"{RAW_DIR}/synthetic_raw.jsonl"

VALIDATED_DATA_PATH = (
    f"{VALIDATED_DIR}/synthetic_validated.jsonl"
)

FINAL_DATA_PATH = (
    f"{FINAL_DIR}/synthetic_training.csv"
)

AUGMENTED_TRAIN_PATH = (
    f"{FINAL_DIR}/augmented_train.csv"
)

METADATA_PATH = (
    "data/synthetic/generation_metadata.json"
)


# ------------------------------------------------------------
# PROMPT VERSIONING
# ------------------------------------------------------------

PROMPT_VERSION = "v1.0"


# ------------------------------------------------------------
# GENERATOR MODEL
# ------------------------------------------------------------

GENERATOR_PROVIDER = "Google Gemini"
GENERATOR_MODEL = "gemini-3.5-flash-lite"