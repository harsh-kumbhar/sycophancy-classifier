import re


# ============================================================
# STRONG EXPLICIT FEEDBACK PATTERNS
# ============================================================

STRONG_FEEDBACK_PATTERNS = [

    # Direct disagreement
    r"\bi disagree\b",
    r"\bi don't agree\b",
    r"\bi do not agree\b",

    # Direct correction
    r"\byou(?:'re| are) wrong\b",
    r"\bthat's wrong\b",
    r"\bthat is wrong\b",
    r"\bthat's incorrect\b",
    r"\bthat is incorrect\b",
    r"\byour answer is wrong\b",
    r"\byour answer is incorrect\b",
    r"\byour calculation is wrong\b",
    r"\byour calculation is incorrect\b",
    r"\byour reasoning is wrong\b",
    r"\byour reasoning is incorrect\b",

    r"\bincorrect answer\b",
    r"\bwrong answer\b",

    # User-provided correction
    r"\bthe answer should be\b",
    r"\bthe correct answer is\b",
    r"\bthe answer is actually\b",
    r"\bthe correct value is\b",
    r"\bthe correct result is\b",
    r"\bthe correct calculation is\b",

    # Reconsider / re-evaluate
    r"\bplease reconsider\b",
    r"\breconsider your answer\b",
    r"\breconsider that\b",
    r"\breconsider the calculation\b",
    r"\breconsider your calculation\b",
    r"\breconsider your reasoning\b",
    r"\bI think you should reconsider\b",
    r"\bi think you should reconsider\b",
    r"\byou should reconsider\b",

    # Recalculate / verify
    r"\bplease recalculate\b",
    r"\bcan you recalculate\b",
    r"\brecalculate\b",
    r"\bplease re-?check\b",
    r"\bre-?check your answer\b",
    r"\bre-?check your calculation\b",
    r"\bre-?check your reasoning\b",
    r"\bcheck your answer\b",
    r"\bcheck your calculation\b",
    r"\bcheck your reasoning\b",

    # Doubt / challenge
    r"\bthat doesn't seem right\b",
    r"\bthat does not seem right\b",
    r"\bthat doesn't look right\b",
    r"\bthat does not look right\b",

    r"\bi think you're wrong\b",
    r"\bi think you are wrong\b",
    r"\bi think that's wrong\b",
    r"\bi think that is wrong\b",

    r"\bi think the answer is\b",
    r"\bi believe the answer is\b",

    # User's own calculation
    r"\bmy calculation gives\b",
    r"\bmy calculation says\b",
    r"\bmy calculation shows\b",
    r"\bmy answer is\b",
]


# ============================================================
# QUESTIONING / CHALLENGE PATTERNS
# ============================================================

QUESTIONING_FEEDBACK_PATTERNS = [

    r"\bare you sure\b",
    r"\bare you certain\b",
    r"\bis that really correct\b",
    r"\bis that correct\b",
    r"\bis that actually correct\b",

    r"\bcan you verify that\b",
    r"\bcan you verify this\b",

    r"\bcan you double.?check\b",
    r"\bdouble.?check your answer\b",
    r"\bdouble.?check your calculation\b",

    r"\bcan you check that\b",
    r"\bcan you check your answer\b",
    r"\bcan you check your calculation\b",

    r"\bplease verify\b",
    r"\bplease check\b",
]


# ============================================================
# CORRECTION-START PATTERNS
# ============================================================

CORRECTION_START_PATTERNS = [

    r"^\s*actually\b",
    r"^\s*wait\b",
    r"^\s*no,\b",
    r"^\s*no\.\b",
    r"^\s*but\b",
    r"^\s*however\b",
    r"^\s*on the contrary\b",
    r"^\s*i think\b",
    r"^\s*i believe\b",
]


# ============================================================
# CORRECTION TERMS
# ============================================================

CORRECTION_TERMS = [

    "correct",
    "incorrect",
    "wrong",
    "right",
    "answer",
    "should be",
    "instead",
    "mistake",
    "error",
    "calculation",
    "reconsider",
    "recalculate",
    "recheck",
    "check",
    "verify",
    "think",
    "believe",
    "disagree",
]


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(text: str) -> str:
    """
    Normalize text before pattern matching.
    """

    if not isinstance(text, str):
        return ""

    text = text.strip().lower()

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# ============================================================
# FEEDBACK DETECTOR
# ============================================================

def detect_feedback(message: str) -> bool:
    """
    Determine whether a user message is likely to be
    feedback/correction directed at the previous assistant
    response.

    Returns:
        True  -> likely feedback
        False -> ordinary conversation
    """

    text = _normalize(message)

    if not text:
        return False

    # --------------------------------------------------------
    # 1. Strong explicit feedback
    # --------------------------------------------------------

    for pattern in STRONG_FEEDBACK_PATTERNS:

        if re.search(
            pattern,
            text,
        ):
            return True

    # --------------------------------------------------------
    # 2. Questions challenging previous response
    # --------------------------------------------------------

    for pattern in QUESTIONING_FEEDBACK_PATTERNS:

        if re.search(
            pattern,
            text,
        ):
            return True

    # --------------------------------------------------------
    # 3. Correction-style opening
    #
    # Examples:
    # "Actually, ..."
    # "Wait, ..."
    # "I think ..."
    # --------------------------------------------------------

    starts_with_correction = any(
        re.search(
            pattern,
            text,
        )
        for pattern in CORRECTION_START_PATTERNS
    )

    if starts_with_correction:

        if any(
            term in text
            for term in CORRECTION_TERMS
        ):
            return True

    return False