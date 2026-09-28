"""
Hybrid Context-Aware Feedback Detector

Layers:
1. Deterministic rules for explicit feedback.
2. Context-aware heuristics for implicit feedback such as "Paper".
3. Optional local SentenceTransformer fallback for ambiguous cases.

The public detect_feedback() function remains backward compatible.
Use analyze_feedback() when you want metadata about the detected event.
"""

import re
from typing import Any, Dict, Optional


# ============================================================
# STRONG EXPLICIT FEEDBACK
# ============================================================

STRONG_FEEDBACK_PATTERNS = [
    r"\bi disagree\b",
    r"\bi don't agree\b",
    r"\bi do not agree\b",

    r"\byou(?:'re| are) wrong\b",
    r"\bthat's wrong\b",
    r"\bthat is wrong\b",
    r"\bthat's incorrect\b",
    r"\bthat is incorrect\b",
    r"\byour answer is wrong\b",
    r"\byour answer is incorrect\b",

    r"\bincorrect answer\b",
    r"\bwrong answer\b",

    r"\bthe answer should be\b",
    r"\bthe correct answer is\b",
    r"\bthe answer is actually\b",
    r"\bthe correct value is\b",
    r"\bthe correct result is\b",

    r"\bplease reconsider\b",
    r"\breconsider your answer\b",

    r"\bplease recalculate\b",
    r"\brecalculate\b",
    r"\bplease re-?check\b",
    r"\bcheck your answer\b",
    r"\bcheck your calculation\b",

    r"\bthat doesn't seem right\b",
    r"\bthat does not seem right\b",
    r"\bthat doesn't look right\b",

    r"\bi think you're wrong\b",
    r"\bi think you are wrong\b",
    r"\bi think that's wrong\b",

    r"\bi think the answer is\b",
    r"\bi believe the answer is\b",

    r"\bmy calculation gives\b",
    r"\bmy calculation says\b",
    r"\bmy answer is\b",
]


# ============================================================
# CHALLENGE / VERIFICATION
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

    r"\bcan you check that\b",
    r"\bcan you check your answer\b",

    r"\bplease verify\b",
    r"\bplease check\b",

    # Natural challenges without "wrong"
    r"\bwhy would (?:that|this) be\b",
    r"\bhow is (?:that|this) correct\b",
    r"\bhow can (?:that|this) be correct\b",
    r"\bdoesn't that contradict\b",
]


# ============================================================
# CORRECTION-STYLE OPENINGS
# ============================================================

CORRECTION_START_PATTERNS = [
    r"^\s*actually\b",
    r"^\s*wait\b",
    r"^\s*no\b",
    r"^\s*but\b",
    r"^\s*however\b",
    r"^\s*on the contrary\b",
    r"^\s*i think\b",
    r"^\s*i believe\b",
    r"^\s*i meant\b",
    r"^\s*i mean\b",
    r"^\s*the answer is\b",
    r"^\s*the correct answer is\b",
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
    "rather",
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
    "mean",
    "meant",
]


# ============================================================
# NORMAL / NON-FEEDBACK PHRASES
# ============================================================

NON_FEEDBACK_PATTERNS = [
    r"^\s*(thanks|thank you|thx)\s*[.!]*$",
    r"^\s*(okay|ok|alright|sure)\s*[.!]*$",
    r"^\s*(got it|understood)\s*[.!]*$",
    r"^\s*(cool|nice|great)\s*[.!]*$",
]


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    return (
        text
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
    )


def _clean(text: str) -> str:
    text = _normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _matches_any(text: str, patterns) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


# ============================================================
# RULE LAYER
# ============================================================

def _explicit_feedback(text: str) -> bool:

    if _matches_any(text, STRONG_FEEDBACK_PATTERNS):
        return True

    if _matches_any(text, QUESTIONING_FEEDBACK_PATTERNS):
        return True

    starts_with_correction = _matches_any(
        text,
        CORRECTION_START_PATTERNS,
    )

    if starts_with_correction:
        return any(
            term in text
            for term in CORRECTION_TERMS
        )

    return False


# ============================================================
# CANDIDATE ANSWER EXTRACTION
# ============================================================

def _extract_candidate(text: str) -> str:

    patterns = [
        r"^\s*i\s+meant\s+(.+)$",
        r"^\s*i\s+mean\s+(.+)$",
        r"^\s*the\s+answer\s+is\s+(.+)$",
        r"^\s*the\s+correct\s+answer\s+is\s+(.+)$",
        r"^\s*no[,.]?\s+(.+)$",
        r"^\s*actually[,.]?\s+(.+)$",
    ]

    for pattern in patterns:

        match = re.match(
            pattern,
            text,
        )

        if match:
            candidate = match.group(1).strip()

            if candidate:
                return candidate

    return ""


def _is_short_answer(text: str) -> bool:

    words = _clean(text).split()

    return bool(words) and len(words) <= 8


def _same_answer(
    message: str,
    previous_assistant_response: str,
) -> bool:

    msg = _clean(message)
    prev = _clean(previous_assistant_response)

    if not msg or not prev:
        return False

    # Exact repetition
    if msg == prev:
        return True

    confirmation_patterns = [
        r"^(yes|correct|right|exactly)$",
        r"^.+\s+is\s+(correct|right)$",
        r"^.+\s+is\s+the\s+answer$",
    ]

    return _matches_any(
        msg,
        confirmation_patterns,
    )


# ============================================================
# CONTEXT-AWARE LAYER
# ============================================================

def _contextual_feedback(
    message: str,
    previous_assistant_response: str,
    original_question: str,
) -> Dict[str, Any]:

    result = {
        "detected": False,
        "type": None,
        "confidence": 0.0,
        "candidate_answer": None,
        "reason": None,
    }

    if not message:
        return result

    # --------------------------------------------------------
    # Explicit replacement
    # --------------------------------------------------------

    candidate = _extract_candidate(message)

    if candidate:

        result.update(
            detected=True,
            type="correction",
            confidence=0.97,
            candidate_answer=candidate,
            reason=(
                "User supplied an explicit replacement/correction."
            ),
        )

        return result

    # --------------------------------------------------------
    # Same-answer confirmation
    # --------------------------------------------------------

    if previous_assistant_response:

        if _same_answer(
            message,
            previous_assistant_response,
        ):

            result.update(
                detected=True,
                type="confirmation",
                confidence=0.94,
                candidate_answer=message.strip(),
                reason=(
                    "User confirmed or repeated the previous answer."
                ),
            )

            return result

    # --------------------------------------------------------
    # Direct rejection
    # --------------------------------------------------------

    rejection_patterns = [
        r"^\s*no[,.]?\s*$",
        r"^\s*not that\s*$",
        r"^\s*not this\s*$",
        r"^\s*that's not what i meant\s*$",
        r"^\s*that isn't what i meant\s*$",
        r"^\s*you misunderstood\s*$",
        r"^\s*that's not it\s*$",
    ]

    if _matches_any(
        message,
        rejection_patterns,
    ):

        result.update(
            detected=True,
            type="rejection",
            confidence=0.95,
            reason=(
                "User rejected the previous response."
            ),
        )

        return result

    # --------------------------------------------------------
    # CRITICAL CASE:
    #
    # Assistant: Ink.
    # User: Paper
    #
    # "Paper" contains no feedback keyword.
    # Context makes it a candidate correction.
    # --------------------------------------------------------

    if (
        _is_short_answer(message)
        and previous_assistant_response
        and original_question
    ):

        msg = _clean(message)
        prev = _clean(previous_assistant_response)

        if msg and prev and msg != prev:

            result.update(
                detected=True,
                type="implicit_correction",
                confidence=0.86,
                candidate_answer=message.strip(),
                reason=(
                    "Short candidate answer differs from the "
                    "previous assistant answer in a contextual "
                    "answer exchange."
                ),
            )

            return result

    # --------------------------------------------------------
    # User-provided alternative answer
    # --------------------------------------------------------

    alternative_patterns = [
        r"\bmy answer\b",
        r"\bi got\b",
        r"\bi get\b",
        r"\bmy result\b",
        r"\bi calculated\b",
        r"\bi think the answer\b",
        r"\bi believe the answer\b",
        r"\bi would say\b",
        r"\bi'd say\b",
        r"\binstead\b",
        r"\brather\b",
        r"\bshould be\b",
    ]

    if _matches_any(
        message,
        alternative_patterns,
    ):

        result.update(
            detected=True,
            type="user_alternative",
            confidence=0.90,
            candidate_answer=message.strip(),
            reason=(
                "User supplied or indicated an alternative answer."
            ),
        )

        return result

    return result


# ============================================================
# OPTIONAL SEMANTIC FALLBACK
# ============================================================

class SemanticFeedbackDetector:
    """
    Optional local SentenceTransformer fallback.

    It is NOT the primary detector.

    It is only used after the deterministic and contextual
    layers cannot decide.
    """

    def __init__(
        self,
        model_name: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
    ):

        self.model_name = model_name
        self._model = None
        self._available = None

    def _load(self) -> bool:

        if self._model is not None:
            return True

        if self._available is False:
            return False

        try:

            from sentence_transformers import (
                SentenceTransformer,
            )

            self._model = SentenceTransformer(
                self.model_name
            )

            self._available = True

            return True

        except Exception:

            self._available = False

            return False

    def similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> Optional[float]:

        if (
            not text_a
            or not text_b
            or not self._load()
        ):
            return None

        try:

            embeddings = self._model.encode(
                [text_a, text_b],
                normalize_embeddings=True,
            )

            score = float(
                embeddings[0] @ embeddings[1]
            )

            return max(
                0.0,
                min(1.0, score),
            )

        except Exception:

            return None

    def detect(
        self,
        message: str,
        previous_assistant_response: str,
        original_question: str,
    ) -> Dict[str, Any]:

        result = {
            "detected": False,
            "type": None,
            "confidence": 0.0,
            "reason": None,
        }

        if (
            not message
            or not previous_assistant_response
        ):
            return result

        answer_similarity = self.similarity(
            message,
            previous_assistant_response,
        )

        if answer_similarity is None:
            return result

        result["semantic_similarity"] = answer_similarity

        # Strong semantic similarity → likely confirmation.
        if answer_similarity >= 0.82:

            result.update(
                detected=True,
                type="semantic_confirmation",
                confidence=min(
                    0.90,
                    answer_similarity,
                ),
                reason=(
                    "User message is semantically close "
                    "to the previous answer."
                ),
            )

            return result

        # Compare against question for ambiguous alternatives.
        question_similarity = self.similarity(
            message,
            original_question,
        )

        if (
            question_similarity is not None
            and question_similarity >= 0.35
        ):

            result.update(
                detected=True,
                type="semantic_alternative",
                confidence=min(
                    0.80,
                    question_similarity,
                ),
                reason=(
                    "User message is related to the question "
                    "but differs from the previous answer."
                ),
            )

        return result


_semantic_detector = None


def _get_semantic_detector():

    global _semantic_detector

    if _semantic_detector is None:
        _semantic_detector = (
            SemanticFeedbackDetector()
        )

    return _semantic_detector


# ============================================================
# PUBLIC HYBRID API
# ============================================================

def analyze_feedback(
    message: str,
    previous_assistant_response: str = "",
    original_question: str = "",
    use_semantic_fallback: bool = True,
) -> Dict[str, Any]:

    result = {
        "is_feedback": False,
        "feedback_type": None,
        "confidence": 0.0,
        "candidate_answer": None,
        "reason": None,
        "detector": "none",
    }

    text = _normalize(message)

    if not text:
        return result

    # Ignore obvious conversational acknowledgements.
    if _matches_any(
        text,
        NON_FEEDBACK_PATTERNS,
    ):

        result["reason"] = (
            "Common non-feedback conversational response."
        )

        return result

    # ========================================================
    # LAYER 1 — DETERMINISTIC RULES
    # ========================================================

    if _explicit_feedback(text):

        result.update(
            is_feedback=True,
            feedback_type="explicit_feedback",
            confidence=0.98,
            reason=(
                "Matched an explicit feedback/challenge pattern."
            ),
            detector="rules",
        )

        return result

    # ========================================================
    # LAYER 2 — CONTEXTUAL HEURISTICS
    # ========================================================

    contextual = _contextual_feedback(
        message=text,
        previous_assistant_response=(
            previous_assistant_response
        ),
        original_question=original_question,
    )

    if contextual["detected"]:

        result.update(
            is_feedback=True,
            feedback_type=contextual["type"],
            confidence=contextual["confidence"],
            candidate_answer=contextual[
                "candidate_answer"
            ],
            reason=contextual["reason"],
            detector="context",
        )

        return result

    # ========================================================
    # LAYER 3 — OPTIONAL SEMANTIC FALLBACK
    # ========================================================

    if use_semantic_fallback:

        semantic = _get_semantic_detector().detect(
            message=text,
            previous_assistant_response=(
                previous_assistant_response
            ),
            original_question=original_question,
        )

        if semantic.get("detected"):

            result.update(
                is_feedback=True,
                feedback_type=semantic["type"],
                confidence=semantic["confidence"],
                reason=semantic["reason"],
                detector="semantic",
            )

            result["semantic_similarity"] = (
                semantic.get("semantic_similarity")
            )

            return result

    return result


# ============================================================
# BACKWARD-COMPATIBLE API
# ============================================================

def detect_feedback(
    message: str,
    previous_assistant_response: str = "",
    original_question: str = "",
    use_semantic_fallback: bool = True,
) -> bool:
    """
    Backward-compatible boolean detector.

    Old usage still works:

        detect_feedback(user_message)

    Recommended usage:

        detect_feedback(
            user_message,
            previous_assistant_response,
            original_question,
        )
    """

    result = analyze_feedback(
        message=message,
        previous_assistant_response=(
            previous_assistant_response
        ),
        original_question=original_question,
        use_semantic_fallback=use_semantic_fallback,
    )

    return bool(result["is_feedback"])


# ============================================================
# DEBUG HELPER
# ============================================================

def debug_feedback(
    message: str,
    previous_assistant_response: str = "",
    original_question: str = "",
) -> Dict[str, Any]:

    return analyze_feedback(
        message=message,
        previous_assistant_response=(
            previous_assistant_response
        ),
        original_question=original_question,
        use_semantic_fallback=True,
    )