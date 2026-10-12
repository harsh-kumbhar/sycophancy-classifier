from feedback_detector import detect_feedback


# ============================================================
# TEST CASES
# ============================================================

POSITIVE_CASES = [

    "I think you're wrong. The answer should be 70.",

    "That's incorrect. Please reconsider your answer.",

    "Are you sure about that?",

    "I disagree with your calculation.",

    "Actually, the correct answer is 42.",

    "Wait, I think the answer should be 15.",

    "Your calculation is wrong.",

    "The correct value is 3.14.",

    "Can you double check your answer?",

    "That doesn't seem right to me.",

    "My calculation gives 48, not 50.",

    "You are wrong. Please check your reasoning.",
        "I think you should reconsider the calculation. "
    "Average speed for the whole journey is total distance "
    "divided by total time. Please recalculate.",

    "Please recalculate the answer.",

    "I think you should reconsider your reasoning.",

    "Can you recheck your calculation?",

    "Actually, your calculation contains an error.",

    "I believe the correct result should be 48.",
]


NEGATIVE_CASES = [

    "What is the capital of France?",

    "Tell me about photosynthesis.",

    "Explain Newton's first law.",

    "What is machine learning?",

    "Can you give me an example?",

    "Tell me a joke.",

    "What happened during World War II?",

    "How does gradient descent work?",

    "Give me three Python project ideas.",

    "What is the difference between AI and ML?",

    "Can you explain this in simple terms?",

    "What should I study next?",

]


# ============================================================
# RUN TESTS
# ============================================================

def main():

    print("=" * 70)
    print("TESTING FEEDBACK DETECTOR")
    print("=" * 70)

    print("\nPOSITIVE CASES")
    print("-" * 70)

    positive_failures = 0

    for message in POSITIVE_CASES:

        result = detect_feedback(message)

        status = "✓" if result else "✗"

        print(
            f"{status} {message}"
        )

        if not result:
            positive_failures += 1

    print("\nNEGATIVE CASES")
    print("-" * 70)

    negative_failures = 0

    for message in NEGATIVE_CASES:

        result = detect_feedback(message)

        status = "✓" if not result else "✗"

        print(
            f"{status} {message}"
        )

        if result:
            negative_failures += 1

    print("\n" + "=" * 70)

    if positive_failures == 0 and negative_failures == 0:

        print("ALL FEEDBACK DETECTOR TESTS PASSED")

    else:

        print("SOME FEEDBACK DETECTOR TESTS FAILED")

        print(
            f"Positive failures: {positive_failures}"
        )

        print(
            f"Negative failures: {negative_failures}"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()