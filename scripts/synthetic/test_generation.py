# ============================================================
# TEST SYNTHETIC GENERATION PIPELINE
# ============================================================

from generate import generate_batch


def main():

    print("=" * 70)
    print("TESTING GEMINI SYNTHETIC GENERATION")
    print("=" * 70)

    examples = generate_batch(
        label="SYCOPHANTIC_SHIFT",
        domain="basic_math",
        difficulty="medium",
        count=2,
    )

    print("\n✓ Gemini generation successful")
    print(f"Examples received: {len(examples)}")

    for i, example in enumerate(
        examples,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"EXAMPLE {i}")
        print("-" * 70)

        print(
            "Domain     :",
            example["domain"],
        )

        print(
            "Difficulty :",
            example["difficulty"],
        )

        print(
            "Label      :",
            example["target_label"],
        )

        print(
            "\nQuestion:"
        )

        print(
            example["question"]
        )

        print(
            "\nInitial answer:"
        )

        print(
            example["initial_answer"]
        )

        print(
            "\nUser feedback:"
        )

        print(
            example["user_feedback"]
        )

        print(
            "\nRevised answer:"
        )

        print(
            example["revised_answer"]
        )

    print("\n" + "=" * 70)
    print("STRUCTURAL GENERATION TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()