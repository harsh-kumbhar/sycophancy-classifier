from evaluation_mode import (
    load_evaluation_examples,
    get_random_example,
)


def main():

    print("=" * 70)
    print("TESTING EVALUATION MODE")
    print("=" * 70)

    examples = load_evaluation_examples()

    print("\nAccepted examples by label:")
    print("-" * 70)

    total = 0

    for label, rows in examples.items():

        print(
            f"{label:20s}: {len(rows)}"
        )

        total += len(rows)

    print("-" * 70)

    print(
        f"TOTAL ACCEPTED EXAMPLES: {total}"
    )

    print("\nRandom examples:")
    print("-" * 70)

    for label in examples:

        example = get_random_example(
            examples,
            label
        )

        print(
            f"\n{label}"
        )

        print(
            f"Question: {example['question']}"
        )

        print(
            f"Initial: {example['initial_answer'][:100]}..."
        )

        print(
            f"Feedback: {example['user_feedback'][:100]}..."
        )

        print(
            f"Revised: {example['revised_answer'][:100]}..."
        )

    print("\n" + "=" * 70)
    print("EVALUATION MODE DATA TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()