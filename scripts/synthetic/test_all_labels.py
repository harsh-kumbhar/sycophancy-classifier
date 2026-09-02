from generate import generate_batch


def main():

    print("=" * 70)
    print("TESTING JUSTIFIED_CHANGE GENERATION")
    print("=" * 70)

    examples = generate_batch(
        label="JUSTIFIED_CHANGE",
        domain="logical_reasoning",
        difficulty="medium",
        count=5,
    )

    print(
        f"\n✓ Generated {len(examples)} examples"
    )

    for i, example in enumerate(
        examples,
        start=1,
    ):

        print("\n" + "-" * 70)
        print(f"EXAMPLE {i}")
        print("-" * 70)

        print("\nQuestion:")
        print(example["question"])

        print("\nInitial answer:")
        print(example["initial_answer"])

        print("\nUser feedback:")
        print(example["user_feedback"])

        print("\nRevised answer:")
        print(example["revised_answer"])


if __name__ == "__main__":
    main()