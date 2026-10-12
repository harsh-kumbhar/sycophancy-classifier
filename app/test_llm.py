import os 
from pathlib import Path
from dotenv import load_dotenv

from llm import GeminiChat


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)

def main():

    print("=" * 70)
    print("TESTING GEMINI CHAT")
    print("=" * 70)

    gemini = GeminiChat()

    messages = [
        {
            "role": "user",
            "content": "What is 2 + 2?"
        }
    ]

    response = gemini.generate_response(
        messages
    )

    print("\nGemini response:")
    print(response)

    print("\n" + "=" * 70)
    print("GEMINI TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()