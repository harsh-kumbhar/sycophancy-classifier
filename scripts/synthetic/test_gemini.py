import os

from dotenv import load_dotenv
from google import genai


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in .env"
    )


client = genai.Client(
    api_key=api_key
)


response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Reply with exactly: GEMINI API CONNECTION SUCCESSFUL"
)


print("=" * 60)
print("GEMINI API TEST")
print("=" * 60)
print(response.text)
print("=" * 60)