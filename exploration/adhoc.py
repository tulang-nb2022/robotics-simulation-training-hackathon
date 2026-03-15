import os

# Load .env so NEBIUS_API_KEY / OPENAI_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# Prefer Nebius key; fall back to OPENAI_API_KEY (OpenAI client requires one of these)
api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "Set NEBIUS_API_KEY or OPENAI_API_KEY in your .env or environment. "
        "Run: pip install python-dotenv and put the key in .env, or export it."
    )

client = OpenAI(
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=api_key,
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[
        {
            "role": "system",
            "content": """SYSTEM_PROMPT"""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """USER_MESSAGE"""
                }
            ]
        }
    ]
)

print(response.to_json())