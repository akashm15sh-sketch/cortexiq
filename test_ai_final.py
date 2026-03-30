import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

models_2026 = [
    "claude-4-6-opus-20260205",
    "claude-4-6-sonnet-20260217",
    "claude-4-opus-20260205",
    "claude-4-sonnet-20260217",
]

for m in models_2026:
    try:
        print(f"Testing {m}...")
        res = client.messages.create(
            model=m,
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"SUCCESS: {m} OK: {res.content[0].text}")
        break
    except Exception as e:
        print(f"FAILED: {m}: {e}")
