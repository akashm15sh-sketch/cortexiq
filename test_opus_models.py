import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

models_to_try = [
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
    "claude-4-opus-latest",
    "claude-opus-4-6",
    "claude-4-6-opus",
    "claude-3-5-sonnet-20240620"
]

for m in models_to_try:
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
