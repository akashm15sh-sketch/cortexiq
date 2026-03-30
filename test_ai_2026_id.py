import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

try:
    print(f"Testing claude-opus-4-6...")
    res = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print(f"SUCCESS: claude-opus-4-6 OK: {res.content[0].text}")
except Exception as e:
    print(f"FAILED: claude-opus-4-6: {e}")

try:
    print(f"Testing claude-sonnet-4-6...")
    res = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print(f"SUCCESS: claude-sonnet-4-6 OK: {res.content[0].text}")
except Exception as e:
    print(f"FAILED: claude-sonnet-4-6: {e}")
