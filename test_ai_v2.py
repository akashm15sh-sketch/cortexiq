import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key)

try:
    print(f"Testing Sonnet 3.5 (v2)...")
    res = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print(f"Sonnet 3.5 v2 OK: {res.content[0].text}")
except Exception as e:
    print(f"Sonnet 3.5 v2 Failed: {e}")

try:
    print(f"Testing Haiku 3.5...")
    res = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=10,
        messages=[{"role": "user", "content": "Hi"}]
    )
    print(f"Haiku 3.5 OK: {res.content[0].text}")
except Exception as e:
    print(f"Haiku 3.5 Failed: {e}")
