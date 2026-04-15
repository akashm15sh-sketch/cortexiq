import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
KILO_API_KEY = os.getenv("KILO_API_KEY", "")
KILO_BASE_URL = "https://api.kilo.ai/api/gateway"
SECRET_KEY = os.getenv("SECRET_KEY", "cortexiq-demo-secret-key-2025")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cortexiq.db")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

# ── PostHog Analytics ──
POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY", "")
POSTHOG_HOST = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
