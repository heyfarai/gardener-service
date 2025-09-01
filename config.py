import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment configuration
API_KEY = os.getenv("GARDENER_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Auto-titler configuration
AUTO_TITLE_THRESHOLD = int(os.getenv("AUTO_TITLE_THRESHOLD", "5"))  # Min snippets to trigger auto-titling
AUTO_TITLE_ENABLED = os.getenv("AUTO_TITLE_ENABLED", "true").lower() == "true"
