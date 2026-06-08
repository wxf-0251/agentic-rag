import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load .env from the same directory as this config file
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
ZHIPUAI_API_KEY = os.environ["ZHIPUAI_API_KEY"]
ZHIPUAI_BASE_URL = os.environ.get("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "embedding-3")
_embedding_dimensions = os.environ.get("EMBEDDING_DIMENSIONS")
EMBEDDING_DIMENSIONS = int(_embedding_dimensions) if _embedding_dimensions else None
MAX_RESEARCH_ITERATIONS = int(os.environ.get("MAX_RESEARCH_ITERATIONS", "3"))
MAX_BUDGET_USD = float(os.environ.get("MAX_BUDGET_USD", "2.00"))

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "claude-opus-4-20250514")
RESEARCHER_MODEL = os.environ.get("RESEARCHER_MODEL", "claude-haiku-4-20250514")
WRITER_MODEL = os.environ.get("WRITER_MODEL", "claude-opus-4-20250514")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "claude-opus-4-20250514")

MAX_SUB_QUESTIONS = int(os.environ.get("MAX_SUB_QUESTIONS", "7"))
MAX_REVIEW_ITERATIONS = int(os.environ.get("MAX_REVIEW_ITERATIONS", "3"))
MAX_TOKENS_PER_SECTION = int(os.environ.get("MAX_TOKENS_PER_SECTION", "6000"))
RESEARCHER_MAX_TOKENS = int(os.environ.get("RESEARCHER_MAX_TOKENS", "2500"))
REVIEWER_MAX_TOKENS = int(os.environ.get("REVIEWER_MAX_TOKENS", "1500"))
ANTHROPIC_TIMEOUT_SECONDS = float(os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "90"))
ANTHROPIC_MAX_RETRIES = int(os.environ.get("ANTHROPIC_MAX_RETRIES", "4"))
MAX_TOTAL_TIME_SECONDS = 300

_client = None


def get_anthropic_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
            timeout=ANTHROPIC_TIMEOUT_SECONDS,
            max_retries=ANTHROPIC_MAX_RETRIES,
        )
    return _client
