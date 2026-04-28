import os
from dotenv import load_dotenv
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
MAX_RESEARCH_ITERATIONS = int(os.environ.get("MAX_RESEARCH_ITERATIONS", "3"))
MAX_BUDGET_USD = float(os.environ.get("MAX_BUDGET_USD", "2.00"))

PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "claude-opus-4-20250514")
RESEARCHER_MODEL = os.environ.get("RESEARCHER_MODEL", "claude-haiku-4-20250514")
WRITER_MODEL = os.environ.get("WRITER_MODEL", "claude-opus-4-20250514")
REVIEWER_MODEL = os.environ.get("REVIEWER_MODEL", "claude-opus-4-20250514")

MAX_SUB_QUESTIONS = 7
MAX_TOKENS_PER_SECTION = 4000
MAX_TOTAL_TIME_SECONDS = 300

_client = None


def get_anthropic_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, base_url=ANTHROPIC_BASE_URL)
    return _client
