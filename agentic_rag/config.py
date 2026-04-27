import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
MAX_RESEARCH_ITERATIONS = int(os.environ.get("MAX_RESEARCH_ITERATIONS", "3"))
MAX_BUDGET_USD = float(os.environ.get("MAX_BUDGET_USD", "2.00"))

PLANNER_MODEL = "claude-opus-4-20250514"
RESEARCHER_MODEL = "claude-haiku-4-20250514"
WRITER_MODEL = "claude-opus-4-20250514"
REVIEWER_MODEL = "claude-opus-4-20250514"

MAX_SUB_QUESTIONS = 7
MAX_TOKENS_PER_SECTION = 4000
MAX_TOTAL_TIME_SECONDS = 300
