import json
from typing import Optional
import anthropic
from agentic_rag.config import ANTHROPIC_API_KEY, PLANNER_MODEL, MAX_SUB_QUESTIONS

_client = None
SYSTEM_PROMPT = None


def _get_client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open("agentic_rag/prompts/planner.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_planner(query: str) -> dict:
    response = _get_client().messages.create(
        model=PLANNER_MODEL,
        max_tokens=2000,
        system=[
            {"type": "text", "text": _load_prompt(),
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": query}],
    )
    raw = response.content[0].text
    result = json.loads(raw)

    # Enforce max sub-questions
    if len(result.get("sub_questions", [])) > MAX_SUB_QUESTIONS:
        result["sub_questions"] = result["sub_questions"][:MAX_SUB_QUESTIONS]

    return result
