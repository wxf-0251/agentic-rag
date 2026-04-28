import os
import json
from agentic_rag.config import get_anthropic_client, PLANNER_MODEL, MAX_SUB_QUESTIONS

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "planner.md")
SYSTEM_PROMPT = None


def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open(_PROMPT_PATH, "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_planner(query: str) -> dict:
    response = get_anthropic_client().messages.create(
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
