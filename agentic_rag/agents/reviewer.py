import json
import anthropic
from agentic_rag.config import ANTHROPIC_API_KEY, REVIEWER_MODEL

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
        with open("agentic_rag/prompts/reviewer.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_reviewer(section: dict, research_results: dict, citations: list) -> dict:
    user_content = json.dumps({
        "section": section,
        "research_results": research_results,
        "citations": citations,
    }, ensure_ascii=False)

    response = _get_client().messages.create(
        model=REVIEWER_MODEL,
        max_tokens=1000,
        system=[
            {"type": "text", "text": _load_prompt(),
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(response.content[0].text)
