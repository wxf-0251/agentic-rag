import json
import anthropic
from agentic_rag.config import ANTHROPIC_API_KEY, WRITER_MODEL, MAX_TOKENS_PER_SECTION

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
        with open("agentic_rag/prompts/writer.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_writer(section_index: int, section_outline: dict, research_results: dict, citations: list) -> dict:
    user_content = json.dumps({
        "section_index": section_index,
        "section_outline": section_outline,
        "research_results": research_results,
        "citations": citations,
    }, ensure_ascii=False)

    response = _get_client().messages.create(
        model=WRITER_MODEL,
        max_tokens=MAX_TOKENS_PER_SECTION,
        system=[
            {"type": "text", "text": _load_prompt(),
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(response.content[0].text)
