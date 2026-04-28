import os
import json
from agentic_rag.config import get_anthropic_client, WRITER_MODEL, MAX_TOKENS_PER_SECTION

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "writer.md")
SYSTEM_PROMPT = None


def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open(_PROMPT_PATH, "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_writer(section_index: int, section_outline: dict, research_results: dict, citations: list) -> dict:
    user_content = json.dumps({
        "section_index": section_index,
        "section_outline": section_outline,
        "research_results": research_results,
        "citations": citations,
    }, ensure_ascii=False)

    response = get_anthropic_client().messages.create(
        model=WRITER_MODEL,
        max_tokens=MAX_TOKENS_PER_SECTION,
        system=[
            {"type": "text", "text": _load_prompt(),
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(response.content[0].text)
