import json
from agentic_rag.config import get_anthropic_client, RESEARCHER_MODEL
from agentic_rag.tools.search import tavily_search
from agentic_rag.tools.chroma_store import chroma_search
from agentic_rag.tools.finance import fetch_stock_data

SYSTEM_PROMPT = None


def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open("agentic_rag/prompts/researcher.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def route_to_tool(source_type: str) -> str:
    mapping = {
        "latest_news": "tavily",
        "technical_concept": "chroma",
        "financial_data": "yfinance",
        "mixed": "tavily+chroma",
    }
    return mapping.get(source_type, "tavily")


def _execute_search(sub_question: dict) -> list:
    source_type = sub_question["source_type"]
    query = sub_question.get("refined_keywords") or sub_question["text"]
    results = []

    if source_type in ("latest_news", "mixed"):
        results.extend(tavily_search(query))
    if source_type in ("technical_concept", "mixed"):
        results.extend(chroma_search(query))
    if source_type == "financial_data":
        finance_result = fetch_stock_data(query)
        if finance_result:
            results.append({"title": finance_result["company_name"], "url": "", "content": json.dumps(finance_result)})

    return results


def run_researcher(sub_question: dict) -> dict:
    search_results = _execute_search(sub_question)

    user_content = f"Sub-question: {sub_question['text']}\n\nSearch results:\n{json.dumps(search_results, ensure_ascii=False)}"

    response = get_anthropic_client().messages.create(
        model=RESEARCHER_MODEL,
        max_tokens=1500,
        system=[{"type": "text", "text": _load_prompt()}],
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(response.content[0].text)
