import json
import os
import re

import anthropic

from agentic_rag.agents import normalize_report_text, parse_json
from agentic_rag.config import get_anthropic_client, RESEARCHER_MAX_TOKENS, RESEARCHER_MODEL
from agentic_rag.tools.chroma_store import chroma_search
from agentic_rag.tools.finance import fetch_stock_data
from agentic_rag.tools.search import tavily_search
from agentic_rag.tools.source_quality import rank_sources

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "researcher.md")
SYSTEM_PROMPT = None


def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
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


def _extract_stock_symbol(query: str) -> str | None:
    explicit_symbol = re.search(r"\$([A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b", query)
    if explicit_symbol:
        return explicit_symbol.group(1)

    exchange_symbol = re.search(r"\b(\d{4,6}\.(?:HK|SH|SS|SZ)|[A-Z]{1,5}\.(?:HK|SH|SS|SZ|US))\b", query)
    if exchange_symbol:
        return exchange_symbol.group(1)

    labeled_symbol = re.search(r"(?i)\b(?:ticker|symbol)\s*[:：]?\s*([A-Z]{1,5}(?:\.[A-Z]{1,3})?)\b", query)
    if labeled_symbol:
        return labeled_symbol.group(1).upper()

    return None


def _execute_search(sub_question: dict) -> list:
    source_type = sub_question["source_type"]
    query = sub_question.get("refined_keywords") or sub_question["text"]
    source_toggles = sub_question.get("source_toggles") or {}
    use_tavily = source_toggles.get("tavily", True)
    use_chroma = source_toggles.get("chroma", True)
    use_yfinance = source_toggles.get("yfinance", True)
    results = []

    if source_type in ("latest_news", "mixed") and use_tavily:
        results.extend(tavily_search(query))
    if source_type in ("technical_concept", "mixed") and use_chroma:
        results.extend(chroma_search(query))
    if source_type == "financial_data":
        symbol = _extract_stock_symbol(query)
        if symbol and use_yfinance:
            finance_result = fetch_stock_data(symbol)
            if finance_result:
                results.append({"title": finance_result["company_name"], "url": "", "content": json.dumps(finance_result)})
        elif not symbol and use_tavily:
            results.extend(tavily_search(query))

    return rank_sources(results, source_type=source_type)


def _compact_search_results(search_results: list, max_results: int = 5, max_content_chars: int = 900) -> list:
    compacted = []
    for result in search_results[:max_results]:
        compacted.append({
            "title": normalize_report_text(result.get("title", "")),
            "url": result.get("url", ""),
            "source": (result.get("metadata") or {}).get("source", ""),
            "content": normalize_report_text(result.get("content", ""))[:max_content_chars],
            "published_date": result.get("published_date", ""),
            "freshness_status": result.get("freshness_status", "unknown"),
            "authority_score": result.get("authority_score", 0),
        })
    return compacted


def _fallback_research_result(sub_question: dict, search_results: list, reason: str = "parse") -> dict:
    citations = []
    snippets = []
    for idx, result in enumerate(search_results[:5], 1):
        title = normalize_report_text(result.get("title", "Untitled"))
        url = result.get("url", "")
        content = normalize_report_text(result.get("content", ""))
        snippet = content[:220]
        citations.append({
            "id": f"c{idx}",
            "title": title,
            "url": url,
            "snippet": snippet,
            "source_type": sub_question.get("source_type", "tavily"),
            "published_date": result.get("published_date", ""),
            "freshness_status": result.get("freshness_status", "unknown"),
            "authority_score": result.get("authority_score", 0),
        })
        if snippet:
            snippets.append(f"{title}: {snippet}")

    if reason == "connection":
        prefix = "research_connection_error: research model connection failed; using ranked search results as a conservative summary."
    else:
        prefix = "research_parse_error: research model returned incomplete JSON; using ranked search results as a conservative summary."

    if snippets:
        findings = prefix + " " + " ".join(snippets)
    else:
        findings = prefix + " No usable sources were found for this sub-question."

    return {
        "sub_question_id": sub_question["id"],
        "findings": findings,
        "citations": citations,
        "sufficient": False,
        "refined_keywords": sub_question.get("refined_keywords") or sub_question.get("text"),
    }


def _merge_citation_quality(citations: list, search_results: list) -> list:
    by_url = {result.get("url", ""): result for result in search_results if result.get("url")}
    by_title = {result.get("title", ""): result for result in search_results if result.get("title")}
    enriched = []
    for citation in citations:
        quality = by_url.get(citation.get("url", "")) or by_title.get(citation.get("title", "")) or {}
        item = dict(citation)
        item.setdefault("published_date", quality.get("published_date", ""))
        item.setdefault("freshness_status", quality.get("freshness_status", "unknown"))
        item.setdefault("authority_score", quality.get("authority_score", 0))
        enriched.append(item)
    return enriched


def run_researcher(sub_question: dict) -> dict:
    search_results = _execute_search(sub_question)
    compact_results = _compact_search_results(search_results)

    user_content = (
        f"Sub-question ID: {sub_question['id']}\n"
        f"Sub-question: {sub_question['text']}\n\n"
        f"Search results:\n{json.dumps(compact_results, ensure_ascii=False)}"
    )

    try:
        response = get_anthropic_client().messages.create(
            model=RESEARCHER_MODEL,
            max_tokens=RESEARCHER_MAX_TOKENS,
            system=[{"type": "text", "text": _load_prompt()}],
            messages=[{"role": "user", "content": user_content}],
        )
    except (anthropic.APIConnectionError, anthropic.APITimeoutError):
        return _fallback_research_result(sub_question, compact_results, reason="connection")

    try:
        result = parse_json(response.content[0].text)
    except ValueError:
        return _fallback_research_result(sub_question, compact_results)

    if "findings" in result and isinstance(result["findings"], str):
        result["findings"] = normalize_report_text(result["findings"])
    if "citations" in result and isinstance(result["citations"], list):
        result["citations"] = _merge_citation_quality(result["citations"], search_results)
    return result
