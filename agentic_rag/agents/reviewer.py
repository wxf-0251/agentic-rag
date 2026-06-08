import os
import json
import anthropic
from agentic_rag.agents import normalize_report_text, parse_json
from agentic_rag.config import get_anthropic_client, REVIEWER_MODEL, REVIEWER_MAX_TOKENS
from agentic_rag.tools.fact_verifier import verify_numeric_claims
from agentic_rag.ui.renderer import dedupe_citations

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "reviewer.md")
SYSTEM_PROMPT = None


def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def _normalize_verdict(value: object) -> str:
    verdict = str(value or "").strip().upper()
    if verdict in ("\u5931\u8d25", "\u4e0d\u901a\u8fc7", "FAILURE", "FAILED"):
        return "FAIL"
    if verdict in ("\u901a\u8fc7", "PASS", "PASSED"):
        return "PASS"
    return verdict


def _uses_stale_current_sources(content: str, citations: list) -> bool:
    content_lower = content.lower()
    current_terms = (
        "current", "latest", "recent", "2025", "2026",
        "当前", "最新", "近期", "今年", "本年", "投资", "估值", "财务", "政策",
    )
    if not any(term in content_lower for term in current_terms):
        return False
    for citation in dedupe_citations(citations):
        source_type = citation.get("source_type", "")
        if citation.get("freshness_status") == "stale" and source_type in ("latest_news", "financial_data", "mixed", "tavily"):
            return True
    return False


def run_reviewer(section: dict, research_results: dict, citations: list) -> dict:
    fact_check = verify_numeric_claims(section.get("content", ""), citations)
    if not fact_check["valid"]:
        values = ", ".join(issue["value"] for issue in fact_check["issues"][:5])
        return {
            "verdict": "FAIL",
            "critique": f"Numeric fact verification failed for values: {values}. Check whether each number is present in the cited source snippet.",
            "issues": ["fact_verification_failed"],
            "suggested_keywords": section.get("title"),
        }

    if _uses_stale_current_sources(section.get("content", ""), citations):
        return {
            "verdict": "FAIL",
            "critique": "Current or investment-sensitive claims are backed by stale sources. Re-run search with newer authoritative sources.",
            "issues": ["stale_source_for_current_claim"],
            "suggested_keywords": section.get("title"),
        }

    user_content = json.dumps({
        "section": section,
        "research_results": research_results,
        "citations": citations,
    }, ensure_ascii=False)

    try:
        response = get_anthropic_client().messages.create(
            model=REVIEWER_MODEL,
            max_tokens=REVIEWER_MAX_TOKENS,
            system=[
                {"type": "text", "text": _load_prompt(),
                 "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user_content}],
        )
    except (anthropic.APIConnectionError, anthropic.APITimeoutError):
        return {
            "verdict": "PASS",
            "critique": "自动审核模型连接失败，已跳过本轮自动审核；请人工复核该章节事实、引用和逻辑。",
            "issues": ["review_connection_error"],
            "suggested_keywords": None,
        }

    raw_text = response.content[0].text
    try:
        result = parse_json(raw_text)
    except ValueError:
        return {
            "verdict": "FAIL",
            "critique": normalize_report_text(raw_text[:500]),
            "issues": ["review_parse_error"],
            "suggested_keywords": None,
        }

    result["verdict"] = _normalize_verdict(result.get("verdict"))
    if "critique" in result and isinstance(result["critique"], str):
        result["critique"] = normalize_report_text(result["critique"])
    return result
