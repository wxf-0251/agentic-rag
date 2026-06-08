import os
import json
import re
import anthropic
from agentic_rag.agents import normalize_report_text, parse_json
from agentic_rag.config import get_anthropic_client, WRITER_MODEL, MAX_TOKENS_PER_SECTION
from agentic_rag.ui.renderer import citations_for_prompt, dedupe_citations, normalize_citation_markers

_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "writer.md")
SYSTEM_PROMPT = None


def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def _extract_json_string_value(raw: str, key: str) -> str | None:
    marker = f'"{key}"'
    start = raw.find(marker)
    if start == -1:
        return None
    colon = raw.find(":", start + len(marker))
    if colon == -1:
        return None
    quote = raw.find('"', colon + 1)
    if quote == -1:
        return None

    tail = raw[quote + 1:]
    end_markers = [f'", "{name}"' for name in ("content", "citations_used", "charts")]
    end_markers += [f'","{name}"' for name in ("content", "citations_used", "charts")]
    positions = [tail.find(marker) for marker in end_markers if tail.find(marker) != -1]
    if positions:
        return tail[:min(positions)]
    last_quote = tail.rfind('"')
    if last_quote != -1:
        return tail[:last_quote]
    return tail


def _fallback_writer_result(raw: str, section_index: int, section_outline: dict, citations: list) -> dict:
    title = _extract_json_string_value(raw, "title") or section_outline.get("title", f"Section {section_index + 1}")
    content = _extract_json_string_value(raw, "content") or raw
    content = re.sub(r"^```(?:json)?", "", content.strip())
    content = re.sub(r"```$", "", content.strip())
    content = normalize_report_text(content)
    content = normalize_citation_markers(content, citations)
    return {
        "section_index": section_index,
        "title": normalize_report_text(title),
        "content": content,
        "charts": [],
        "citations_used": [],
        "writer_parse_recovered": True,
    }


def _fallback_writer_from_research(
    section_index: int,
    section_outline: dict,
    research_results: dict,
    citations: list,
) -> dict:
    citations = dedupe_citations(citations)
    title = section_outline.get("title", f"Section {section_index + 1}")
    citation_refs = [f"[{idx}]" for idx, _ in enumerate(citations[:3], 1)]
    default_ref = citation_refs[0] if citation_refs else ""
    content_lines = [
        "写作模型连接失败，本节先根据已检索资料生成保守草稿；请在网络恢复后重新运行以获得完整论证和图表。",
    ]

    if research_results:
        content_lines.append("")
        content_lines.append("### 关键发现")
        for idx, finding in enumerate(research_results.values(), 1):
            text = normalize_report_text(str(finding)).strip()
            if not text:
                continue
            text = re.sub(r"\s+", " ", text)
            ref = citation_refs[min(idx - 1, len(citation_refs) - 1)] if citation_refs else default_ref
            content_lines.append(f"{idx}. {text[:360]} {ref}".strip())
    else:
        content_lines.append("")
        content_lines.append("当前章节没有可用研究结果，建议检查检索源和模型连接后重新生成。")

    if citations:
        content_lines.append("")
        content_lines.append("### 可用来源")
        for citation in citations[:5]:
            number = citation.get("number", "")
            source_title = citation.get("title", "Untitled source")
            snippet = normalize_report_text(citation.get("snippet", ""))
            content_lines.append(f"- [{number}] {source_title}: {snippet[:160]}")

    content = normalize_citation_markers("\n".join(content_lines), citations)
    return {
        "section_index": section_index,
        "title": normalize_report_text(title),
        "content": content,
        "charts": [],
        "citations_used": [citation.get("id", "") for citation in citations],
        "writer_connection_recovered": True,
    }


def run_writer(section_index: int, section_outline: dict, research_results: dict, citations: list) -> dict:
    prompt_citations = citations_for_prompt(citations)
    user_content = json.dumps({
        "section_index": section_index,
        "section_outline": section_outline,
        "research_results": research_results,
        "citations": prompt_citations,
    }, ensure_ascii=False)

    try:
        response = get_anthropic_client().messages.create(
            model=WRITER_MODEL,
            max_tokens=MAX_TOKENS_PER_SECTION,
            system=[
                {"type": "text", "text": _load_prompt(),
                 "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user_content}],
        )
    except (anthropic.APIConnectionError, anthropic.APITimeoutError):
        return _fallback_writer_from_research(section_index, section_outline, research_results, citations)
    raw = response.content[0].text
    try:
        result = parse_json(raw)
    except ValueError:
        return _fallback_writer_result(raw, section_index, section_outline, citations)
    if "content" in result:
        result["content"] = normalize_report_text(result["content"])
        result["content"] = normalize_citation_markers(result["content"], citations)
    return result
