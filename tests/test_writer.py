from unittest.mock import patch, MagicMock
import anthropic
import httpx
import agentic_rag.config as config_mod
from agentic_rag.agents.writer import run_writer


def test_writer_returns_research_based_fallback_on_connection_error():
    config_mod._client = None
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=request)
        mock_anthropic.return_value = mock_client

        result = run_writer(
            section_index=1,
            section_outline={"title": "第二章：科技股分析", "sub_questions": ["q1", "q2"]},
            research_results={
                "q1": "头部平台公司盈利能力改善，云业务收入保持增长。",
                "q2": "AI 资本开支上行会压缩短期现金流，但提升长期算力壁垒。",
            },
            citations=[
                {"id": "c1", "title": "财报来源", "url": "https://example.com/a", "snippet": "云业务收入增长"},
                {"id": "c2", "title": "行业来源", "url": "https://example.com/b", "snippet": "AI 资本开支"},
            ],
        )

    assert result["section_index"] == 1
    assert result["title"] == "第二章：科技股分析"
    assert result["writer_connection_recovered"] is True
    assert "写作模型连接失败" in result["content"]
    assert "[1]" in result["content"]


def _make_mock_response(json_str: str) -> MagicMock:
    """Helper to build a mock Anthropic Messages response with a text content block."""
    mock_content_block = MagicMock()
    mock_content_block.text = json_str
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    return mock_response


def test_writer_synthesizes_section():
    # Reset the singleton so lazy init picks up the mocked class
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"section_index":0,"title":"Market Overview","content":"The humanoid robot market is growing rapidly [1]. Key players include Tesla and Boston Dynamics [2].\\n\\n<chart type=\\"bar\\" title=\\"Market Growth\\">\\n  x: [\\"2024\\",\\"2025\\",\\"2026\\"]\\n  y: [5.2, 8.1, 13.4]\\n</chart>","citations_used":["c1","c2"]}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        section_outline = {"title": "Market Overview", "sub_questions": ["q1", "q2"]}
        research = {"q1": "Market data shows 40% CAGR.", "q2": "Tesla Optimus leads."}
        citations = [
            {"id": "c1", "title": "Report A", "url": "https://a.com", "snippet": "", "source_type": "tavily"},
            {"id": "c2", "title": "Report B", "url": "https://b.com", "snippet": "", "source_type": "tavily"}
        ]

        result = run_writer(section_index=0, section_outline=section_outline, research_results=research, citations=citations)

    assert result["section_index"] == 0
    assert "[1]" in result["content"]
    assert "<chart" in result["content"]
    assert len(result["citations_used"]) == 2


def test_writer_handles_empty_research():
    # Reset the singleton so lazy init picks up the mocked class
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"section_index":0,"title":"Risks","content":"Data on this aspect remains limited. Further research is needed.","citations_used":[]}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_writer(section_index=0, section_outline={"title": "Risks", "sub_questions": ["q1"]}, research_results={}, citations=[])

    assert "Data on this aspect remains limited" in result["content"]
    assert result["citations_used"] == []


def test_writer_returns_fallback_section_when_json_is_malformed():
    config_mod._client = None
    mock_response = _make_mock_response(
        '```json {"section_index":4,"title":"第五章：投资风险","content":"## 第五章：投资风险\\n正文引用了来源 [c1]，但 JSON 被截断'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_writer(
            section_index=4,
            section_outline={"title": "第五章：投资风险", "sub_questions": ["q1"]},
            research_results={"q1": "风险信息"},
            citations=[{"id": "c1", "title": "来源A", "url": "https://a.com", "snippet": ""}],
        )

    assert result["section_index"] == 4
    assert result["title"] == "第五章：投资风险"
    assert "第五章" in result["content"]
