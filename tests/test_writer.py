from unittest.mock import patch, MagicMock
import agentic_rag.agents.writer as writer_mod
from agentic_rag.agents.writer import run_writer


def _make_mock_response(json_str: str) -> MagicMock:
    """Helper to build a mock Anthropic Messages response with a text content block."""
    mock_content_block = MagicMock()
    mock_content_block.text = json_str
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    return mock_response


def test_writer_synthesizes_section():
    # Reset the module-level singleton so lazy init picks up the mocked class
    writer_mod._client = None
    mock_response = _make_mock_response(
        '{"section_index":0,"title":"Market Overview","content":"The humanoid robot market is growing rapidly [1]. Key players include Tesla and Boston Dynamics [2].\\n\\n<chart type=\\"bar\\" title=\\"Market Growth\\">\\n  x: [\\"2024\\",\\"2025\\",\\"2026\\"]\\n  y: [5.2, 8.1, 13.4]\\n</chart>","citations_used":["c1","c2"]}'
    )
    with patch("agentic_rag.agents.writer.anthropic.Anthropic") as mock_anthropic:
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
    # Reset the module-level singleton so lazy init picks up the mocked class
    writer_mod._client = None
    mock_response = _make_mock_response(
        '{"section_index":0,"title":"Risks","content":"Data on this aspect remains limited. Further research is needed.","citations_used":[]}'
    )
    with patch("agentic_rag.agents.writer.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_writer(section_index=0, section_outline={"title": "Risks", "sub_questions": ["q1"]}, research_results={}, citations=[])

    assert "Data on this aspect remains limited" in result["content"]
    assert result["citations_used"] == []
