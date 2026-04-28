from unittest.mock import patch, MagicMock
import agentic_rag.agents.reviewer as reviewer_mod
from agentic_rag.agents.reviewer import run_reviewer


def _make_mock_response(json_str: str) -> MagicMock:
    """Helper to build a mock Anthropic Messages response with a text content block."""
    mock_content_block = MagicMock()
    mock_content_block.text = json_str
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    return mock_response


def test_reviewer_passes_good_section():
    reviewer_mod._client = None
    mock_response = _make_mock_response(
        '{"verdict":"PASS","critique":"Well-sourced and logically sound.","issues":[],"suggested_keywords":null}'
    )
    with patch("agentic_rag.agents.reviewer.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        section = {"section_index": 0, "title": "Overview", "content": "Market grows 40% [1].", "citations_used": ["c1"], "charts": []}
        research = {"q1": "Market grows at 40% CAGR."}
        citations = [{"id": "c1", "title": "Source", "url": "https://x.com", "snippet": "40% CAGR", "source_type": "tavily"}]

        result = run_reviewer(section, research, citations)

    assert result["verdict"] == "PASS"
    assert result["issues"] == []


def test_reviewer_fails_uncited_claims():
    reviewer_mod._client = None
    mock_response = _make_mock_response(
        '{"verdict":"FAIL","critique":"Claims without citations.","issues":["uncited_claim"],"suggested_keywords":"humanoid robot market share 2026"}'
    )
    with patch("agentic_rag.agents.reviewer.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        section = {"section_index": 0, "title": "Overview", "content": "The market is huge. Many players exist.", "citations_used": [], "charts": []}
        result = run_reviewer(section, {}, [])

    assert result["verdict"] == "FAIL"
    assert result["suggested_keywords"] is not None
