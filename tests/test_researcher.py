from unittest.mock import patch, MagicMock
import agentic_rag.config as config_mod
from agentic_rag.agents.researcher import run_researcher, route_to_tool


def _make_mock_response(json_str: str) -> MagicMock:
    """Helper to build a mock Anthropic Messages response with a text content block."""
    mock_content_block = MagicMock()
    mock_content_block.text = json_str
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    return mock_response


def test_route_to_tool_returns_correct_tool():
    assert route_to_tool("latest_news") == "tavily"
    assert route_to_tool("technical_concept") == "chroma"
    assert route_to_tool("financial_data") == "yfinance"
    assert route_to_tool("mixed") == "tavily+chroma"


def test_run_researcher_with_tavily():
    # Reset the singleton so lazy init picks up the mocked class
    config_mod._client = None
    mock_search = [{"title": "Test", "url": "https://x.com", "content": "Market is $10B"}]
    mock_response = _make_mock_response(
        '{"sub_question_id":"q1","findings":"Market is $10B.","citations":[{"id":"c1","title":"Test","url":"https://x.com","snippet":"Market is $10B","source_type":"tavily"}],"sufficient":true,"refined_keywords":null}'
    )

    with patch("agentic_rag.agents.researcher.tavily_search", return_value=mock_search):
        with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            result = run_researcher({"id": "q1", "text": "Market size?", "source_type": "latest_news", "refined_keywords": None})

    assert result["sub_question_id"] == "q1"
    assert len(result["citations"]) == 1
    assert result["sufficient"] is True


def test_run_researcher_with_empty_results():
    # Reset the singleton so lazy init picks up the mocked class
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"sub_question_id":"q1","findings":"No data found.","citations":[],"sufficient":false,"refined_keywords":"broader market search"}'
    )

    with patch("agentic_rag.agents.researcher.tavily_search", return_value=[]):
        with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            result = run_researcher({"id": "q1", "text": "Market size?", "source_type": "latest_news", "refined_keywords": None})

    assert result["sufficient"] is False
    assert result["refined_keywords"] is not None
