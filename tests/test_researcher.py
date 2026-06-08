from unittest.mock import patch, MagicMock
import json
import anthropic
import httpx
import agentic_rag.config as config_mod
from agentic_rag.agents.researcher import _execute_search, run_researcher, route_to_tool


def _make_mock_response(json_str: str) -> MagicMock:
    """Helper to build a mock Anthropic Messages response with a text content block."""
    mock_content_block = MagicMock()
    mock_content_block.text = json_str
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    return mock_response


def test_run_researcher_returns_search_fallback_on_connection_error():
    config_mod._client = None
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    mock_search = [
        {
            "title": "Market Note",
            "url": "https://example.com/market",
            "content": "科技股估值受盈利增长和AI资本开支共同影响。",
        }
    ]

    with patch("agentic_rag.agents.researcher.tavily_search", return_value=mock_search):
        with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=request)
            mock_anthropic.return_value = mock_client

            result = run_researcher({"id": "q1", "text": "科技股估值?", "source_type": "latest_news", "refined_keywords": None})

    assert result["sub_question_id"] == "q1"
    assert "research_connection_error" in result["findings"]
    assert result["citations"][0]["url"] == "https://example.com/market"
    assert result["sufficient"] is False


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


def test_run_researcher_uses_configured_token_limit():
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"sub_question_id":"q1","findings":"Brief findings.","citations":[],"sufficient":true,"refined_keywords":null}'
    )

    with patch("agentic_rag.agents.researcher.tavily_search", return_value=[]):
        with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            run_researcher({"id": "q1", "text": "Market size?", "source_type": "latest_news", "refined_keywords": None})

    assert mock_client.messages.create.call_args.kwargs["max_tokens"] == config_mod.RESEARCHER_MAX_TOKENS


def test_run_researcher_returns_fallback_when_json_is_truncated():
    config_mod._client = None
    mock_search = [
        {
            "title": "Policy Update",
            "url": "https://example.com/policy",
            "content": "Semiconductor policy pressure remains high.",
        }
    ]
    mock_response = _make_mock_response(
        '```json\n{"sub_question_id":"q1","findings":"This response is truncated'
    )

    with patch("agentic_rag.agents.researcher.tavily_search", return_value=mock_search):
        with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.return_value = mock_client

            result = run_researcher({"id": "q1", "text": "Policy risk?", "source_type": "latest_news", "refined_keywords": None})

    assert result["sub_question_id"] == "q1"
    assert "research_parse_error" in result["findings"]
    assert result["citations"][0]["url"] == "https://example.com/policy"
    assert result["sufficient"] is False


def test_financial_data_without_ticker_uses_web_search_instead_of_yfinance():
    sub_question = {
        "id": "q1",
        "text": "主要储能电池厂商的财务表现和资本市场估值情况如何？",
        "source_type": "financial_data",
        "refined_keywords": None,
    }
    mock_search = [{"title": "Battery market report", "url": "https://example.com", "content": "Revenue data"}]

    with patch("agentic_rag.agents.researcher.fetch_stock_data") as mock_fetch:
        with patch("agentic_rag.agents.researcher.tavily_search", return_value=mock_search) as mock_tavily:
            results = _execute_search(sub_question)

    mock_fetch.assert_not_called()
    mock_tavily.assert_called_once_with(sub_question["text"])
    assert results[0]["title"] == "Battery market report"
    assert results[0]["url"] == "https://example.com"
    assert "freshness_status" in results[0]
    assert "authority_score" in results[0]


def test_financial_data_with_explicit_ticker_uses_yfinance():
    sub_question = {
        "id": "q1",
        "text": "What are $TSLA revenue, market cap, and recent stock performance?",
        "source_type": "financial_data",
        "refined_keywords": None,
    }
    mock_finance = {
        "symbol": "TSLA",
        "company_name": "Tesla Inc",
        "current_price": 250.0,
        "market_cap": 800000000000,
        "historical_prices": [],
    }

    with patch("agentic_rag.agents.researcher.fetch_stock_data", return_value=mock_finance) as mock_fetch:
        with patch("agentic_rag.agents.researcher.tavily_search") as mock_tavily:
            results = _execute_search(sub_question)

    mock_fetch.assert_called_once_with("TSLA")
    mock_tavily.assert_not_called()
    assert results[0]["title"] == "Tesla Inc"
    assert results[0]["url"] == ""
    assert results[0]["content"] == json.dumps(mock_finance)
    assert "freshness_status" in results[0]
    assert "authority_score" in results[0]


def test_execute_search_respects_disabled_source_toggles():
    sub_question = {
        "id": "q1",
        "text": "Latest market news?",
        "source_type": "latest_news",
        "refined_keywords": None,
        "source_toggles": {"tavily": False, "chroma": True, "yfinance": True},
    }

    with patch("agentic_rag.agents.researcher.tavily_search") as mock_tavily:
        results = _execute_search(sub_question)

    mock_tavily.assert_not_called()
    assert results == []


def test_execute_search_enriches_and_ranks_source_quality():
    sub_question = {
        "id": "q1",
        "text": "Latest exchange disclosures?",
        "source_type": "latest_news",
        "refined_keywords": None,
    }
    mock_search = [
        {
            "title": "Old forum post",
            "url": "https://zhuanlan.zhihu.com/p/old",
            "content": "Published on 2024-01-01. Unofficial view.",
        },
        {
            "title": "Fresh exchange disclosure",
            "url": "https://www.sse.com.cn/disclosure/listedinfo/announcement/",
            "content": "Published on 2026-05-28. Official filing.",
        },
    ]

    with patch("agentic_rag.agents.researcher.tavily_search", return_value=mock_search):
        results = _execute_search(sub_question)

    assert results[0]["title"] == "Fresh exchange disclosure"
    assert results[0]["freshness_status"] == "fresh"
    assert results[0]["authority_score"] > results[1]["authority_score"]
