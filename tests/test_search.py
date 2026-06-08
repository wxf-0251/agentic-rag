from unittest.mock import patch
from agentic_rag.tools.search import tavily_search, SearchResult


def test_tavily_search_returns_structured_results():
    mock_response = {
        "results": [
            {"title": "Humanoid Robot Market 2026", "url": "https://example.com/1", "content": "The market is growing rapidly...", "published_date": "2026-05-18"},
            {"title": "Supply Chain Analysis", "url": "https://example.com/2", "content": "Key suppliers include..."},
        ]
    }
    with patch("agentic_rag.tools.search._client") as mock_client:
        mock_client.search.return_value = mock_response
        results = tavily_search("humanoid robot supply chain 2026", max_results=5)

    assert len(results) == 2
    assert results[0]["title"] == "Humanoid Robot Market 2026"
    assert results[0]["url"] == "https://example.com/1"
    assert "growing rapidly" in results[0]["content"]
    assert results[0]["published_date"] == "2026-05-18"


def test_tavily_search_handles_empty_results():
    with patch("agentic_rag.tools.search._client") as mock_client:
        mock_client.search.return_value = {"results": []}
        results = tavily_search("xyzabc123nonexistent", max_results=5)

    assert results == []


def test_tavily_search_handles_api_error():
    with patch("agentic_rag.tools.search._client") as mock_client:
        mock_client.search.side_effect = Exception("Rate limit exceeded")
        results = tavily_search("test query", max_results=5)

    assert results == []  # graceful degradation
