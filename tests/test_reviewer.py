from unittest.mock import patch, MagicMock
import anthropic
import httpx
import agentic_rag.config as config_mod
from agentic_rag.agents.reviewer import run_reviewer


def _make_mock_response(json_str: str) -> MagicMock:
    """Helper to build a mock Anthropic Messages response with a text content block."""
    mock_content_block = MagicMock()
    mock_content_block.text = json_str
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    return mock_response


def test_reviewer_skips_auto_review_on_connection_error():
    config_mod._client = None
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")

    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.APIConnectionError(request=request)
        mock_anthropic.return_value = mock_client

        result = run_reviewer({"content": "OK"}, {}, [])

    assert result["verdict"] == "PASS"
    assert "review_connection_error" in result["issues"]
    assert "自动审核模型连接失败" in result["critique"]


def test_reviewer_passes_good_section():
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"verdict":"PASS","critique":"Well-sourced and logically sound.","issues":[],"suggested_keywords":null}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
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
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"verdict":"FAIL","critique":"Claims without citations.","issues":["uncited_claim"],"suggested_keywords":"humanoid robot market share 2026"}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        section = {"section_index": 0, "title": "Overview", "content": "The market is huge. Many players exist.", "citations_used": [], "charts": []}
        result = run_reviewer(section, {}, [])

    assert result["verdict"] == "FAIL"
    assert result["suggested_keywords"] is not None


def test_reviewer_fails_before_llm_when_numeric_claim_not_supported():
    config_mod._client = None
    section = {
        "section_index": 0,
        "title": "Financials",
        "content": "Revenue reached 50 billion yuan in 2026 [1].",
        "citations_used": ["c1"],
        "charts": [],
    }
    citations = [
        {
            "id": "c1",
            "title": "Company filing",
            "url": "https://example.com/filing",
            "snippet": "The company reported revenue of 40 billion yuan in 2026.",
            "source_type": "tavily",
        }
    ]

    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        result = run_reviewer(section, {}, citations)

    mock_anthropic.assert_not_called()
    assert result["verdict"] == "FAIL"
    assert "fact_verification_failed" in result["issues"]
    assert result["suggested_keywords"] == "Financials"


def test_reviewer_fails_current_claims_backed_only_by_stale_sources():
    config_mod._client = None
    section = {
        "section_index": 0,
        "title": "Current Market View",
        "content": "Current market revenue reached 50 billion yuan in 2026 [1].",
        "citations_used": ["c1"],
        "charts": [],
    }
    citations = [
        {
            "id": "c1",
            "title": "Old market post",
            "url": "https://example.com/old",
            "snippet": "Current market revenue reached 50 billion yuan in 2026.",
            "source_type": "latest_news",
            "freshness_status": "stale",
            "published_date": "2024-01-01",
        }
    ]

    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        result = run_reviewer(section, {}, citations)

    mock_anthropic.assert_not_called()
    assert result["verdict"] == "FAIL"
    assert "stale_source_for_current_claim" in result["issues"]


def test_reviewer_uses_configured_token_limit():
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"verdict":"PASS","critique":"Good.","issues":[],"suggested_keywords":null}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        run_reviewer({"content": "OK"}, {}, [])

    assert mock_client.messages.create.call_args.kwargs["max_tokens"] == config_mod.REVIEWER_MAX_TOKENS


def test_reviewer_normalizes_chinese_fail_verdict():
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"verdict":"失败","critique":"存在未引用声明。","issues":["uncited_claim"],"suggested_keywords":"科技股估值"}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_reviewer({"content": "OK"}, {}, [])

    assert result["verdict"] == "FAIL"


def test_reviewer_returns_fail_when_json_is_truncated():
    config_mod._client = None
    mock_response = _make_mock_response('```json\n{"verdict":"FAIL","critique":"bad","suggested_keywords":"未闭合')
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_reviewer({"content": "OK"}, {}, [])

    assert result["verdict"] == "FAIL"
    assert "review_parse_error" in result["issues"]
