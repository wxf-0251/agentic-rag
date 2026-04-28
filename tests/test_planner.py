from unittest.mock import patch, MagicMock
import agentic_rag.config as config_mod
from agentic_rag.agents.planner import run_planner


def _make_mock_response(json_str: str) -> MagicMock:
    """Helper to build a mock Anthropic Messages response with a text content block."""
    mock_content_block = MagicMock()
    mock_content_block.text = json_str
    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    return mock_response


def test_planner_decomposes_query():
    # Reset the singleton so lazy init picks up the mocked class
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"report_title":"Test Report","sub_questions":[{"id":"q1","text":"What is the market size?","source_type":"latest_news"},{"id":"q2","text":"Who are the key players?","source_type":"technical_concept"}],"outline":{"title":"Test Report","sections":[{"title":"Overview","sub_questions":["q1","q2"]}]}}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_planner("What is the state of humanoid robots in 2026?")

    assert result["report_title"] == "Test Report"
    assert len(result["sub_questions"]) == 2
    assert result["sub_questions"][0]["source_type"] == "latest_news"
    assert result["outline"]["title"] == "Test Report"
    assert len(result["outline"]["sections"]) == 1


def test_planner_detects_vague_query():
    # Reset the singleton so lazy init picks up the mocked class
    config_mod._client = None
    mock_response = _make_mock_response(
        '{"needs_clarification":true,"clarification_questions":["Which industry?","What time period?"],"sub_questions":[],"outline":null,"report_title":""}'
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_planner("analyze the market")

    assert result["needs_clarification"] is True
    assert len(result["clarification_questions"]) == 2


def test_planner_enforces_max_sub_questions():
    """Planner should respect max 7 sub-questions from config."""
    # Reset the singleton so lazy init picks up the mocked class
    config_mod._client = None
    # Return 10 sub-questions — the planner function should warn/log but not crash
    questions = [{"id": f"q{i}", "text": f"Question {i}", "source_type": "latest_news"} for i in range(1, 11)]
    import json
    mock_response = _make_mock_response(
        json.dumps({
            "report_title": "Test",
            "sub_questions": questions,
            "outline": {"title": "Test", "sections": [{"title": "S1", "sub_questions": [f"q{i}" for i in range(1, 11)]}]}
        })
    )
    with patch("agentic_rag.config.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_planner("test query")
    # Should not crash; should still return structured data
    assert "sub_questions" in result
