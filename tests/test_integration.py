from unittest.mock import patch
from agentic_rag.graph.workflow import build_graph
from agentic_rag.graph.state import AgentState


def test_full_graph_execution_with_mocks():
    """End-to-end test with all external APIs mocked."""
    mock_planner_result = {
        "report_title": "Test Report",
        "sub_questions": [
            {"id": "q1", "text": "Test question 1?", "source_type": "latest_news"},
            {"id": "q2", "text": "Test question 2?", "source_type": "technical_concept"},
        ],
        "outline": {
            "title": "Test Report",
            "sections": [
                {"title": "Section 1", "sub_questions": ["q1", "q2"]},
            ],
        },
    }

    mock_researcher_result = {
        "sub_question_id": "q1",
        "findings": "Test findings.",
        "citations": [{"id": "c1", "title": "Source", "url": "https://x.com", "snippet": "data", "source_type": "tavily"}],
        "sufficient": True,
        "refined_keywords": None,
    }

    mock_writer_result = {
        "section_index": 0,
        "title": "Section 1",
        "content": "Test content with citation [1].",
        "citations_used": ["c1"],
    }

    mock_reviewer_result = {
        "verdict": "PASS",
        "critique": "Good.",
        "issues": [],
        "suggested_keywords": None,
    }

    def mock_planner(*args, **kwargs):
        return mock_planner_result

    def mock_researcher(sub_q):
        return {**mock_researcher_result, "sub_question_id": sub_q["id"]}

    def mock_writer(*args, **kwargs):
        return mock_writer_result

    def mock_reviewer(*args, **kwargs):
        return mock_reviewer_result

    with patch("agentic_rag.graph.workflow.run_planner", side_effect=mock_planner):
        with patch("agentic_rag.graph.workflow.run_researcher", side_effect=mock_researcher):
            with patch("agentic_rag.graph.workflow.run_writer", side_effect=mock_writer):
                with patch("agentic_rag.graph.workflow.run_reviewer", side_effect=mock_reviewer):
                    graph = build_graph()
                    initial_state: AgentState = {
                        "query": "test",
                        "sub_questions": [],
                        "outline": None,
                        "research_results": {},
                        "citations": [],
                        "completed_sections": [],
                        "current_section_idx": 0,
                        "review_history": [],
                        "search_iteration": 0,
                        "max_iterations": 3,
                        "progress_messages": [],
                    }
                    config = {"configurable": {"thread_id": "integration-test"}}
                    events = list(graph.stream(initial_state, config, stream_mode="updates"))

    # Verify events were emitted
    assert len(events) > 0
    node_names = [list(e.keys())[0] for e in events]
    assert "planner" in node_names
    assert "researcher" in node_names
    assert "writer" in node_names
    assert "reviewer" in node_names


def test_reviewer_loop_respects_max_iterations():
    """Verify the Reviewer→Researcher loop stops after max_iterations."""
    mock_planner_result = {
        "report_title": "Test",
        "sub_questions": [{"id": "q1", "text": "Q1?", "source_type": "latest_news"}],
        "outline": {"title": "Test", "sections": [{"title": "S1", "sub_questions": ["q1"]}]},
    }

    mock_researcher_result = {
        "sub_question_id": "q1",
        "findings": "Findings.",
        "citations": [],
        "sufficient": True,
        "refined_keywords": None,
    }

    mock_writer_result = {
        "section_index": 0,
        "title": "S1",
        "content": "Content.",
        "citations_used": [],
    }

    fail_count = [0]  # mutable counter

    def mock_reviewer(*args, **kwargs):
        fail_count[0] += 1
        # Always FAIL — tests that max_iterations stops the loop
        return {
            "verdict": "FAIL",
            "critique": "Not good enough.",
            "issues": ["uncited_claim"],
            "suggested_keywords": "refined search",
        }

    with patch("agentic_rag.graph.workflow.run_planner", return_value=mock_planner_result):
        with patch("agentic_rag.graph.workflow.run_researcher", side_effect=lambda sq: {**mock_researcher_result, "sub_question_id": sq["id"]}):
            with patch("agentic_rag.graph.workflow.run_writer", return_value=mock_writer_result):
                with patch("agentic_rag.graph.workflow.run_reviewer", side_effect=mock_reviewer):
                    graph = build_graph()
                    initial_state: AgentState = {
                        "query": "test",
                        "sub_questions": [],
                        "outline": None,
                        "research_results": {},
                        "citations": [],
                        "completed_sections": [],
                        "current_section_idx": 0,
                        "review_history": [],
                        "search_iteration": 0,
                        "max_iterations": 3,
                        "progress_messages": [],
                    }
                    config = {"configurable": {"thread_id": "loop-test"}}
                    events = list(graph.stream(initial_state, config, stream_mode="updates"))

    # max_iterations=3 means the reviewer-loop runs at most 3 times before forcing accept
    assert fail_count[0] <= 3, f"Reviewer called {fail_count[0]} times, should be capped at 3"


def test_reviewer_fail_triggers_research_with_suggested_keywords():
    """A failed review should re-search the current section with reviewer keywords."""
    mock_planner_result = {
        "report_title": "Test",
        "sub_questions": [{"id": "q1", "text": "Original query?", "source_type": "latest_news"}],
        "outline": {"title": "Test", "sections": [{"title": "S1", "sub_questions": ["q1"]}]},
    }
    writer_result = {
        "section_index": 0,
        "title": "S1",
        "content": "Uncited content.",
        "citations_used": [],
    }
    reviewer_results = [
        {
            "verdict": "FAIL",
            "critique": "Needs stronger evidence.",
            "issues": ["insufficient_sources"],
            "suggested_keywords": "better evidence query",
        },
        {
            "verdict": "PASS",
            "critique": "Improved.",
            "issues": [],
            "suggested_keywords": None,
        },
    ]
    researched_questions = []

    def mock_researcher(sub_q):
        researched_questions.append(dict(sub_q))
        return {
            "sub_question_id": sub_q["id"],
            "findings": f"Findings for {sub_q.get('refined_keywords') or sub_q['text']}",
            "citations": [],
            "sufficient": True,
            "refined_keywords": sub_q.get("refined_keywords"),
        }

    with patch("agentic_rag.graph.workflow.run_planner", return_value=mock_planner_result):
        with patch("agentic_rag.graph.workflow.run_researcher", side_effect=mock_researcher):
            with patch("agentic_rag.graph.workflow.run_writer", return_value=writer_result):
                with patch("agentic_rag.graph.workflow.run_reviewer", side_effect=reviewer_results):
                    graph = build_graph()
                    initial_state: AgentState = {
                        "query": "test",
                        "sub_questions": [],
                        "outline": None,
                        "research_results": {},
                        "citations": [],
                        "completed_sections": [],
                        "current_section_idx": 0,
                        "review_history": [],
                        "search_iteration": 0,
                        "max_iterations": 3,
                        "progress_messages": [],
                    }
                    config = {"configurable": {"thread_id": "re-search-test"}}
                    list(graph.stream(initial_state, config, stream_mode="updates"))

    assert len(researched_questions) == 2
    assert researched_questions[1]["refined_keywords"] == "better evidence query"


def test_workflow_rejects_missing_inline_citation_before_llm_review():
    mock_planner_result = {
        "report_title": "Test",
        "sub_questions": [{"id": "q1", "text": "Q1?", "source_type": "latest_news"}],
        "outline": {"title": "Test", "sections": [{"title": "S1", "sub_questions": ["q1"]}]},
    }
    mock_researcher_result = {
        "sub_question_id": "q1",
        "findings": "Findings.",
        "citations": [{"id": "c1", "title": "Source", "url": "https://x.com", "snippet": "data", "source_type": "tavily"}],
        "sufficient": True,
        "refined_keywords": None,
    }
    mock_writer_result = {
        "section_index": 0,
        "title": "S1",
        "content": "This cites a missing source [2].",
        "citations_used": ["c2"],
    }

    with patch("agentic_rag.graph.workflow.run_planner", return_value=mock_planner_result):
        with patch("agentic_rag.graph.workflow.run_researcher", return_value=mock_researcher_result):
            with patch("agentic_rag.graph.workflow.run_writer", return_value=mock_writer_result):
                with patch(
                    "agentic_rag.graph.workflow.run_reviewer",
                    return_value={"verdict": "PASS", "critique": "ok", "issues": [], "suggested_keywords": None},
                ) as mock_reviewer:
                    graph = build_graph()
                    initial_state: AgentState = {
                        "query": "test",
                        "sub_questions": [],
                        "outline": None,
                        "research_results": {},
                        "citations": [],
                        "completed_sections": [],
                        "current_section_idx": 0,
                        "review_history": [],
                        "search_iteration": 0,
                        "max_iterations": 1,
                        "progress_messages": [],
                    }
                    config = {"configurable": {"thread_id": "citation-validation-test"}}
                    list(graph.stream(initial_state, config, stream_mode="updates"))

    mock_reviewer.assert_not_called()
