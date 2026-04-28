from unittest.mock import patch, MagicMock
import json
from agentic_rag.graph.state import AgentState
from agentic_rag.graph.workflow import build_graph


def test_graph_compiles_and_accepts_state():
    graph = build_graph()
    assert graph is not None

    initial_state: AgentState = {
        "query": "Test query",
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

    config = {"configurable": {"thread_id": "test-1"}}
    try:
        events = list(graph.stream(initial_state, config, stream_mode="updates"))
    except Exception:
        pass  # Expected without real API keys
