from agentic_rag.ui.stream_events import extract_node_update


def test_extract_node_update_returns_name_and_output():
    assert extract_node_update({"planner": {"outline": {}}}) == ("planner", {"outline": {}})


def test_extract_node_update_skips_none_output():
    assert extract_node_update({"researcher": None}) is None


def test_extract_node_update_skips_empty_event():
    assert extract_node_update({}) is None
