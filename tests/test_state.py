from agentic_rag.graph.state import SubQuestion, Citation, ReportOutline, ReportSection, ReviewResult, AgentState


def test_sub_question_creation():
    sq = SubQuestion(id="q1", text="What is the market size?", source_type="latest_news", refined_keywords=None)
    assert sq["id"] == "q1"
    assert sq["source_type"] == "latest_news"
    assert sq["refined_keywords"] is None


def test_citation_creation():
    c = Citation(id="c1", title="Market Report 2026", url="https://example.com", snippet="Market size is $10B", source_type="tavily")
    assert c["id"] == "c1"
    assert c["source_type"] == "tavily"


def test_report_outline_creation():
    sections = [
        {"title": "Executive Summary", "sub_questions": ["q1", "q2"]},
        {"title": "Supply Chain Analysis", "sub_questions": ["q3"]},
    ]
    outline = ReportOutline(title="Test Report", sections=sections)
    assert len(outline["sections"]) == 2
    assert outline["sections"][0]["title"] == "Executive Summary"


def test_agent_state_defaults():
    state = AgentState(
        query="test query",
        sub_questions=[],
        outline=None,
        research_results={},
        citations=[],
        completed_sections=[],
        current_section_idx=0,
        review_history=[],
        search_iteration=0,
        max_iterations=3,
        progress_messages=[]
    )
    assert state["search_iteration"] == 0
    assert state["max_iterations"] == 3
