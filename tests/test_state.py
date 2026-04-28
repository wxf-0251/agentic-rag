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


def test_report_section_creation():
    section = ReportSection(
        section_index=0,
        title="Overview",
        content="Market analysis with [1] citation.",
        charts=[{"type": "bar", "title": "Growth", "data": {"x": ["2024"], "y": [10]}}],
        citations_used=["c1", "c2"],
    )
    assert section["section_index"] == 0
    assert section["title"] == "Overview"
    assert "[1]" in section["content"]
    assert len(section["charts"]) == 1
    assert section["citations_used"] == ["c1", "c2"]


def test_review_result_creation():
    review = ReviewResult(
        section_index=1,
        verdict="PASS",
        critique="Well-sourced section.",
        suggested_keywords=None,
    )
    assert review["section_index"] == 1
    assert review["verdict"] == "PASS"
    assert review["suggested_keywords"] is None

    fail_review = ReviewResult(
        section_index=0,
        verdict="FAIL",
        critique="Missing citations.",
        suggested_keywords="market size 2026",
    )
    assert fail_review["verdict"] == "FAIL"
    assert fail_review["suggested_keywords"] is not None


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
    assert state["sub_questions"] == []
    assert state["citations"] == []
    assert state["completed_sections"] == []
    assert state["review_history"] == []
    assert state["progress_messages"] == []
    assert state["outline"] is None
    assert state["research_results"] == {}
