from typing import TypedDict, List, Optional, Literal


class SubQuestion(TypedDict):
    id: str
    text: str
    source_type: Literal["latest_news", "technical_concept", "financial_data", "mixed"]
    refined_keywords: Optional[str]


class Citation(TypedDict):
    id: str
    title: str
    url: str
    snippet: str
    source_type: Literal["tavily", "chroma", "yfinance"]


class ReportOutline(TypedDict):
    title: str
    sections: List[dict]   # [{"title": str, "sub_questions": List[str]}]


class ReportSection(TypedDict):
    section_index: int
    title: str
    content: str            # markdown with inline citations
    charts: List[dict]      # [{"type": str, "title": str, "data": dict}]
    citations_used: List[str]  # citation IDs


class ReviewResult(TypedDict):
    section_index: int
    verdict: Literal["PASS", "FAIL"]
    critique: str
    suggested_keywords: Optional[str]


class AgentState(TypedDict):
    query: str
    sub_questions: List[SubQuestion]
    outline: Optional[ReportOutline]
    research_results: dict              # {question_id: str}
    citations: List[Citation]
    completed_sections: List[ReportSection]
    current_section_idx: int
    review_history: List[ReviewResult]
    search_iteration: int
    max_iterations: int
    progress_messages: List[str]
