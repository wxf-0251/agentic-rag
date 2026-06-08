from typing import TypedDict, List, Optional, Literal, Annotated, NotRequired
from operator import add


def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


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
    research_results: Annotated[dict, _merge_dicts]     # {question_id: str}
    citations: Annotated[List[Citation], add]
    completed_sections: List[ReportSection]
    current_section_idx: int
    review_history: List[ReviewResult]
    search_iteration: int
    max_iterations: int
    progress_messages: Annotated[List[str], add]
    source_toggles: NotRequired[dict]
