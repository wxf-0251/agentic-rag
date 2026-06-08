from typing import Literal
import langgraph.graph as lg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langgraph.constants import END
from agentic_rag.graph.state import AgentState
from agentic_rag.agents.planner import run_planner
from agentic_rag.agents.researcher import run_researcher
from agentic_rag.agents.writer import run_writer
from agentic_rag.agents.reviewer import run_reviewer
from agentic_rag.ui.renderer import dedupe_citations, normalize_citation_markers, validate_inline_citations


def _with_source_toggles(sub_question: dict, state: AgentState) -> dict:
    source_toggles = state.get("source_toggles")
    if not source_toggles:
        return dict(sub_question)
    return {**sub_question, "source_toggles": source_toggles}


def _current_section_questions(state: AgentState, refined_keywords: str | None) -> list[dict]:
    outline = state.get("outline") or {"sections": []}
    idx = state.get("current_section_idx", 0)
    if idx >= len(outline.get("sections", [])):
        return []

    section_question_ids = set(outline["sections"][idx].get("sub_questions", []))
    questions = []
    for sub_question in state.get("sub_questions", []):
        if sub_question["id"] not in section_question_ids:
            continue
        refined = _with_source_toggles(sub_question, state)
        if refined_keywords:
            refined["refined_keywords"] = refined_keywords
        questions.append(refined)
    return questions


def planner_node(state: AgentState) -> dict:
    result = run_planner(state["query"])
    if result.get("needs_clarification"):
        return {"progress_messages": [f"CLARIFY: {'; '.join(result.get('clarification_questions', []))}"]}
    sub_questions = [
        _with_source_toggles(q, state)
        for q in result["sub_questions"]
    ]
    return {
        "sub_questions": sub_questions,
        "outline": result["outline"],
        "progress_messages": [f"Planned: {len(sub_questions)} sub-questions, {len(result['outline']['sections'])} sections"],
    }


def researcher_fanout(state: AgentState) -> list:
    return [Send("researcher", {"sub_question": q}) for q in state["sub_questions"]]


def researcher_node(state: AgentState) -> dict:
    sub_q = state.get("sub_question")
    questions = []
    if sub_q:
        questions = [_with_source_toggles(sub_q, state)]
    else:
        last_review = (state.get("review_history") or [{}])[-1]
        questions = _current_section_questions(state, last_review.get("suggested_keywords"))

    research_results = {}
    citations = []
    messages = []
    for question in questions:
        result = run_researcher(question)
        key = result["sub_question_id"]
        new_citations = result.get("citations", [])
        research_results[key] = result.get("findings", "")
        citations.extend(new_citations)
        refined = question.get("refined_keywords")
        suffix = f" using '{refined}'" if refined else ""
        messages.append(f"Researched {key}{suffix}: {len(new_citations)} sources found")

    if not research_results:
        return {}
    return {
        "research_results": research_results,
        "citations": citations,
        "progress_messages": messages,
    }


def router(state: AgentState) -> Literal["writer", "researcher"]:
    questions = state["sub_questions"]
    results = state["research_results"]
    if len(results) >= len(questions):
        return "writer"
    return "researcher"


def writer_node(state: AgentState) -> dict:
    idx = state["current_section_idx"]
    outline = state["outline"]
    section_outline = outline["sections"][idx]
    citations = dedupe_citations(state["citations"])

    result = run_writer(
        section_index=idx,
        section_outline=section_outline,
        research_results=state["research_results"],
        citations=citations,
    )
    if "content" in result:
        result["content"] = normalize_citation_markers(result["content"], citations)
    sections = list(state.get("completed_sections", []))
    if idx < len(sections):
        sections[idx] = result
    else:
        sections.append(result)
    return {
        "completed_sections": sections,
        "progress_messages": [f"Written section {idx+1}/{len(outline['sections'])}: {result['title']}"],
    }


def reviewer_node(state: AgentState) -> dict:
    idx = state["current_section_idx"]
    current_section = state["completed_sections"][idx]
    citations = dedupe_citations(state["citations"])

    citation_check = validate_inline_citations(current_section.get("content", ""), citations)
    if citation_check["valid"]:
        result = run_reviewer(
            section=current_section,
            research_results=state["research_results"],
            citations=citations,
        )
    else:
        result = {
            "verdict": "FAIL",
            "critique": f"Missing citation references: {citation_check['missing_numbers']}",
            "issues": ["missing_citation_reference"],
            "suggested_keywords": current_section.get("title") or state.get("query"),
        }

    verdict = result["verdict"]
    iteration = state.get("search_iteration", 0)
    max_iter = state.get("max_iterations", 3)
    history = list(state.get("review_history", []))

    updates = {"review_history": history + [result], "progress_messages": [f"Review {idx+1}: {verdict}"]}

    if verdict == "PASS":
        updates["search_iteration"] = 0
        updates["current_section_idx"] = idx + 1
    elif iteration + 1 >= max_iter:
        updates["search_iteration"] = 0
        updates["current_section_idx"] = idx + 1
        updates["progress_messages"] = [f"Review {idx+1}: FORCED ACCEPT (max {max_iter} iterations)"]
    else:
        updates["search_iteration"] = iteration + 1

    return updates


def review_gate(state: AgentState) -> Literal["writer", "researcher", "end"]:
    outline = state["outline"]
    idx = state["current_section_idx"]
    review_history = state.get("review_history", [])

    if not review_history:
        return "writer"

    last_review = review_history[-1]

    if last_review["verdict"] == "PASS":
        if idx >= len(outline["sections"]):
            return "end"
        return "writer"

    # FAIL verdict: search_iteration 0 means force-accepted
    if state.get("search_iteration", 0) == 0:
        if idx >= len(outline["sections"]):
            return "end"
        return "writer"

    return "researcher"


def build_graph():
    graph = lg.StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", researcher_fanout, {"researcher": "researcher"})
    graph.add_conditional_edges("researcher", router, {"writer": "writer", "researcher": "researcher"})
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", review_gate, {
        "writer": "writer",
        "researcher": "researcher",
        "end": END,
    })

    return graph.compile(checkpointer=MemorySaver())
