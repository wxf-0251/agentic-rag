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


def planner_node(state: AgentState) -> dict:
    result = run_planner(state["query"])
    if result.get("needs_clarification"):
        return {"progress_messages": [f"CLARIFY: {'; '.join(result.get('clarification_questions', []))}"]}
    return {
        "sub_questions": result["sub_questions"],
        "outline": result["outline"],
        "progress_messages": [f"Planned: {len(result['sub_questions'])} sub-questions, {len(result['outline']['sections'])} sections"],
    }


def researcher_fanout(state: AgentState) -> list:
    return [Send("researcher", {"sub_question": q}) for q in state["sub_questions"]]


def researcher_node(state: AgentState) -> dict:
    sub_q = state.get("sub_question")
    if sub_q:
        result = run_researcher(sub_q)
        key = result["sub_question_id"]
        new_citations = result.get("citations", [])
        existing_citations = state.get("citations", [])
        existing_results = state.get("research_results", {})
        return {
            "research_results": {**existing_results, key: result.get("findings", "")},
            "citations": existing_citations + new_citations,
            "progress_messages": [f"Researched {key}: {len(new_citations)} sources found"],
        }
    return {}


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

    result = run_writer(
        section_index=idx,
        section_outline=section_outline,
        research_results=state["research_results"],
        citations=state["citations"],
    )
    completed = state.get("completed_sections", [])
    return {
        "completed_sections": completed + [result],
        "progress_messages": [f"Written section {idx+1}/{len(outline['sections'])}: {result['title']}"],
    }


def reviewer_node(state: AgentState) -> dict:
    idx = state["current_section_idx"]
    current_section = state["completed_sections"][idx]

    result = run_reviewer(
        section=current_section,
        research_results=state["research_results"],
        citations=state["citations"],
    )
    review_history = state.get("review_history", [])
    return {
        "review_history": review_history + [result],
        "progress_messages": [f"Review {idx+1}: {result['verdict']}"],
    }


def review_gate(state: AgentState) -> Literal["writer", "researcher", "end"]:
    outline = state["outline"]
    idx = state["current_section_idx"]
    review_history = state.get("review_history", [])

    if not review_history:
        return "writer"

    last_review = review_history[-1]

    if last_review["verdict"] == "PASS":
        next_idx = idx + 1
        if next_idx >= len(outline["sections"]):
            return "end"
        return "writer"

    # FAIL: check iteration cap
    iterations = state.get("search_iteration", 0)
    max_iter = state.get("max_iterations", 3)
    if iterations >= max_iter:
        # Force accept with caveat
        next_idx = idx + 1
        if next_idx >= len(outline["sections"]):
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
