"""CLI tool to run a full research report from the command line."""
import sys
import os
import json

# Force UTF-8 stdout/stderr so titles/snippets containing zero-width spaces,
# CJK chars, or other non-GBK characters don't crash print() on Windows.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _safe(text) -> str:
    """Return a string safe to print on the current stdout encoding."""
    s = str(text) if text is not None else ""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        s.encode(enc)
        return s
    except (UnicodeEncodeError, LookupError):
        return s.encode(enc, errors="replace").decode(enc, errors="replace")


_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from agentic_rag.graph.workflow import build_graph
from agentic_rag.graph.state import AgentState
from agentic_rag.config import MAX_REVIEW_ITERATIONS
from agentic_rag.ui.renderer import clean_markdown


import builtins as _builtins


def _print(*args, **kwargs):
    """print wrapper that sanitizes each arg through _safe()."""
    _builtins.print(*[_safe(a) for a in args], **kwargs)


def run_report(query: str):
    _print(f"\n{'='*70}")
    _print(f"QUERY: {query}")
    _print(f"{'='*70}\n")

    graph = build_graph()
    initial_state: AgentState = {
        "query": query,
        "sub_questions": [],
        "outline": None,
        "research_results": {},
        "citations": [],
        "completed_sections": [],
        "current_section_idx": 0,
        "review_history": [],
        "search_iteration": 0,
        "max_iterations": MAX_REVIEW_ITERATIONS,
        "progress_messages": [],
    }
    config = {"configurable": {"thread_id": f"cli-{hash(query) % 10000}"}}

    for event in graph.stream(initial_state, config, stream_mode="updates"):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        if node_output is None:
            continue  # router/gate edges produce no state updates

        if node_name == "planner":
            outline = node_output.get("outline", {})
            _print(f"[Planner] {node_output.get('progress_messages', [''])[0]}")
            if outline:
                _print(f"  Title: {outline.get('title', 'N/A')}")
                for i, sec in enumerate(outline.get("sections", [])):
                    _print(f"  Section {i+1}: {sec['title']} (Q: {', '.join(sec['sub_questions'])})")

        elif node_name == "researcher":
            msgs = node_output.get("progress_messages", [])
            for msg in msgs:
                _print(f"[Researcher] {msg}")
            citations = node_output.get("citations", [])
            if citations:
                latest = citations[-1]
                _print(f"  Latest source: {latest.get('title', 'N/A')[:80]}")

        elif node_name == "writer":
            sections = node_output.get("completed_sections", [])
            msgs = node_output.get("progress_messages", [])
            for msg in msgs:
                _print(f"[Writer] {msg}")
            if sections:
                latest = sections[-1]
                content_preview = clean_markdown(latest.get("content", ""))[:300]
                _print(f"  Content preview: {content_preview}...")

        elif node_name == "reviewer":
            history = node_output.get("review_history", [])
            msgs = node_output.get("progress_messages", [])
            for msg in msgs:
                _print(f"[Reviewer] {msg}")
            if history:
                last = history[-1]
                _print(f"  Verdict: {last.get('verdict')}")
                _print(f"  Critique: {last.get('critique', '')[:200]}")

    _print(f"\n{'='*70}")
    _print("FINAL REPORT")
    _print(f"{'='*70}")

    # Read final state
    final_state = graph.get_state(config)
    values = final_state.values if final_state else {}
    sections = values.get("completed_sections", [])
    citations = values.get("citations", [])

    if not sections:
        _print("(No sections produced - the graph may need Streamlit session to inspect state)")
        return

    for i, section in enumerate(sections):
        _print(f"\n--- Section {i+1}: {section.get('title', 'Untitled')} ---")
        clean_text = clean_markdown(section.get("content", ""))
        _print(clean_text[:800])
        if len(clean_text) > 800:
            _print(f"...(truncated, total {len(clean_text)} chars)")

    _print(f"\n{'='*70}")
    _print(f"CITATIONS ({len(citations)} sources)")
    _print(f"{'='*70}")
    for i, c in enumerate(citations, 1):
        _print(f"  [{i}] {c.get('title', 'N/A')}")
        _print(f"      {c.get('url', 'N/A')}")
        _print(f"      {c.get('snippet', 'N/A')[:120]}")

    return sections, citations


if __name__ == "__main__":
    queries = [
        "What are the top 3 electric vehicle battery manufacturers in 2026 and their market shares?",
        "Analyze the current state of AI code assistants market: key players, pricing models, and competitive advantages",
        "How is the global semiconductor supply chain evolving after 2025? Focus on geopolitical risks and new fab investments",
    ]

    if len(sys.argv) > 1:
        queries = [" ".join(sys.argv[1:])]

    for q in queries:
        run_report(q)
        _print(f"\n{'#'*70}")
        _print("#" + " REPORT COMPLETE ".center(68) + "#")
        _print(f"{'#'*70}")
