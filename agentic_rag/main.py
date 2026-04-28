import streamlit as st
from agentic_rag.graph.workflow import build_graph
from agentic_rag.graph.state import AgentState
from agentic_rag.ui.components import render_sidebar, render_citation_panel, render_research_progress, render_outline_preview
from agentic_rag.ui.renderer import parse_chart_tags, render_chart, clean_markdown
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Agentic RAG — Deep Research")

# Initialize session state
if "report_started" not in st.session_state:
    st.session_state.report_started = False
if "report_complete" not in st.session_state:
    st.session_state.report_complete = False

# Sidebar
query, submitted, source_toggles, progress_placeholder, log_placeholder = render_sidebar()

# Main area
col_main, col_sources = st.columns([3, 1])


def reset_report():
    st.session_state.report_started = False
    st.session_state.report_complete = False


if submitted and query.strip():
    reset_report()
    st.session_state.report_started = True
    st.session_state.query = query

if st.session_state.report_started:
    graph = build_graph()

    initial_state: AgentState = {
        "query": st.session_state.query,
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

    config = {"configurable": {"thread_id": "report-1"}}

    outline_placeholder = col_main.empty()
    progress_placeholder_main = col_main.empty()
    draft_placeholder = col_main.empty()
    sources_container = col_sources.container()

    log_messages = []

    for event in graph.stream(initial_state, config, stream_mode="updates"):
        node_name = list(event.keys())[0]
        node_output = event[node_name]

        if node_name == "planner":
            if node_output.get("outline"):
                with outline_placeholder.container():
                    render_outline_preview(node_output["outline"])

        elif node_name == "researcher":
            results = node_output.get("research_results", {})
            total_qs = len(st.session_state.get("sub_questions", [])) or 1
            with progress_placeholder_main.container():
                render_research_progress(len(results), total_qs)

        elif node_name == "writer":
            sections = node_output.get("completed_sections", [])
            if sections:
                latest = sections[-1]
                with draft_placeholder.container():
                    st.markdown(f"## {latest['title']}")
                    clean_text = clean_markdown(latest["content"])
                    st.markdown(clean_text, unsafe_allow_html=False)

                    charts = parse_chart_tags(latest["content"])
                    for chart_data in charts:
                        fig = render_chart(chart_data)
                        st.pyplot(fig)
                        plt.close(fig)

        elif node_name == "reviewer":
            review_history = node_output.get("review_history", [])
            if review_history:
                last = review_history[-1]
                status = "✅" if last["verdict"] == "PASS" else "🔄 Re-searching..."
                log_messages.append(f"{status} {last.get('critique', '')[:100]}")

        msgs = node_output.get("progress_messages", [])
        log_messages.extend(msgs)
        with log_placeholder.container():
            for msg in log_messages[-5:]:  # show last 5
                st.caption(msg)

    st.session_state.report_complete = True
    st.session_state.final_sections = st.session_state.get("completed_sections", [])
    st.session_state.final_citations = st.session_state.get("citations", [])

# Restore completed report
if st.session_state.get("report_complete") and not st.session_state.get("report_started"):
    if "final_sections" in st.session_state:
        for section in st.session_state["final_sections"]:
            col_main.markdown(f"## {section['title']}")
            col_main.markdown(clean_markdown(section["content"]))
        col_main.download_button(
            "Download Markdown",
            "\n\n".join([s["content"] for s in st.session_state["final_sections"]]),
            file_name="report.md",
        )
    if "final_citations" in st.session_state:
        with col_sources:
            render_citation_panel(st.session_state["final_citations"])
