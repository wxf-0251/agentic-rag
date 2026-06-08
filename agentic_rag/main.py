import sys
import os
# Ensure the parent directory is on sys.path so 'agentic_rag' is importable
# from any working directory (e.g. when running `cd agentic_rag && streamlit run main.py`)
_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import streamlit as st
from agentic_rag.graph.workflow import build_graph
from agentic_rag.graph.state import AgentState
from agentic_rag.config import MAX_REVIEW_ITERATIONS
from agentic_rag.ui.components import render_sidebar, render_citation_panel, render_research_progress, render_outline_preview
from agentic_rag.ui.renderer import (
    build_report_markdown,
    build_report_pdf,
    clean_markdown,
    dedupe_citations,
    normalize_citation_markers,
    parse_chart_tags,
    render_chart,
    should_render_chart,
    validate_inline_citations,
)
from agentic_rag.ui.report_sections import extract_completed_sections
from agentic_rag.ui.stream_events import extract_node_update
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


def format_runtime_error(exc: Exception) -> str:
    error_name = exc.__class__.__name__
    if error_name in ("APIConnectionError", "APITimeoutError"):
        return (
            "模型 API 连接失败。请检查网络/代理、ANTHROPIC_API_KEY、ANTHROPIC_BASE_URL，"
            "然后重新点击开始研究。已增加超时和重试；如果发生在写作或审核阶段，系统会尽量用已检索资料兜底。"
        )
    return f"研究流程运行失败：{error_name}: {exc}"


def stream_graph_safely(graph, initial_state: AgentState, config: dict):
    try:
        yield from graph.stream(initial_state, config, stream_mode="updates")
    except Exception as exc:
        yield {"__runtime_error__": {"message": format_runtime_error(exc)}}


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
        "max_iterations": MAX_REVIEW_ITERATIONS,
        "progress_messages": [],
        "source_toggles": source_toggles,
    }

    config = {"configurable": {"thread_id": "report-1"}}

    outline_placeholder = col_main.empty()
    progress_placeholder_main = col_main.empty()
    report_container = col_main.container()
    sources_placeholder = col_sources.empty()

    log_messages = []
    research_results_seen = {}
    latest_sections = []
    latest_citations = []
    latest_title = st.session_state.query
    total_sub_questions = 1

    for event in stream_graph_safely(graph, initial_state, config):
        if "__runtime_error__" in event:
            st.session_state.report_started = False
            col_main.error(event["__runtime_error__"]["message"])
            st.stop()

        node_update = extract_node_update(event)
        if node_update is None:
            continue
        node_name, node_output = node_update

        if node_name == "planner":
            if node_output.get("outline"):
                total_sub_questions = len(node_output.get("sub_questions", [])) or 1
                latest_title = node_output["outline"].get("title", latest_title)
                with outline_placeholder.container():
                    render_outline_preview(node_output["outline"])

        elif node_name == "researcher":
            results = node_output.get("research_results", {})
            research_results_seen.update(results)
            latest_citations.extend(node_output.get("citations", []))
            latest_citations = dedupe_citations(latest_citations)
            with progress_placeholder_main.container():
                render_research_progress(len(research_results_seen), total_sub_questions)
            sources_placeholder.info(f"已收集 {len(latest_citations)} 个去重来源，报告完成后展示完整来源列表。")

        elif node_name == "writer":
            sections = extract_completed_sections(node_output)
            if sections:
                latest_sections = sections
                with report_container:
                    section = latest_sections[-1]
                    st.markdown(f"## {section['title']}")
                    clean_text = normalize_citation_markers(clean_markdown(section["content"]), latest_citations)
                    citation_check = validate_inline_citations(clean_text, latest_citations)
                    st.markdown(clean_text, unsafe_allow_html=False)
                    if not citation_check["valid"]:
                        st.warning(f"Missing citation references: {citation_check['missing_numbers']}")

                    charts = parse_chart_tags(section["content"])
                    for chart_data in charts:
                        if not should_render_chart(chart_data):
                            continue
                        fig = render_chart(chart_data)
                        st.pyplot(fig)
                        plt.close(fig)
                    st.divider()

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
    st.session_state.report_started = False
    st.session_state.final_sections = extract_completed_sections({"completed_sections": latest_sections})
    st.session_state.final_citations = dedupe_citations(latest_citations)
    st.session_state.final_title = latest_title
    with sources_placeholder.container():
        render_citation_panel(st.session_state.final_citations)
    st.rerun()

# Restore completed report
if st.session_state.get("report_complete") and not st.session_state.get("report_started"):
    if "final_sections" in st.session_state:
        report_title = st.session_state.get("final_title", st.session_state.get("query", "Research Report"))
        markdown_report = build_report_markdown(
            report_title,
            st.session_state["final_sections"],
            st.session_state.get("final_citations", []),
        )
        for section in st.session_state["final_sections"]:
            col_main.markdown(f"## {section['title']}")
            col_main.markdown(normalize_citation_markers(clean_markdown(section["content"]), st.session_state.get("final_citations", [])))
            for chart_data in parse_chart_tags(section["content"]):
                if not should_render_chart(chart_data):
                    continue
                fig = render_chart(chart_data)
                col_main.pyplot(fig)
                plt.close(fig)
        if st.session_state.get("final_citations"):
            col_main.markdown("## References")
            for citation in dedupe_citations(st.session_state["final_citations"]):
                number = citation.get("number", "")
                title = citation.get("title", "Untitled source")
                url = citation.get("url", "")
                snippet = citation.get("snippet", "")
                if url:
                    col_main.markdown(f"{number}. [{title}]({url}) - {snippet}")
                else:
                    col_main.markdown(f"{number}. {title} - {snippet}")
        col_main.download_button(
            "Download Markdown",
            markdown_report,
            file_name="report.md",
        )
        try:
            pdf_report = build_report_pdf(
                report_title,
                st.session_state["final_sections"],
                st.session_state.get("final_citations", []),
            )
            col_main.download_button(
                "Download PDF",
                pdf_report,
                file_name="report.pdf",
                mime="application/pdf",
            )
        except Exception as exc:
            col_main.caption(f"PDF export unavailable: {exc}")
    if "final_citations" in st.session_state:
        with col_sources:
            render_citation_panel(st.session_state["final_citations"])
