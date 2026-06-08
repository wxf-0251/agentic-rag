import streamlit as st
from typing import List
from agentic_rag.ui.renderer import dedupe_citations


def render_sidebar() -> tuple:
    with st.sidebar:
        st.title("Agentic RAG")
        st.markdown("Deep Research Report Generator")

        query = st.text_area(
            "Research Query",
            placeholder="E.g.: 分析2026年人形机器人的核心供应链及投资风险",
            height=120,
        )

        st.divider()

        st.caption("Data Sources")
        use_tavily = st.checkbox("Tavily Web Search", value=True)
        use_chroma = st.checkbox("Chroma Local DB", value=True)
        use_yfinance = st.checkbox("Yahoo Finance", value=True)

        st.divider()

        submitted = st.button("Start Research", type="primary", use_container_width=True)

        st.divider()
        st.caption("Progress")
        progress_placeholder = st.empty()
        log_placeholder = st.empty()

    return query, submitted, {"tavily": use_tavily, "chroma": use_chroma, "yfinance": use_yfinance}, progress_placeholder, log_placeholder


def render_citation_panel(citations: List[dict]):
    st.markdown("### Sources")
    citations = dedupe_citations(citations)
    if not citations:
        st.caption("No sources yet...")
        return
    for c in citations:
        i = c.get("number", 0)
        with st.container(border=True):
            st.markdown(f"**[{i}]** [{c['title']}]({c['url']})")
            st.caption(c.get("snippet", "")[:200])
            quality = (
                f"Source: {c.get('source_type', 'unknown')} | "
                f"Freshness: {c.get('freshness_status', 'unknown')} | "
                f"Date: {c.get('published_date', 'unknown')} | "
                f"Authority: {c.get('authority_score', 0)}"
            )
            st.caption(quality)


def render_research_progress(completed: int, total: int):
    st.progress(completed / total if total > 0 else 0, text=f"Research: {completed}/{total} sub-questions")


def render_outline_preview(outline: dict):
    st.markdown("### Report Outline")
    for i, section in enumerate(outline.get("sections", [])):
        st.markdown(f"**{i+1}. {section['title']}**")
