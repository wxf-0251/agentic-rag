# Agentic RAG — Autonomous Deep Research Agent

> An Agentic RAG system that produces **20-page cited research reports** through multi-agent reasoning loops, parallel retrieval, and streaming UI.

## Overview

Unlike naive RAG (single retrieval → generation), this system uses a **multi-agent reasoning loop**: decompose, research in parallel, write, critique, re-search if needed.

**Input example:** "Analyze the core supply chain and investment risks of humanoid robots in 2026"

**Output:** A 20-page report with inline citations, auto-generated charts, and exportable PDF/Markdown.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: Planner Agent (Opus 4) │
│  - Decompose query into 5-7     │
│    sub-questions                │
│  - Generate report outline      │
│  - Classify by data source type │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Stage 2: Researcher Agents      │
│          (Haiku, N parallel)    │
│  - Route per sub-question:      │
│    · latest_news → Tavily       │
│    · technical_concept → Chroma │
│    · financial_data → yfinance  │
│    · mixed → Tavily + Chroma    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Stage 3: Writer Agent (Opus 4)  │
│  - Synthesize one section       │
│  - Markdown + <chart> tags      │
│    + inline citations           │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Stage 4: Reviewer Agent (Opus 4)│
│  - Citation coverage check      │
│  - Data consistency check       │
│  - Logic gap detection          │
│  → PASS: next section           │
│  → FAIL: re-search (max 3x)    │
└─────────────────────────────────┘
```

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| LLM (Planner/Writer/Reviewer) | Claude Opus 4 via Anthropic API | Best tool use + 200K context |
| LLM (Researcher) | Claude Haiku 4 | 10x cheaper, same tool capability |
| Agent Orchestration | LangGraph (StateGraph) | Conditional edges for reasoning loops, checkpointing/streaming |
| Web Search | Tavily Search API | AI-optimized structured results |
| Vector DB | ChromaDB (local persist) | Stores domain reports/books |
| Financial Data | yfinance | Free, no API key needed |
| Embeddings | OpenAI text-embedding-3-small | Cost-effective, 1536-dim |
| Frontend | Streamlit | Rapid UI with native streaming |
| Charts | matplotlib | Generated from structured `<chart>` tags |
| PDF Export | weasyprint | Markdown → PDF |

## Project Structure

```
agentic_rag/
├── main.py                    # Streamlit entry point
├── config.py                  # API keys, model config, env vars
├── cli.py                     # CLI entry point
├── requirements.txt
├── .env.example               # Environment template
├── graph/
│   ├── __init__.py
│   ├── state.py               # AgentState TypedDict + data models
│   └── workflow.py            # LangGraph StateGraph definition
├── agents/
│   ├── __init__.py
│   ├── planner.py             # Query decomposition + outline generation
│   ├── researcher.py          # Parallel retrieval with tool routing
│   ├── writer.py              # Section synthesis with citation enforcement
│   └── reviewer.py            # Quality gate: citations, consistency, logic
├── tools/
│   ├── __init__.py
│   ├── search.py              # Tavily API wrapper
│   ├── chroma_store.py        # ChromaDB vector search
│   ├── finance.py             # Yahoo Finance via yfinance
│   ├── source_quality.py      # Source credibility scoring
│   └── fact_verifier.py       # Cross-reference fact checking
├── prompts/
│   ├── planner.md
│   ├── researcher.md
│   ├── writer.md
│   └── reviewer.md
├── ui/
│   ├── __init__.py
│   ├── components.py          # Streamlit reusable widgets
│   ├── renderer.py            # Markdown + <chart> → matplotlib
│   ├── report_sections.py     # Report section rendering
│   └── stream_events.py       # Streaming event types
└── tests/
    ├── test_state.py
    ├── test_search.py
    ├── test_chroma_store.py
    ├── test_finance.py
    ├── test_planner.py
    ├── test_researcher.py
    ├── test_writer.py
    ├── test_reviewer.py
    ├── test_workflow.py
    ├── test_renderer.py
    ├── test_integration.py
    ├── test_config.py
    ├── test_source_quality.py
    ├── test_fact_verifier.py
    ├── test_report_sections.py
    ├── test_stream_events.py
    └── test_text_quality.py
```

## Quick Start

### Prerequisites

- Python 3.11+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
- Tavily API key ([tavily.com](https://tavily.com))
- OpenAI API key (for embeddings, [platform.openai.com](https://platform.openai.com))

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/agentic-rag.git
cd agentic-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Running the App

**Streamlit UI:**
```bash
streamlit run agentic_rag/main.py
```

**CLI:**
```bash
python -m agentic_rag.cli "Analyze the humanoid robot supply chain in 2026"
```

### Running Tests

```bash
pytest tests/ -v
```

## Features

- **Multi-agent reasoning loop** — Planner → Researcher (parallel) → Writer → Reviewer → (re-search if needed)
- **Intelligent tool routing** — Auto-selects Tavily/ChromaDB/YFinance based on query type
- **Real-time streaming** — See research progress, draft sections, and citations live in Streamlit
- **Inline citations** — Every claim is backed by a traceable source
- **Auto-generated charts** — Writer outputs `<chart>` XML tags, renderer converts to matplotlib
- **Quality gate** — Reviewer checks citation coverage, data consistency, and logic gaps
- **Graceful degradation** — Every external dependency has a fallback path
- **Cost control** — Configurable max budget per report, prompt caching on expensive models
- **PDF export** — Full report export via weasyprint

## Configuration

| Env Variable | Default | Purpose |
|-------------|---------|---------|
| `ANTHROPIC_API_KEY` | (required) | Claude API access |
| `TAVILY_API_KEY` | (required) | Web search |
| `OPENAI_API_KEY` | (required) | ChromaDB embeddings |
| `CHROMA_PERSIST_DIR` | `./chroma_data` | Local vector DB path |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MAX_RESEARCH_ITERATIONS` | `3` | Re-search loop cap |
| `MAX_BUDGET_USD` | `2.00` | Hard cost limit per report |
| `PLANNER_MODEL` | `claude-opus-4-20250514` | Model for planning |
| `WRITER_MODEL` | `claude-opus-4-20250514` | Model for writing |
| `REVIEWER_MODEL` | `claude-opus-4-20250514` | Model for reviewing |
| `RESEARCHER_MODEL` | `claude-haiku-4-20250514` | Model for research |

## Design Principles

1. **Not a linear pipeline** — Reviewer can trigger re-search, creating a reasoning loop
2. **Router decides data source per sub-question** — not one-size-fits-all retrieval
3. **Shared typed state via LangGraph StateGraph** — all agents read/write the same TypedDict
4. **Streaming at every stage** — user sees live progress, not a 3-minute spinner
5. **Graceful degradation** — every external dependency has a fallback
6. **Prompt caching** — planner/writer/reviewer system prompts cached, ~90% cost reduction

## Cost Estimate

- **$0.50–1.50** per full report (20 pages, 5-7 sections)
- Max budget hard cap: $2.00/report (configurable)

## License

MIT

## Acknowledgments

Built with:
- [Anthropic Claude API](https://docs.anthropic.com)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Tavily Search API](https://tavily.com)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io)
