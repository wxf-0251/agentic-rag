# Agentic RAG — Deep Research Report Generator

## Overview

An Agentic RAG system that produces 20-page cited research reports. Unlike naive RAG (single retrieval → generation), this system uses a multi-agent reasoning loop: decompose, research in parallel, write, critique, re-search if needed.

**User input example:** "分析2026年人形机器人的核心供应链及投资风险"
**Output:** A 20-page report with inline citations, auto-generated charts, and exportable PDF/Markdown.

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| LLM (Planner/Writer/Reviewer) | Claude Opus 4 via Anthropic API | Best tool use + 200K context for long reports |
| LLM (Researcher) | Claude Haiku 4 | 10x cheaper, same tool use capability |
| Agent Orchestration | LangGraph (StateGraph) | Native conditional edges for reasoning loops, built-in checkpointing/streaming |
| Web Search | Tavily Search API | AI-optimized structured results, $0.003/query |
| Vector DB | ChromaDB (local persist) | Stores domain reports/books; fallback for when Tavily is unavailable |
| Financial Data | yfinance (Yahoo Finance) | Free, no API key needed |
| Embeddings | OpenAI text-embedding-3-small | Cost-effective, 1536-dim |
| Frontend | Streamlit | Rapid UI with native streaming support |
| Charts | matplotlib | Generated from structured `<chart>` tags in Writer output |
| PDF Export | weasyprint | Markdown → PDF conversion |

## Architecture

### Agent Pipeline (4 stages)

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│ Stage 1: Planner Agent (Opus 4) │
│  - Decompose query into 5-7     │
│    sub-questions                │
│  - Generate report outline      │
│  - Classify each sub-question   │
│    by data source type          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Stage 2: Researcher Agents      │
│          (Haiku 4, N parallel)  │
│  - Router per sub-question:     │
│    · latest_news → Tavily       │
│    · technical_concept → Chroma │
│    · financial_data → yfinance  │
│    · mixed → Tavily + Chroma    │
│  - Store context + citations    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Stage 3: Writer Agent (Opus 4)  │
│  - Synthesize one section       │
│  - Output markdown + <chart>    │
│    tags + inline citations      │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Stage 4: Reviewer Agent (Opus 4)│
│  - Citation coverage check      │
│  - Data consistency check       │
│  - Logic gap detection          │
│  → PASS: advance to next section│
│  → FAIL: re-search (max 3x)    │
└─────────────────────────────────┘
```

### LangGraph StateGraph

```
[START] → planner → researcher (N parallel via Send API) → writer → reviewer
                        ↖─────────────────────────────────────────────┘
                        FAIL: back to researcher with refined keywords
                        PASS all sections → END
```

### AgentState (TypedDict)

```python
class AgentState(TypedDict):
    query: str
    sub_questions: List[SubQuestion]
    outline: ReportOutline
    research_results: dict       # {question_id: RetrievedContext}
    citations: List[Citation]
    completed_sections: List[ReportSection]
    current_section_idx: int
    review_history: List[ReviewResult]
    search_iteration: int        # re-search loop counter
    max_iterations: int          # safety cap (default 3)
    progress_messages: List[str]
```

## Project Structure

```
agentic_rag/
├── main.py                    # Streamlit entry point
├── config.py                  # API keys, model config, env vars
├── graph/
│   ├── __init__.py
│   ├── state.py               # AgentState TypedDict + SubQuestion/ReportOutline models
│   └── workflow.py            # LangGraph StateGraph definition
├── agents/
│   ├── __init__.py
│   ├── planner.py             # Query decomposition + outline generation
│   ├── researcher.py          # Parallel retrieval with tool routing
│   ├── writer.py              # Section synthesis with citation enforcement
│   └── reviewer.py            # Quality gate: citation coverage, consistency, logic
├── tools/
│   ├── __init__.py
│   ├── search.py              # Tavily API wrapper
│   ├── chroma_store.py        # ChromaDB vector search
│   └── finance.py             # Yahoo Finance via yfinance
├── prompts/
│   ├── planner.md
│   ├── researcher.md
│   ├── writer.md
│   └── reviewer.md
├── ui/
│   ├── __init__.py
│   ├── components.py          # Streamlit reusable widgets
│   └── renderer.py            # Markdown + <chart> → matplotlib + citations
├── requirements.txt
└── .env.example
```

## UI Design

### 3-Panel Layout

- **Left Sidebar (300px):** Query input, model selector, data source toggles, Start button, live progress indicator, collapsible research log
- **Main Panel:** Streaming report — outline preview, research progress cards, draft sections stream token-by-token, inline citation badges, auto-generated charts
- **Right Panel:** Live citation list, source quality scores, clickable links, export buttons (PDF / Markdown)

### Streaming Strategy

4 independent streams via `graph.stream(stream_mode="updates")`:

1. `outline_stream` — Planner completes → render chapter cards
2. `research_stream` — Each Researcher finishes → update progress cards
3. `draft_stream` — Writer yields tokens → stream markdown into main panel
4. `review_stream` — Reviewer completes → status badge + re-search indicator

### Chart Generation

Writer outputs structured `<chart>` tags; renderer.py converts to matplotlib:

```xml
<chart type="bar" title="Top 5 Harmonic Reducer Suppliers">
  x: ["Leaderdrive","Harmonic Drive","Laifu",...]
  y: [35,28,15,...]
</chart>
```

## Error Handling

| Failure | Recovery | UX |
|---------|----------|----|
| Tavily rate limit / timeout | Exponential backoff (1s→2s→4s), max 3 retries. All fail → fallback to Chroma | "Retrying..." → "Using local backup" |
| Yahoo Finance: symbol not found | Return empty result with error metadata; Writer notes "data unavailable" | Inline note in report |
| Chroma DB empty / no results | Auto-escalate to Tavily search | "No local results → searching web..." |
| Reviewer rejects section 3+ times | max_iterations=3 cap; accept with caveat flag after 3rd fail | Yellow banner: "Limited source coverage" |
| LLM API error (5xx, rate limit) | Retry with backoff; if persistent, save partial state for resume | "Service interruption — partial report saved" |
| Vague / unresearchable query | Planner detects vagueness, asks clarifying questions via UI | Chat-style clarification in sidebar |

## Safety Limits

- max_iterations per section: 3
- max_sub_questions: 7
- max_total_time: 300s (5 min)
- max_tokens_per_section: 4000
- max_budget per report: $2.00
- Prompt caching enabled on planner/writer/reviewer system prompts
- Estimated cost per report: $0.50–1.50

## Configuration (config.py / .env)

| Key | Default | Purpose |
|-----|---------|---------|
| ANTHROPIC_API_KEY | env var | Claude Opus + Haiku access |
| TAVILY_API_KEY | env var | Web search |
| CHROMA_PERSIST_DIR | ./chroma_data | Local vector DB path |
| EMBEDDING_MODEL | text-embedding-3-small | OpenAI embeddings for Chroma |
| MAX_RESEARCH_ITERATIONS | 3 | Re-search loop cap |
| MAX_BUDGET_USD | 2.00 | Hard cost limit per report |

## Dependencies (requirements.txt)

```
streamlit
langgraph
langgraph-checkpoint-sqlite
anthropic
tavily-python
chromadb
openai          # embeddings only
yfinance
matplotlib
weasyprint
pandas
python-dotenv
```

## Key Design Principles

1. **Not a linear pipeline** — Reviewer can trigger re-search, creating a reasoning loop
2. **Router decides data source per sub-question** — not one-size-fits-all retrieval
3. **Shared typed state via LangGraph StateGraph** — all agents read/write the same TypedDict
4. **Streaming at every stage** — user sees live progress, not a 3-minute spinner
5. **Graceful degradation** — every external dependency has a fallback path
6. **Prompt caching** — planner/writer/reviewer system prompts are cached to reduce cost by ~90%
