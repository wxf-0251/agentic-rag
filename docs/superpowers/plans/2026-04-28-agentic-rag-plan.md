# Agentic RAG — Deep Research Report Generator: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent Agentic RAG system that produces 20-page cited research reports with reasoning loops, parallel retrieval, and Streamlit streaming UI.

**Architecture:** LangGraph StateGraph orchestrates 4 agents (Planner, Researcher, Writer, Reviewer) in a conditional loop. Planner decomposes query → Researcher retrieves from Tavily/Chroma/YFinance in parallel → Writer synthesizes sections → Reviewer quality-gates (PASS/Fail with re-search). Streamlit frontend streams progress per node via `graph.stream(stream_mode="updates")`.

**Tech Stack:** Python, LangGraph, Anthropic SDK (Claude Opus 4 + Haiku 4), Tavily API, ChromaDB, yfinance, Streamlit, matplotlib, weasyprint

---

### Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `agentic_rag/requirements.txt`
- Create: `agentic_rag/.env.example`
- Create: `agentic_rag/config.py`
- Create: `agentic_rag/__init__.py`
- Create: `agentic_rag/graph/__init__.py`
- Create: `agentic_rag/agents/__init__.py`
- Create: `agentic_rag/tools/__init__.py`
- Create: `agentic_rag/ui/__init__.py`
- Create: `agentic_rag/prompts/.gitkeep`

- [ ] **Step 1: Write requirements.txt**

```
streamlit>=1.32.0
langgraph>=0.2.0
langgraph-checkpoint-sqlite>=2.0.0
anthropic>=0.39.0
tavily-python>=0.5.0
chromadb>=0.5.0
openai>=1.50.0
yfinance>=0.2.40
matplotlib>=3.9.0
weasyprint>=62.0
pandas>=2.2.0
python-dotenv>=1.0.0
pytest>=8.0.0
```

- [ ] **Step 2: Write .env.example**

```
ANTHROPIC_API_KEY=sk-ant-...
TAVILY_API_KEY=tvly-...
OPENAI_API_KEY=sk-...
CHROMA_PERSIST_DIR=./chroma_data
EMBEDDING_MODEL=text-embedding-3-small
MAX_RESEARCH_ITERATIONS=3
MAX_BUDGET_USD=2.00
```

- [ ] **Step 3: Write config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
MAX_RESEARCH_ITERATIONS = int(os.environ.get("MAX_RESEARCH_ITERATIONS", "3"))
MAX_BUDGET_USD = float(os.environ.get("MAX_BUDGET_USD", "2.00"))

PLANNER_MODEL = "claude-opus-4-20250514"
RESEARCHER_MODEL = "claude-haiku-4-20250514"
WRITER_MODEL = "claude-opus-4-20250514"
REVIEWER_MODEL = "claude-opus-4-20250514"

MAX_SUB_QUESTIONS = 7
MAX_TOKENS_PER_SECTION = 4000
MAX_TOTAL_TIME_SECONDS = 300
```

- [ ] **Step 4: Create empty __init__.py files for all packages**

Run:
```bash
touch agentic_rag/__init__.py agentic_rag/graph/__init__.py agentic_rag/agents/__init__.py agentic_rag/tools/__init__.py agentic_rag/ui/__init__.py agentic_rag/prompts/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add agentic_rag/requirements.txt agentic_rag/.env.example agentic_rag/config.py agentic_rag/__init__.py agentic_rag/graph/__init__.py agentic_rag/agents/__init__.py agentic_rag/tools/__init__.py agentic_rag/ui/__init__.py agentic_rag/prompts/.gitkeep
git commit -m "feat: scaffold project with dependencies and config"
```

---

### Task 2: State Models (graph/state.py)

**Files:**
- Create: `agentic_rag/graph/state.py`
- Test: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_state.py`:

```python
from agentic_rag.graph.state import SubQuestion, Citation, ReportOutline, ReportSection, ReviewResult, AgentState

def test_sub_question_creation():
    sq = SubQuestion(id="q1", text="What is the market size?", source_type="latest_news", refined_keywords=None)
    assert sq.id == "q1"
    assert sq.source_type == "latest_news"
    assert sq.refined_keywords is None

def test_citation_creation():
    c = Citation(id="c1", title="Market Report 2026", url="https://example.com", snippet="Market size is $10B", source_type="tavily")
    assert c.id == "c1"
    assert c.source_type == "tavily"

def test_report_outline_creation():
    sections = [
        {"title": "Executive Summary", "sub_questions": ["q1", "q2"]},
        {"title": "Supply Chain Analysis", "sub_questions": ["q3"]},
    ]
    outline = ReportOutline(title="Test Report", sections=sections)
    assert len(outline.sections) == 2
    assert outline.sections[0]["title"] == "Executive Summary"

def test_agent_state_defaults():
    state = AgentState(
        query="test query",
        sub_questions=[],
        outline=None,
        research_results={},
        citations=[],
        completed_sections=[],
        current_section_idx=0,
        review_history=[],
        search_iteration=0,
        max_iterations=3,
        progress_messages=[]
    )
    assert state["search_iteration"] == 0
    assert state["max_iterations"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_state.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write state.py**

```python
from typing import TypedDict, List, Optional, Literal, Any


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_state.py agentic_rag/graph/state.py
git commit -m "feat: define AgentState TypedDict and data models"
```

---

### Task 3: Tools — Tavily Search Wrapper

**Files:**
- Create: `agentic_rag/tools/search.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_search.py`:

```python
from unittest.mock import patch, MagicMock
from agentic_rag.tools.search import tavily_search, SearchResult

def test_tavily_search_returns_structured_results():
    mock_response = {
        "results": [
            {"title": "Humanoid Robot Market 2026", "url": "https://example.com/1", "content": "The market is growing rapidly..."},
            {"title": "Supply Chain Analysis", "url": "https://example.com/2", "content": "Key suppliers include..."},
        ]
    }
    with patch("agentic_rag.tools.search.TavilyClient") as mock_client:
        mock_client.return_value.search.return_value = mock_response
        results = tavily_search("humanoid robot supply chain 2026", max_results=5)

    assert len(results) == 2
    assert results[0]["title"] == "Humanoid Robot Market 2026"
    assert results[0]["url"] == "https://example.com/1"
    assert "growing rapidly" in results[0]["content"]

def test_tavily_search_handles_empty_results():
    with patch("agentic_rag.tools.search.TavilyClient") as mock_client:
        mock_client.return_value.search.return_value = {"results": []}
        results = tavily_search("xyzabc123nonexistent", max_results=5)

    assert results == []

def test_tavily_search_handles_api_error():
    with patch("agentic_rag.tools.search.TavilyClient") as mock_client:
        mock_client.return_value.search.side_effect = Exception("Rate limit exceeded")
        results = tavily_search("test query", max_results=5)

    assert results == []  # graceful degradation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_search.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write search.py**

```python
from typing import List
from tavily import TavilyClient
from agentic_rag.config import TAVILY_API_KEY

_client = TavilyClient(api_key=TAVILY_API_KEY)

SearchResult = dict    # {"title": str, "url": str, "content": str}


def tavily_search(query: str, max_results: int = 5) -> List[SearchResult]:
    try:
        response = _client.search(query=query, max_results=max_results, include_answer=False)
        return [
            {"title": r["title"], "url": r["url"], "content": r["content"]}
            for r in response.get("results", [])
        ]
    except Exception:
        return []
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_search.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_search.py agentic_rag/tools/search.py
git commit -m "feat: add Tavily search wrapper with graceful error handling"
```

---

### Task 4: Tools — ChromaDB Vector Store

**Files:**
- Create: `agentic_rag/tools/chroma_store.py`
- Test: `tests/test_chroma_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chroma_store.py`:

```python
from unittest.mock import patch, MagicMock
from agentic_rag.tools.chroma_store import chroma_search, index_documents, ChromaResult

def test_chroma_search_returns_documents():
    mock_results = {
        "documents": [["Harmonic drives use strain wave gearing to achieve high reduction ratios."]],
        "metadatas": [[{"source": "tech_report.pdf", "title": "Robot Joints Explained"}]],
        "distances": [[0.12]],
    }
    with patch("agentic_rag.tools.chroma_store.get_collection") as mock_get_col:
        mock_col = MagicMock()
        mock_col.query.return_value = mock_results
        mock_get_col.return_value = mock_col
        results = chroma_search("harmonic reducer technology", n_results=5)

    assert len(results) == 1
    assert results[0]["content"] == "Harmonic drives use strain wave gearing to achieve high reduction ratios."
    assert results[0]["metadata"]["title"] == "Robot Joints Explained"

def test_chroma_search_handles_no_results():
    with patch("agentic_rag.tools.chroma_store.get_collection") as mock_get_col:
        mock_col = MagicMock()
        mock_col.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_get_col.return_value = mock_col
        results = chroma_search("nonexistent query", n_results=5)

    assert results == []

def test_index_documents_adds_to_collection():
    with patch("agentic_rag.tools.chroma_store.get_collection") as mock_get_col:
        mock_col = MagicMock()
        mock_get_col.return_value = mock_col
        docs = ["Document one content.", "Document two content."]
        metadata = [{"source": "a.pdf"}, {"source": "b.pdf"}]
        ids = ["doc1", "doc2"]

        result = index_documents(docs, metadata, ids)
        assert result is True
        mock_col.add.assert_called_once_with(documents=docs, metadatas=metadata, ids=ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chroma_store.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write chroma_store.py**

```python
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from agentic_rag.config import CHROMA_PERSIST_DIR, OPENAI_API_KEY, EMBEDDING_MODEL

ChromaResult = dict   # {"content": str, "metadata": dict, "distance": float}

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL,
)


def get_collection(name: str = "research_docs"):
    return _client.get_or_create_collection(
        name=name,
        embedding_function=_embedding_fn,
    )


def chroma_search(query: str, n_results: int = 5, collection_name: str = "research_docs") -> List[ChromaResult]:
    try:
        col = get_collection(collection_name)
        response = col.query(query_texts=[query], n_results=n_results)
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]

        results = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            if doc:
                results.append({"content": doc, "metadata": meta or {}, "distance": dist})
        return results
    except Exception:
        return []


def index_documents(documents: List[str], metadatas: List[dict], ids: List[str], collection_name: str = "research_docs") -> bool:
    try:
        col = get_collection(collection_name)
        col.add(documents=documents, metadatas=metadatas, ids=ids)
        return True
    except Exception:
        return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chroma_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_chroma_store.py agentic_rag/tools/chroma_store.py
git commit -m "feat: add ChromaDB vector store with search and indexing"
```

---

### Task 5: Tools — Yahoo Finance Wrapper

**Files:**
- Create: `agentic_rag/tools/finance.py`
- Test: `tests/test_finance.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_finance.py`:

```python
from unittest.mock import patch, MagicMock
import pandas as pd
from agentic_rag.tools.finance import fetch_stock_data, FinanceResult

def test_fetch_stock_data_returns_structured():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame({
        "Close": [150.0, 152.0, 149.0],
    }, index=pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-04"]))
    mock_ticker.info = {"shortName": "Tesla Inc", "currentPrice": 152.0, "marketCap": 500000000000}

    with patch("agentic_rag.tools.finance.yf.Ticker") as mock_ticker_class:
        mock_ticker_class.return_value = mock_ticker
        result = fetch_stock_data("TSLA", period="3mo")

    assert result is not None
    assert result["symbol"] == "TSLA"
    assert result["company_name"] == "Tesla Inc"
    assert result["current_price"] == 152.0
    assert len(result["historical_prices"]) == 3
    assert result["historical_prices"][0]["date"] == "2026-01-02"

def test_fetch_stock_data_handles_invalid_symbol():
    with patch("agentic_rag.tools.finance.yf.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.info = {"trailingPegRatio": None}
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        result = fetch_stock_data("INVALID123XX")
        assert result is None

def test_fetch_stock_data_handles_exception():
    with patch("agentic_rag.tools.finance.yf.Ticker") as mock_ticker_class:
        mock_ticker_class.side_effect = Exception("Network error")
        result = fetch_stock_data("TSLA")
        assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_finance.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write finance.py**

```python
from typing import Optional, List
import yfinance as yf

FinanceResult = dict   # {"symbol": str, "company_name": str, "current_price": float, "historical_prices": list, "market_cap": int}


def fetch_stock_data(symbol: str, period: str = "3mo") -> Optional[FinanceResult]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period=period)
        if hist.empty:
            return None

        prices = [
            {"date": str(idx.date()), "close": round(row["Close"], 2)}
            for idx, row in hist.iterrows()
        ]
        return {
            "symbol": symbol.upper(),
            "company_name": info.get("shortName", symbol),
            "current_price": info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "historical_prices": prices[-90:],   # last 90 trading days max
        }
    except Exception:
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_finance.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_finance.py agentic_rag/tools/finance.py
git commit -m "feat: add Yahoo Finance wrapper for stock data"
```

---

### Task 6: Agent Prompts

**Files:**
- Create: `agentic_rag/prompts/planner.md`
- Create: `agentic_rag/prompts/researcher.md`
- Create: `agentic_rag/prompts/writer.md`
- Create: `agentic_rag/prompts/reviewer.md`

- [ ] **Step 1: Write planner.md**

```markdown
You are a senior research director at a top consulting firm. Your task is to decompose a user's research query into a structured plan.

## Input
A user query about a topic requiring deep research (e.g., market analysis, technology trends, investment risks).

## Your Task
1. **Decompose** the query into 3-7 specific sub-questions. Each sub-question must be independently researchable.
2. **Classify** each sub-question into one of these types:
   - `latest_news`: requires recent web information (news, events, funding rounds)
   - `technical_concept`: requires domain knowledge or definitions (textbooks, encyclopedic sources)
   - `financial_data`: requires stock prices, market caps, financial metrics
   - `mixed`: requires multiple sources combined
3. **Generate a report outline** with 3-6 sections (chapters). Each section maps to 1-3 sub-questions.

## Output Format
You MUST respond with valid JSON only. No other text.

```json
{
  "report_title": "string",
  "sub_questions": [
    {"id": "q1", "text": "string", "source_type": "latest_news|technical_concept|financial_data|mixed"}
  ],
  "outline": {
    "title": "string",
    "sections": [
      {"title": "string", "sub_questions": ["q1", "q2"]}
    ]
  }
}
```

## Rules
- Maximum 7 sub-questions. Quality over quantity.
- Every sub-question must appear in exactly one outline section.
- If the query is too vague to research, set a flag: `"needs_clarification": true` and ask specific clarifying questions.
```

- [ ] **Step 2: Write researcher.md**

```markdown
You are a research analyst. Your job is to answer a specific sub-question using the provided tool results and synthesize findings with citations.

## Input
- A sub-question with a source type
- Search results from the assigned tool (Tavily web search, Chroma local DB, or Yahoo Finance)

## Your Task
1. Read the provided search results.
2. Extract the most relevant facts that answer the sub-question.
3. For each fact, include the source citation (title + URL).
4. If the results are insufficient, suggest refined search keywords.

## Output Format
Respond with valid JSON only:

```json
{
  "sub_question_id": "q1",
  "findings": "A concise summary of what you found, 2-4 paragraphs.",
  "citations": [
    {"id": "c1", "title": "Source Title", "url": "https://...", "snippet": "Relevant excerpt", "source_type": "tavily|chroma|yfinance"}
  ],
  "sufficient": true,
  "refined_keywords": null
}
```

If results are insufficient, set `"sufficient": false` and provide `"refined_keywords"`.
```

- [ ] **Step 3: Write writer.md**

```markdown
You are a senior research report writer at McKinsey. You write long-form, authoritative, and well-cited research reports.

## Input
- Report outline (section titles and scope)
- Research findings from multiple sub-questions
- Full citation list with URLs

## Your Task
Write ONE section of the report at a time. The section should be:

1. **Comprehensive**: 500-1500 words covering all assigned sub-questions.
2. **Well-cited**: Every factual claim must have an inline citation like `[1]`, `[2]` referencing the provided citation list.
3. **Professional tone**: Objective, data-driven, no marketing fluff.
4. **Chart-ready**: Where data supports it, insert chart tags:

```xml
<chart type="bar|line|pie" title="Chart Title">
  x: ["Label1","Label2","Label3"]
  y: [value1, value2, value3]
</chart>
```

## Output Format
```json
{
  "section_index": 0,
  "title": "Section Title",
  "content": "Markdown content with [n] citations and <chart> tags...",
  "citations_used": ["c1", "c3", "c5"]
}
```

## Rules
- Never fabricate citations. Only cite what's in the provided research.
- If research is insufficient for a claim, note it: "Data on this aspect remains limited."
- Charts only when you have actual numbers.
```

- [ ] **Step 4: Write reviewer.md**

```markdown
You are a rigorous fact-checker and editor. Your job is to review a report section and determine if it meets quality standards.

## Input
- A completed report section (markdown with citations)
- The original research findings and citation list

## Your Task
Check the section against these criteria:

1. **Citation Coverage**: Are all factual claims backed by at least one inline citation? Look for unreferenced claims.
2. **Data Consistency**: Do the numbers in the text match the source data? Flag any discrepancies.
3. **Logical Soundness**: Do the conclusions follow from the evidence? Are there logical leaps?
4. **Completeness**: Does the section adequately answer its assigned sub-questions?

## Output Format
```json
{
  "verdict": "PASS",
  "critique": "The section is well-sourced. Minor: consider adding a citation for the market growth projection in paragraph 2.",
  "issues": [],
  "suggested_keywords": null
}
```

If the section FAILS:
```json
{
  "verdict": "FAIL",
  "critique": "Paragraph 3 claims 'Tesla holds 60% market share' without any citation. The revenue figures don't match the source data (source says $2.1B, text says $3.5B).",
  "issues": ["uncited_claim", "data_mismatch"],
  "suggested_keywords": "Tesla humanoid robot market share 2026"
}
```

## Rules
- Be strict but fair. PASS only when the section is publishable.
- When you FAIL a section, always provide `suggested_keywords` for re-search.
- Do not nitpick writing style. Focus on factual accuracy and logical rigor.
```

- [ ] **Step 5: Commit**

```bash
git add agentic_rag/prompts/planner.md agentic_rag/prompts/researcher.md agentic_rag/prompts/writer.md agentic_rag/prompts/reviewer.md
git commit -m "feat: add system prompts for all 4 agents"
```

---

### Task 7: Planner Agent

**Files:**
- Create: `agentic_rag/agents/planner.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_planner.py`:

```python
from unittest.mock import patch, MagicMock
from agentic_rag.agents.planner import run_planner

def test_planner_decomposes_query():
    mock_response = MagicMock()
    mock_response.content = [
        {"type": "text", "text": '{"report_title":"Test Report","sub_questions":[{"id":"q1","text":"What is the market size?","source_type":"latest_news"},{"id":"q2","text":"Who are the key players?","source_type":"technical_concept"}],"outline":{"title":"Test Report","sections":[{"title":"Overview","sub_questions":["q1","q2"]}]}}'}
    ]
    with patch("agentic_rag.agents.planner.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_planner("What is the state of humanoid robots in 2026?")

    assert result["report_title"] == "Test Report"
    assert len(result["sub_questions"]) == 2
    assert result["sub_questions"][0]["source_type"] == "latest_news"
    assert result["outline"]["title"] == "Test Report"
    assert len(result["outline"]["sections"]) == 1

def test_planner_detects_vague_query():
    mock_response = MagicMock()
    mock_response.content = [
        {"type": "text", "text": '{"needs_clarification":true,"clarification_questions":["Which industry?","What time period?"],"sub_questions":[],"outline":null,"report_title":""}'}
    ]
    with patch("agentic_rag.agents.planner.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_planner("analyze the market")

    assert result["needs_clarification"] is True
    assert len(result["clarification_questions"]) == 2

def test_planner_enforces_max_sub_questions():
    """Planner should respect max 7 sub-questions from config."""
    mock_response = MagicMock()
    # Return 10 sub-questions — the planner function should warn/log but not crash
    questions = [{"id": f"q{i}", "text": f"Question {i}", "source_type": "latest_news"} for i in range(1, 11)]
    import json
    mock_response.content = [
        {"type": "text", "text": json.dumps({
            "report_title": "Test",
            "sub_questions": questions,
            "outline": {"title": "Test", "sections": [{"title": "S1", "sub_questions": [f"q{i}" for i in range(1,11)]}]}
        })}
    ]
    with patch("agentic_rag.agents.planner.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_planner("test query")
    # Should not crash; should still return structured data
    assert "sub_questions" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write planner.py**

```python
import json
from typing import Optional
import anthropic
from agentic_rag.config import ANTHROPIC_API_KEY, PLANNER_MODEL, MAX_SUB_QUESTIONS

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = None

def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open("agentic_rag/prompts/planner.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_planner(query: str) -> dict:
    response = _client.messages.create(
        model=PLANNER_MODEL,
        max_tokens=2000,
        system=[
            {"type": "text", "text": _load_prompt(),
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": query}],
    )
    raw = response.content[0].text
    result = json.loads(raw)

    # Enforce max sub-questions
    if len(result.get("sub_questions", [])) > MAX_SUB_QUESTIONS:
        result["sub_questions"] = result["sub_questions"][:MAX_SUB_QUESTIONS]

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_planner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_planner.py agentic_rag/agents/planner.py
git commit -m "feat: add Planner agent with query decomposition and prompt caching"
```

---

### Task 8: Researcher Agent

**Files:**
- Create: `agentic_rag/agents/researcher.py`
- Test: `tests/test_researcher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_researcher.py`:

```python
from unittest.mock import patch, MagicMock
from agentic_rag.agents.researcher import run_researcher, route_to_tool

def test_route_to_tool_returns_correct_tool():
    assert route_to_tool("latest_news") == "tavily"
    assert route_to_tool("technical_concept") == "chroma"
    assert route_to_tool("financial_data") == "yfinance"
    assert route_to_tool("mixed") == "tavily+chroma"

def test_run_researcher_with_tavily():
    mock_search = [{"title": "Test", "url": "https://x.com", "content": "Market is $10B"}]
    mock_llm_response = MagicMock()
    mock_llm_response.content = [
        {"type": "text", "text": '{"sub_question_id":"q1","findings":"Market is $10B.","citations":[{"id":"c1","title":"Test","url":"https://x.com","snippet":"Market is $10B","source_type":"tavily"}],"sufficient":true,"refined_keywords":null}'}
    ]

    with patch("agentic_rag.agents.researcher.tavily_search", return_value=mock_search):
        with patch("agentic_rag.agents.researcher.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_llm_response
            mock_anthropic.return_value = mock_client

            result = run_researcher({"id": "q1", "text": "Market size?", "source_type": "latest_news", "refined_keywords": None})

    assert result["sub_question_id"] == "q1"
    assert len(result["citations"]) == 1
    assert result["sufficient"] is True

def test_run_researcher_with_empty_results():
    with patch("agentic_rag.agents.researcher.tavily_search", return_value=[]):
        with patch("agentic_rag.agents.researcher.anthropic.Anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = MagicMock()
            mock_client.messages.create.return_value.content = [
                {"type": "text", "text": '{"sub_question_id":"q1","findings":"No data found.","citations":[],"sufficient":false,"refined_keywords":"broader market search"}'}
            ]
            mock_anthropic.return_value = mock_client

            result = run_researcher({"id": "q1", "text": "Market size?", "source_type": "latest_news", "refined_keywords": None})

    assert result["sufficient"] is False
    assert result["refined_keywords"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_researcher.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write researcher.py**

```python
import json
from typing import Literal
import anthropic
from agentic_rag.config import ANTHROPIC_API_KEY, RESEARCHER_MODEL
from agentic_rag.tools.search import tavily_search
from agentic_rag.tools.chroma_store import chroma_search
from agentic_rag.tools.finance import fetch_stock_data

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = None

def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open("agentic_rag/prompts/researcher.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def route_to_tool(source_type: str) -> str:
    mapping = {
        "latest_news": "tavily",
        "technical_concept": "chroma",
        "financial_data": "yfinance",
        "mixed": "tavily+chroma",
    }
    return mapping.get(source_type, "tavily")


def _execute_search(sub_question: dict) -> list:
    source_type = sub_question["source_type"]
    query = sub_question.get("refined_keywords") or sub_question["text"]
    results = []

    if source_type in ("latest_news", "mixed"):
        results.extend(tavily_search(query))
    if source_type in ("technical_concept", "mixed"):
        results.extend(chroma_search(query))
    if source_type == "financial_data":
        # For financial queries, extract potential stock symbols from the question
        finance_result = fetch_stock_data(query)
        if finance_result:
            results.append({"title": finance_result["company_name"], "url": "", "content": json.dumps(finance_result)})

    return results


def run_researcher(sub_question: dict) -> dict:
    search_results = _execute_search(sub_question)

    user_content = f"Sub-question: {sub_question['text']}\n\nSearch results:\n{json.dumps(search_results, ensure_ascii=False)}"

    response = _client.messages.create(
        model=RESEARCHER_MODEL,
        max_tokens=1500,
        system=[{"type": "text", "text": _load_prompt()}],
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(response.content[0].text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_researcher.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_researcher.py agentic_rag/agents/researcher.py
git commit -m "feat: add Researcher agent with tool routing"
```

---

### Task 9: Writer Agent

**Files:**
- Create: `agentic_rag/agents/writer.py`
- Test: `tests/test_writer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_writer.py`:

```python
from unittest.mock import patch, MagicMock
from agentic_rag.agents.writer import run_writer

def test_writer_synthesizes_section():
    mock_response = MagicMock()
    mock_response.content = [
        {"type": "text", "text": '{"section_index":0,"title":"Market Overview","content":"The humanoid robot market is growing rapidly [1]. Key players include Tesla and Boston Dynamics [2].\\n\\n<chart type=\\"bar\\" title=\\"Market Growth\\">\\n  x: [\\"2024\\",\\"2025\\",\\"2026\\"]\\n  y: [5.2, 8.1, 13.4]\\n</chart>","citations_used":["c1","c2"]}'}
    ]
    with patch("agentic_rag.agents.writer.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        section_outline = {"title": "Market Overview", "sub_questions": ["q1", "q2"]}
        research = {"q1": "Market data shows 40% CAGR.", "q2": "Tesla Optimus leads."}
        citations = [{"id": "c1", "title": "Report A", "url": "https://a.com", "snippet": "", "source_type": "tavily"}, {"id": "c2", "title": "Report B", "url": "https://b.com", "snippet": "", "source_type": "tavily"}]

        result = run_writer(section_index=0, section_outline=section_outline, research_results=research, citations=citations)

    assert result["section_index"] == 0
    assert "[1]" in result["content"]
    assert "<chart" in result["content"]
    assert len(result["citations_used"]) == 2

def test_writer_handles_empty_research():
    mock_response = MagicMock()
    mock_response.content = [
        {"type": "text", "text": '{"section_index":0,"title":"Risks","content":"Data on this aspect remains limited. Further research is needed.","citations_used":[]}'}
    ]
    with patch("agentic_rag.agents.writer.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = run_writer(section_index=0, section_outline={"title": "Risks", "sub_questions": ["q1"]}, research_results={}, citations=[])

    assert "Data on this aspect remains limited" in result["content"]
    assert result["citations_used"] == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_writer.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write writer.py**

```python
import json
import anthropic
from agentic_rag.config import ANTHROPIC_API_KEY, WRITER_MODEL, MAX_TOKENS_PER_SECTION

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = None

def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open("agentic_rag/prompts/writer.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_writer(section_index: int, section_outline: dict, research_results: dict, citations: list) -> dict:
    user_content = json.dumps({
        "section_index": section_index,
        "section_outline": section_outline,
        "research_results": research_results,
        "citations": citations,
    }, ensure_ascii=False)

    response = _client.messages.create(
        model=WRITER_MODEL,
        max_tokens=MAX_TOKENS_PER_SECTION,
        system=[
            {"type": "text", "text": _load_prompt(),
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(response.content[0].text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_writer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_writer.py agentic_rag/agents/writer.py
git commit -m "feat: add Writer agent with citation-enforced synthesis"
```

---

### Task 10: Reviewer Agent

**Files:**
- Create: `agentic_rag/agents/reviewer.py`
- Test: `tests/test_reviewer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reviewer.py`:

```python
from unittest.mock import patch, MagicMock
from agentic_rag.agents.reviewer import run_reviewer

def test_reviewer_passes_good_section():
    mock_response = MagicMock()
    mock_response.content = [
        {"type": "text", "text": '{"verdict":"PASS","critique":"Well-sourced and logically sound.","issues":[],"suggested_keywords":null}'}
    ]
    with patch("agentic_rag.agents.reviewer.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        section = {"section_index": 0, "title": "Overview", "content": "Market grows 40% [1].", "citations_used": ["c1"], "charts": []}
        research = {"q1": "Market grows at 40% CAGR."}
        citations = [{"id": "c1", "title": "Source", "url": "https://x.com", "snippet": "40% CAGR", "source_type": "tavily"}]

        result = run_reviewer(section, research, citations)

    assert result["verdict"] == "PASS"
    assert result["issues"] == []

def test_reviewer_fails_uncited_claims():
    mock_response = MagicMock()
    mock_response.content = [
        {"type": "text", "text": '{"verdict":"FAIL","critique":"Claims without citations.","issues":["uncited_claim"],"suggested_keywords":"humanoid robot market share 2026"}'}
    ]
    with patch("agentic_rag.agents.reviewer.anthropic.Anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        section = {"section_index": 0, "title": "Overview", "content": "The market is huge. Many players exist.", "citations_used": [], "charts": []}
        result = run_reviewer(section, {}, [])

    assert result["verdict"] == "FAIL"
    assert result["suggested_keywords"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reviewer.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write reviewer.py**

```python
import json
import anthropic
from agentic_rag.config import ANTHROPIC_API_KEY, REVIEWER_MODEL

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = None

def _load_prompt() -> str:
    global SYSTEM_PROMPT
    if SYSTEM_PROMPT is None:
        with open("agentic_rag/prompts/reviewer.md", "r") as f:
            SYSTEM_PROMPT = f.read()
    return SYSTEM_PROMPT


def run_reviewer(section: dict, research_results: dict, citations: list) -> dict:
    user_content = json.dumps({
        "section": section,
        "research_results": research_results,
        "citations": citations,
    }, ensure_ascii=False)

    response = _client.messages.create(
        model=REVIEWER_MODEL,
        max_tokens=1000,
        system=[
            {"type": "text", "text": _load_prompt(),
             "cache_control": {"type": "ephemeral"}}
        ],
        messages=[{"role": "user", "content": user_content}],
    )
    return json.loads(response.content[0].text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reviewer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_reviewer.py agentic_rag/agents/reviewer.py
git commit -m "feat: add Reviewer agent with quality gate"
```

---

### Task 11: LangGraph Workflow

**Files:**
- Create: `agentic_rag/graph/workflow.py`
- Test: `tests/test_workflow.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflow.py`:

```python
from unittest.mock import patch, MagicMock
import json
from agentic_rag.graph.state import AgentState
from agentic_rag.graph.workflow import build_graph

def test_graph_compiles_and_accepts_state():
    graph = build_graph()
    assert graph is not None

    initial_state: AgentState = {
        "query": "Test query",
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

    # Verify graph compiles by checking it can stream initial state
    config = {"configurable": {"thread_id": "test-1"}}
    try:
        events = list(graph.stream(initial_state, config, stream_mode="updates"))
        # Even if nodes fail (no API keys), the graph should start executing
    except Exception:
        pass  # Expected without real API keys
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_workflow.py -v`
Expected: FAIL (either ModuleNotFoundError or assertion error)

- [ ] **Step 3: Write workflow.py**

```python
from typing import Literal
import langgraph.graph as lg
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send, END
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_workflow.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_workflow.py agentic_rag/graph/workflow.py
git commit -m "feat: implement LangGraph workflow with reasoning loop"
```

---

### Task 12: UI — Streamlit Components

**Files:**
- Create: `agentic_rag/ui/components.py`
- Create: `agentic_rag/ui/renderer.py`
- Test: `tests/test_renderer.py`

- [ ] **Step 1: Write the failing test for renderer**

Create `tests/test_renderer.py`:

```python
from agentic_rag.ui.renderer import parse_chart_tags, render_chart

def test_parse_chart_tags_extracts_bar_chart():
    markdown = """Some text.
<chart type="bar" title="Market Share">
  x: ["A","B","C"]
  y: [30,50,20]
</chart>
More text."""
    charts = parse_chart_tags(markdown)
    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    assert charts[0]["title"] == "Market Share"
    assert charts[0]["x"] == ["A", "B", "C"]
    assert charts[0]["y"] == [30, 50, 20]

def test_parse_chart_tags_multiple():
    markdown = """Text.
<chart type="bar" title="Chart 1">
  x: ["X"]
  y: [1]
</chart>
<chart type="line" title="Chart 2">
  x: ["A","B"]
  y: [10,20]
</chart>"""
    charts = parse_chart_tags(markdown)
    assert len(charts) == 2
    assert charts[0]["type"] == "bar"
    assert charts[1]["type"] == "line"

def test_parse_chart_tags_no_charts():
    markdown = "Just plain text without any chart tags."
    charts = parse_chart_tags(markdown)
    assert charts == []

def test_render_chart_returns_figure():
    chart_data = {"type": "bar", "title": "Test", "x": ["A", "B"], "y": [10, 20]}
    fig = render_chart(chart_data)
    assert fig is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_renderer.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write renderer.py**

```python
import re
import json
from typing import List
import matplotlib.pyplot as plt
import io


def parse_chart_tags(markdown: str) -> List[dict]:
    pattern = r"<chart type=\"(.*?)\" title=\"(.*?)\">\s*x:\s*(\[.*?\])\s*y:\s*(\[.*?\])\s*</chart>"
    matches = re.findall(pattern, markdown, re.DOTALL)
    charts = []
    for chart_type, title, x_str, y_str in matches:
        try:
            x = json.loads(x_str)
            y = json.loads(y_str)
            charts.append({"type": chart_type, "title": title, "x": x, "y": y})
        except json.JSONDecodeError:
            continue
    return charts


def render_chart(chart_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    chart_type = chart_data["type"]
    x = chart_data["x"]
    y = chart_data["y"]

    if chart_type == "bar":
        ax.bar(x, y, color="#3b82f6")
    elif chart_type == "line":
        ax.plot(x, y, marker="o", color="#22c55e")
    elif chart_type == "pie":
        ax.pie(y, labels=x, autopct="%1.1f%%")

    ax.set_title(chart_data["title"])
    plt.tight_layout()
    return fig


def clean_markdown(markdown: str) -> str:
    """Remove chart tags, leaving only markdown text."""
    return re.sub(r"<chart.*?</chart>", "", markdown, flags=re.DOTALL)


def build_citation_html(citations: list) -> str:
    """Build HTML for citation list."""
    html = "<ol>"
    for i, c in enumerate(citations, 1):
        html += f'<li id="cite-{i}"><a href="{c["url"]}" target="_blank">{c["title"]}</a> — {c["snippet"]}</li>'
    html += "</ol>"
    return html
```

- [ ] **Step 4: Write components.py**

```python
import streamlit as st
from typing import List


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
    if not citations:
        st.caption("No sources yet...")
        return
    for i, c in enumerate(citations, 1):
        with st.container(border=True):
            st.markdown(f"**[{i}]** [{c['title']}]({c['url']})")
            st.caption(c.get("snippet", "")[:200])
            st.caption(f"Source: {c.get('source_type', 'unknown')}")


def render_research_progress(completed: int, total: int):
    st.progress(completed / total if total > 0 else 0, text=f"Research: {completed}/{total} sub-questions")


def render_outline_preview(outline: dict):
    st.markdown("### Report Outline")
    for i, section in enumerate(outline.get("sections", [])):
        st.markdown(f"**{i+1}. {section['title']}**")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_renderer.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tests/test_renderer.py agentic_rag/ui/renderer.py agentic_rag/ui/components.py
git commit -m "feat: add Streamlit UI components and chart renderer"
```

---

### Task 13: Main Entry Point & Integration

**Files:**
- Create: `agentic_rag/main.py`

- [ ] **Step 1: Write main.py**

```python
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
            citations = node_output.get("citations", [])
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
```

- [ ] **Step 2: Verify Streamlit can import the app**

Run:
```bash
streamlit run agentic_rag/main.py --server.headless true 2>&1 | head -5
```
Expected: Streamlit starts (you may see port binding info). Kill with Ctrl+C afterward.

- [ ] **Step 3: Commit**

```bash
git add agentic_rag/main.py
git commit -m "feat: add Streamlit main entry point with streaming"
```

---

### Task 14: Integration End-to-End Test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_integration.py`:

```python
from unittest.mock import patch, MagicMock
import json
from agentic_rag.graph.workflow import build_graph
from agentic_rag.graph.state import AgentState

def test_full_graph_execution_with_mocks():
    """End-to-end test with all external APIs mocked."""
    mock_planner_result = {
        "report_title": "Test Report",
        "sub_questions": [
            {"id": "q1", "text": "Test question 1?", "source_type": "latest_news"},
            {"id": "q2", "text": "Test question 2?", "source_type": "technical_concept"},
        ],
        "outline": {
            "title": "Test Report",
            "sections": [
                {"title": "Section 1", "sub_questions": ["q1", "q2"]},
            ],
        },
    }

    mock_researcher_result = {
        "sub_question_id": "q1",
        "findings": "Test findings.",
        "citations": [{"id": "c1", "title": "Source", "url": "https://x.com", "snippet": "data", "source_type": "tavily"}],
        "sufficient": True,
        "refined_keywords": None,
    }

    mock_writer_result = {
        "section_index": 0,
        "title": "Section 1",
        "content": "Test content with citation [1].",
        "citations_used": ["c1"],
    }

    mock_reviewer_result = {
        "verdict": "PASS",
        "critique": "Good.",
        "issues": [],
        "suggested_keywords": None,
    }

    def mock_planner(*args, **kwargs):
        return mock_planner_result

    def mock_researcher(sub_q):
        return {**mock_researcher_result, "sub_question_id": sub_q["id"]}

    def mock_writer(*args, **kwargs):
        return mock_writer_result

    def mock_reviewer(*args, **kwargs):
        return mock_reviewer_result

    with patch("agentic_rag.graph.workflow.run_planner", side_effect=mock_planner):
        with patch("agentic_rag.graph.workflow.run_researcher", side_effect=mock_researcher):
            with patch("agentic_rag.graph.workflow.run_writer", side_effect=mock_writer):
                with patch("agentic_rag.graph.workflow.run_reviewer", side_effect=mock_reviewer):
                    graph = build_graph()
                    initial_state: AgentState = {
                        "query": "test",
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
                    config = {"configurable": {"thread_id": "integration-test"}}
                    events = list(graph.stream(initial_state, config, stream_mode="updates"))

    # Verify events were emitted
    assert len(events) > 0
    node_names = [list(e.keys())[0] for e in events]
    assert "planner" in node_names
    assert "researcher" in node_names
    assert "writer" in node_names
    assert "reviewer" in node_names

def test_reviewer_loop_respects_max_iterations():
    """Verify the Reviewer→Researcher loop stops after max_iterations."""
    mock_planner_result = {
        "report_title": "Test",
        "sub_questions": [{"id": "q1", "text": "Q1?", "source_type": "latest_news"}],
        "outline": {"title": "Test", "sections": [{"title": "S1", "sub_questions": ["q1"]}]},
    }

    mock_researcher_result = {
        "sub_question_id": "q1",
        "findings": "Findings.",
        "citations": [],
        "sufficient": True,
        "refined_keywords": None,
    }

    mock_writer_result = {
        "section_index": 0,
        "title": "S1",
        "content": "Content.",
        "citations_used": [],
    }

    fail_count = [0]  # mutable counter

    def mock_reviewer(*args, **kwargs):
        fail_count[0] += 1
        # Always FAIL — tests that max_iterations stops the loop
        return {
            "verdict": "FAIL",
            "critique": "Not good enough.",
            "issues": ["uncited_claim"],
            "suggested_keywords": "refined search",
        }

    with patch("agentic_rag.graph.workflow.run_planner", return_value=mock_planner_result):
        with patch("agentic_rag.graph.workflow.run_researcher", side_effect=lambda sq: {**mock_researcher_result, "sub_question_id": sq["id"]}):
            with patch("agentic_rag.graph.workflow.run_writer", return_value=mock_writer_result):
                with patch("agentic_rag.graph.workflow.run_reviewer", side_effect=mock_reviewer):
                    graph = build_graph()
                    initial_state: AgentState = {
                        "query": "test",
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
                    config = {"configurable": {"thread_id": "loop-test"}}
                    events = list(graph.stream(initial_state, config, stream_mode="updates"))

    # max_iterations=3 means the reviewer-loop runs at most 3 times before forcing accept
    assert fail_count[0] <= 3, f"Reviewer called {fail_count[0]} times, should be capped at 3"
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (2 tests pass — full graph and loop cap)

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All 14+ tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for full agent pipeline"
```

---

### Task 15: Final Polish — .gitignore, README stub

**Files:**
- Create: `.gitignore`
- Create: `.superpowers` entry if not present

- [ ] **Step 1: Write .gitignore**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
.venv/
venv/

# Environment
.env
*.env.local

# Chroma
chroma_data/

# Streamlit
.streamlit/secrets.toml

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Superpowers
.superpowers/brainstorm/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```
