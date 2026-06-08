from typing import List
from tavily import TavilyClient
from agentic_rag.agents import normalize_report_text
from agentic_rag.config import TAVILY_API_KEY

_client = TavilyClient(api_key=TAVILY_API_KEY)

SearchResult = dict    # {"title": str, "url": str, "content": str}


def tavily_search(query: str, max_results: int = 5) -> List[SearchResult]:
    try:
        response = _client.search(query=query, max_results=max_results, include_answer=False)
        return [
            {
                "title": normalize_report_text(r.get("title", "")),
                "url": r.get("url", ""),
                "content": normalize_report_text(r.get("content", "")),
                "published_date": r.get("published_date", ""),
            }
            for r in response.get("results", [])
        ]
    except Exception:
        return []
