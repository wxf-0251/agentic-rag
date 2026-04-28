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
