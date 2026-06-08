from typing import Any, Dict, List, Optional
import requests
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from agentic_rag.config import (
    CHROMA_PERSIST_DIR,
    ZHIPUAI_API_KEY,
    ZHIPUAI_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS,
)

ChromaResult = dict   # {"content": str, "metadata": dict, "distance": float}

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


class ZhipuAIEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
        self,
        api_key: str,
        model_name: str = "embedding-3",
        api_base: str = "https://open.bigmodel.cn/api/paas/v4",
        dimensions: Optional[int] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        payload: Dict[str, Any] = {"model": self.model_name, "input": list(input)}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        response = requests.post(
            f"{self.api_base}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]

    @staticmethod
    def name() -> str:
        return "zhipuai"

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "ZhipuAIEmbeddingFunction":
        return ZhipuAIEmbeddingFunction(**config)

    def get_config(self) -> Dict[str, Any]:
        return {
            "api_key": self.api_key,
            "model_name": self.model_name,
            "api_base": self.api_base,
            "dimensions": self.dimensions,
        }


_embedding_fn = ZhipuAIEmbeddingFunction(
    api_key=ZHIPUAI_API_KEY,
    model_name=EMBEDDING_MODEL,
    api_base=ZHIPUAI_BASE_URL,
    dimensions=EMBEDDING_DIMENSIONS,
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
