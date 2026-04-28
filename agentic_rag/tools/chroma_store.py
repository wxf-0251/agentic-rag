from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from agentic_rag.config import CHROMA_PERSIST_DIR, OPENAI_API_KEY, OPENAI_BASE_URL, EMBEDDING_MODEL

ChromaResult = dict   # {"content": str, "metadata": dict, "distance": float}

_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
_embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL,
    api_base=OPENAI_BASE_URL,
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
