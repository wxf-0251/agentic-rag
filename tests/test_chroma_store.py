from unittest.mock import patch, MagicMock
import pytest
from agentic_rag.tools.chroma_store import (
    ZhipuAIEmbeddingFunction,
    chroma_search,
    index_documents,
    ChromaResult,
)


def test_zhipuai_embedding_function_calls_embedding_endpoint():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4]},
        ]
    }

    with patch("agentic_rag.tools.chroma_store.requests.post", return_value=mock_response) as mock_post:
        embedding_fn = ZhipuAIEmbeddingFunction(
            api_key="test-key",
            model_name="embedding-3",
            api_base="https://open.bigmodel.cn/api/paas/v4",
            dimensions=1536,
        )
        embeddings = embedding_fn(["hello", "world"])

    mock_post.assert_called_once_with(
        "https://open.bigmodel.cn/api/paas/v4/embeddings",
        headers={
            "Authorization": "Bearer test-key",
            "Content-Type": "application/json",
        },
        json={
            "model": "embedding-3",
            "input": ["hello", "world"],
            "dimensions": 1536,
        },
        timeout=60,
    )
    mock_response.raise_for_status.assert_called_once()
    assert len(embeddings) == 2
    assert embeddings[0].tolist() == pytest.approx([0.1, 0.2])
    assert embeddings[1].tolist() == pytest.approx([0.3, 0.4])


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
