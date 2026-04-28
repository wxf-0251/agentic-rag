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
