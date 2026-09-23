import os
import pytest
from src.rag.indexer import KnowledgeBaseIndexer
from src.rag.retriever import VectorRetriever


@pytest.fixture
def retriever():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set.")
    indexer = KnowledgeBaseIndexer("knowledge-base")
    return VectorRetriever(indexer=indexer)


def test_retriever_returns_current_policy_and_citations(retriever):
    results = retriever.retrieve("What is the return window for items?", top_k=2)
    assert len(results) > 0

    top_chunk, score = results[0]
    # Verify authoritative source is retrieved
    assert top_chunk.filename == "01-returns-policy-current.md"
    # Ensure source metadata contains citation details
    assert "heading" in top_chunk.model_dump()
    assert len(top_chunk.heading) > 0