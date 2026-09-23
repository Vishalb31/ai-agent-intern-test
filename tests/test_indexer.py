import pytest
from src.rag.indexer import KnowledgeBaseIndexer


@pytest.fixture
def indexer():
    return KnowledgeBaseIndexer("knowledge-base")


def test_filters_superseded_and_internal_notes(indexer):
    chunks = indexer.load_indexable_chunks()
    filenames = {c.filename for c in chunks}

    # 14-internal-content-migration-notes.md must NEVER be indexed
    assert "14-internal-content-migration-notes.md" not in filenames
    
    # 02-returns-policy-legacy.md must NEVER be indexed
    assert "02-returns-policy-legacy.md" not in filenames

    # Current returns policy MUST be indexed
    assert "01-returns-policy-current.md" in filenames


def test_chunk_metadata_retains_headings_and_source(indexer):
    chunks = indexer.load_indexable_chunks()
    for chunk in chunks:
        assert chunk.filename.endswith(".md")
        assert len(chunk.heading) > 0
        assert len(chunk.content) > 0