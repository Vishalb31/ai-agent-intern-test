import os
from typing import List, Tuple
import numpy as np
from openai import OpenAI
from src.rag.indexer import DocumentChunk, KnowledgeBaseIndexer


class VectorRetriever:
    def __init__(
        self,
        indexer: KnowledgeBaseIndexer,
        model: str = "text-embedding-3-small",
        api_key: str | None = None
    ):
        self.indexer = indexer
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.array([])
        self._build_index()

    def _build_index(self) -> None:
        """Loads eligible chunks and precomputes embeddings."""
        self.chunks = self.indexer.load_indexable_chunks()
        if not self.chunks:
            return

        texts_to_embed = [
            f"Title: {c.title}\nHeading: {c.heading}\n\n{c.content}"
            for c in self.chunks
        ]

        response = self.client.embeddings.create(
            input=texts_to_embed,
            model=self.model
        )

        vectors = [item.embedding for item in response.data]
        self.embeddings = np.array(vectors, dtype=np.float32)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

    def retrieve(self, query: str, top_k: int = 3, min_similarity: float = 0.25) -> List[Tuple[DocumentChunk, float]]:
        """
        Embeds the incoming query and performs cosine similarity search.
        Returns a list of (DocumentChunk, score) tuples.
        """
        if not query or len(self.chunks) == 0:
            return []

        response = self.client.embeddings.create(
            input=[query],
            model=self.model
        )
        query_vec = np.array(response.data[0].embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm

        # Cosine similarity via dot product
        similarities = np.dot(self.embeddings, query_vec)

        # Rank and filter
        ranked_indices = np.argsort(similarities)[::-1]
        results: List[Tuple[DocumentChunk, float]] = []

        for idx in ranked_indices[:top_k]:
            score = float(similarities[idx])
            if score >= min_similarity:
                results.append((self.chunks[idx], score))

        return results