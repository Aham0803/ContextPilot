"""
Task 3: Embedding Generation
Converts text chunks and user queries into dense vector embeddings using SentenceTransformers.
Embeddings capture the contextual and conceptual semantics of text, overcoming the limitations
of exact keyword matching.
"""
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer with normalized vector output
    for efficient cosine similarity calculations.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        # Initializes sentence-transformers model
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

    def generate_embeddings(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Generates dense vector embeddings for a batch of text chunks.
        L2 normalization allows inner product to directly compute cosine similarity.
        """
        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
        return embeddings.astype(np.float32)

    def generate_query_embedding(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Generates a 1D vector embedding for a user query string.
        """
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
        return embedding.astype(np.float32)

    @staticmethod
    def explain_embeddings() -> str:
        """
        Conceptual explanation for Task 3:
        Why embeddings are crucial for semantic search.
        """
        return (
            "Why Embeddings are Essential for Semantic Search:\n"
            "1. Semantic Representation: Traditional lexical search (e.g., BM25, TF-IDF) relies on exact keyword overlap. "
            "If a user searches for 'vacation days' but the policy document uses 'paid time off', keyword search fails completely. "
            "Dense embeddings map text into a continuous high-dimensional vector space where semantically synonymous words and phrases "
            "cluster closely together.\n"
            "2. Geometric Proximity: Vectors that share similar conceptual context have high cosine similarity (close to 1.0), "
            "allowing retrieval algorithms to capture intent, nuances, and domain jargon.\n"
            "3. Cross-Phrasing Robustness: Embeddings abstract beyond morphology, allowing the RAG system to match colloquial or "
            "rephrased queries to formal technical or policy prose."
        )
