"""
Task 4 & Task 7: Vector Database, Retrieval Engine & Metadata Filtering
Stores chunk embeddings and rich metadata using FAISS (IndexFlatIP for exact cosine similarity).
Provides Top-K semantic retrieval and post/pre-filtering by metadata fields (department, category, year).
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from src.chunker import DocumentChunk


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    score: float  # Cosine similarity score (between -1.0 and 1.0, typically 0.0 to 1.0)
    metadata: Dict[str, Any]


class VectorStore:
    """
    Vector Database implementation supporting FAISS with full metadata filtering.
    Falls back gracefully to vectorized NumPy inner product if FAISS C-extensions are absent.
    """
    def __init__(self, embedding_dim: int = 384, use_faiss: bool = True):
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None

        if self.use_faiss:
            # IndexFlatIP computes exact inner product (equivalent to cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(embedding_dim)
        else:
            self.index = None

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """
        Ingests document chunks and their corresponding normalized vector embeddings.
        """
        if len(chunks) == 0:
            return

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} embeddings.")

        self.chunks.extend(chunks)

        if self.embeddings is None:
            self.embeddings = embeddings.astype(np.float32)
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings.astype(np.float32)])

        if self.use_faiss:
            self.index.add(embeddings.astype(np.float32))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieves top_k most semantically relevant chunks for the given query vector.
        Applies metadata filtering when specified.
        """
        if not self.chunks or self.embeddings is None:
            return []

        # Ensure 2D query shape
        if query_embedding.ndim == 1:
            q_emb = np.expand_dims(query_embedding, axis=0).astype(np.float32)
        else:
            q_emb = query_embedding.astype(np.float32)

        # If metadata filtering is requested:
        if metadata_filter:
            # Filter candidate chunk indices first
            candidate_indices = []
            for idx, c in enumerate(self.chunks):
                match = True
                for f_key, f_val in metadata_filter.items():
                    if f_val is None or f_val == "" or f_val == "All":
                        continue
                    chunk_val = c.metadata.get(f_key)
                    if str(chunk_val).lower() != str(f_val).lower():
                        match = False
                        break
                if match:
                    candidate_indices.append(idx)

            if not candidate_indices:
                return []

            # Compute similarities on candidate subset
            candidate_embs = self.embeddings[candidate_indices]
            scores = np.dot(candidate_embs, q_emb.T).squeeze(axis=1)
            sorted_order = np.argsort(-scores)

            results: List[RetrievalResult] = []
            for rank_idx in sorted_order[:top_k]:
                orig_idx = candidate_indices[rank_idx]
                chunk = self.chunks[orig_idx]
                score = float(scores[rank_idx])
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    score=round(score, 4),
                    metadata=chunk.metadata
                ))
            return results

        # Without metadata filter:
        if self.use_faiss:
            k = min(top_k, len(self.chunks))
            distances, indices = self.index.search(q_emb, k)
            results = []
            for score, idx in zip(distances[0], indices[0]):
                if idx < 0 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    score=round(float(score), 4),
                    metadata=chunk.metadata
                ))
            return results
        else:
            scores = np.dot(self.embeddings, q_emb.T).squeeze(axis=1)
            sorted_indices = np.argsort(-scores)[:top_k]
            results = []
            for idx in sorted_indices:
                chunk = self.chunks[idx]
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    document_name=chunk.document_name,
                    page_number=chunk.page_number,
                    score=round(float(scores[idx]), 4),
                    metadata=chunk.metadata
                ))
            return results

    @staticmethod
    def explain_metadata_filtering() -> str:
        """
        Conceptual explanation for Task 7 & Section 8.
        """
        return (
            "How Metadata Filtering Improves Retrieval:\n"
            "1. Elimination of Cross-Domain False Positives: Semantic similarity alone can match general terminology across unrelated domains "
            "(e.g., searching 'retention policy' might retrieve both HR employee retention and cloud log retention). Filtering by department "
            "or document category guarantees context purity.\n"
            "2. Temporal Precision: In enterprise documents, older policies or guides (e.g., 2021) may conflict with updated ones (e.g., 2024). "
            "Filtering by Year ensures only active policies are fed to the LLM.\n"
            "3. Search Space Efficiency: Restricting the search space to a specific category reduces the candidate pool, improving signal-to-noise ratio."
        )
