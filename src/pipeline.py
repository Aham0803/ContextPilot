"""
End-to-End RAG Pipeline Orchestrator
Coordinates Document Ingestion -> Chunking -> Embeddings -> FAISS Vector Storage ->
Semantic Retrieval -> LLM Grounding -> Citation Generation.
"""
from typing import List, Dict, Any, Optional
from src.ingestion import DocumentIngestionPipeline, DocumentPage, IngestionReport
from src.chunker import TextChunker, DocumentChunk
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore, RetrievalResult
from src.llm import LLMClient, RAGAnswer


class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: Optional[str] = None,
        llm_base_url: Optional[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize pipeline components
        self.ingestion = DocumentIngestionPipeline()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = EmbeddingModel(model_name=embedding_model_name)
        self.vector_store = VectorStore(embedding_dim=self.embedding_model.embedding_dimension)
        self.llm = LLMClient(model_name=llm_model_name, base_url=llm_base_url)

        self.pages: List[DocumentPage] = []
        self.chunks: List[DocumentChunk] = []
        self.ingestion_report: Optional[IngestionReport] = None

    def load_documents(self, directory_path: str) -> IngestionReport:
        """
        Runs ingestion, chunking, embedding generation, and vector indexing.
        """
        self.pages, self.ingestion_report = self.ingestion.ingest_directory(directory_path)
        self.chunks = self.chunker.chunk_pages(self.pages)

        # Generate embeddings for chunks
        chunk_texts = [c.text for c in self.chunks]
        embeddings = self.embedding_model.generate_embeddings(chunk_texts)

        # Add to vector store
        self.vector_store.add_chunks(self.chunks, embeddings)
        return self.ingestion_report

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Generates query embedding and retrieves top-k semantically relevant chunks.
        """
        query_embedding = self.embedding_model.generate_query_embedding(query)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter
        )

    def ask(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> RAGAnswer:
        """
        Executes complete RAG workflow: Retrieval -> Answer Generation -> Citations.
        """
        retrieved_chunks = self.retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        answer = self.llm.generate_answer(question, retrieved_chunks)
        return answer
