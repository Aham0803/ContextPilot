"""
Task 2: Text Chunking
Splits extracted document pages into semantically cohesive chunks using
configurable chunk sizes and overlap, while preserving full lineage and metadata.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion import DocumentPage


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    chunk_index: int
    char_length: int
    document_name: str
    page_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class TextChunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        """
        Initializes the chunker with configurable size and overlap.
        Separators follow recursive character boundaries: paragraphs, sentences, words.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            length_function=len
        )

    def chunk_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """
        Splits list of DocumentPage objects into DocumentChunk objects,
        preserving all document-level and page-level metadata.
        """
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in pages:
            if not page.text.strip():
                continue

            page_splits = self.splitter.split_text(page.text)
            for split_idx, split_text in enumerate(page_splits):
                chunk_id = f"{page.document_name}_p{page.page_number}_c{split_idx}"
                chunk_meta = {
                    **page.metadata,
                    "chunk_id": chunk_id,
                    "chunk_index": split_idx,
                    "global_chunk_index": global_chunk_idx,
                    "chunk_size_config": self.chunk_size,
                    "chunk_overlap_config": self.chunk_overlap
                }

                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    text=split_text.strip(),
                    chunk_index=split_idx,
                    char_length=len(split_text.strip()),
                    document_name=page.document_name,
                    page_number=page.page_number,
                    metadata=chunk_meta
                ))
                global_chunk_idx += 1

        return chunks

    @staticmethod
    def get_sample_chunks(chunks: List[DocumentChunk], num_samples: int = 3) -> List[Dict[str, Any]]:
        """
        Returns a formatted representation of sample chunks for inspection and logging.
        """
        samples = []
        step = max(1, len(chunks) // max(1, num_samples))
        for i in range(0, min(len(chunks), step * num_samples), step):
            c = chunks[i]
            samples.append({
                "chunk_id": c.chunk_id,
                "document": c.document_name,
                "page": c.page_number,
                "length": c.char_length,
                "preview": c.text[:120] + ("..." if len(c.text) > 120 else ""),
                "metadata": c.metadata
            })
        return samples
