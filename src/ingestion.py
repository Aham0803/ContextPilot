"""
Task 1: Document Ingestion Pipeline
Loads multiple PDF files, extracts text page-by-page, handles empty/corrupt pages,
and preserves comprehensive metadata for downstream retrieval and citation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import pypdf


@dataclass
class DocumentPage:
    text: str
    page_number: int  # 1-indexed
    total_pages: int
    document_name: str
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionReport:
    total_documents: int = 0
    total_pages_scanned: int = 0
    total_valid_pages: int = 0
    empty_pages_encountered: int = 0
    document_summaries: List[Dict[str, Any]] = field(default_factory=list)


# Known default metadata mapping for sample documents (can be extended or overridden)
DEFAULT_METADATA_REGISTRY = {
    "employee_handbook_2024.pdf": {
        "department": "Human Resources",
        "category": "Policy",
        "year": 2024
    },
    "cloud_architecture_guide.pdf": {
        "department": "Engineering",
        "category": "Technical",
        "year": 2023
    },
    "ai_governance_policy.pdf": {
        "department": "Compliance",
        "category": "Governance",
        "year": 2024
    }
}


class DocumentIngestionPipeline:
    def __init__(self, metadata_registry: Optional[Dict[str, Dict[str, Any]]] = None):
        self.metadata_registry = metadata_registry or DEFAULT_METADATA_REGISTRY

    def ingest_pdf(self, pdf_path: str, custom_metadata: Optional[Dict[str, Any]] = None) -> List[DocumentPage]:
        """
        Ingests a single PDF, extracting text page-by-page and attaching metadata.
        Filters out empty/whitespace-only pages while recording them.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        doc_name = path.name
        doc_metadata = dict(self.metadata_registry.get(doc_name, {}))
        if custom_metadata:
            doc_metadata.update(custom_metadata)

        pages: List[DocumentPage] = []
        reader = pypdf.PdfReader(str(path))
        num_pages = len(reader.pages)

        for idx, page in enumerate(reader.pages):
            page_num = idx + 1
            raw_text = page.extract_text() or ""
            cleaned_text = " ".join(raw_text.split())

            # Even if empty, create entry or flag so calling pipeline knows
            page_meta = {
                **doc_metadata,
                "document_name": doc_name,
                "page_number": page_num,
                "total_pages": num_pages,
                "is_empty": len(cleaned_text.strip()) == 0
            }

            pages.append(DocumentPage(
                text=cleaned_text,
                page_number=page_num,
                total_pages=num_pages,
                document_name=doc_name,
                file_path=str(path.resolve()),
                metadata=page_meta
            ))

        return pages

    def ingest_directory(self, directory_path: str, custom_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> tuple[List[DocumentPage], IngestionReport]:
        """
        Ingests all PDF documents in a directory.
        Returns non-empty pages and an IngestionReport detailing counts.
        """
        dir_path = Path(directory_path)
        pdf_files = sorted(list(dir_path.glob("*.pdf")))

        valid_pages: List[DocumentPage] = []
        report = IngestionReport(total_documents=len(pdf_files))

        for pdf_file in pdf_files:
            meta = custom_metadata.get(pdf_file.name) if custom_metadata else None
            pages = self.ingest_pdf(str(pdf_file), meta)
            doc_valid = 0
            doc_empty = 0

            for p in pages:
                report.total_pages_scanned += 1
                if p.metadata.get("is_empty", False):
                    report.empty_pages_encountered += 1
                    doc_empty += 1
                else:
                    valid_pages.append(p)
                    report.total_valid_pages += 1
                    doc_valid += 1

            report.document_summaries.append({
                "document_name": pdf_file.name,
                "total_pages": len(pages),
                "valid_pages": doc_valid,
                "empty_pages": doc_empty,
                "metadata": {k: v for k, v in pages[0].metadata.items() if k not in ("is_empty", "page_number")} if pages else {}
            })

        return valid_pages, report
