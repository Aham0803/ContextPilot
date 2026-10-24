"""
Task 5 & Task 6: LLM Answer Generation & Source Citations
Interfaces with Ollama (mistral:latest) with strict grounding instructions,
hallucination avoidance, and citation generation.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import requests
from src.vector_store import RetrievalResult


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: List[Dict[str, Any]]  # List of dicts with {"document_name": str, "page_number": int}
    retrieved_chunks: List[RetrievalResult]
    raw_prompt: str
    model_name: str


class LLMClient:
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "mistral:latest")
        raw_base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw_base_url.rstrip("/")
        self.generate_url = f"{self.base_url}/api/generate"

    def format_prompt(self, question: str, retrieved_chunks: List[RetrievalResult]) -> str:
        """
        Builds a grounded prompt instructing the LLM to strictly adhere
        to the retrieved context without hallucination.
        """
        context_blocks = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            context_blocks.append(
                f"[Source {idx} | Document: {chunk.document_name} | Page: {chunk.page_number}]\n"
                f"{chunk.text}"
            )

        context_str = "\n\n".join(context_blocks) if context_blocks else "NO CONTEXT RETRIEVED."

        prompt = f"""You are a helpful, accurate, and truthful Document Question-Answering AI assistant.
Answer the user's question using ONLY the retrieved context below.

STRICT INSTRUCTIONS:
1. Answer using ONLY the facts directly stated in the retrieved context.
2. Avoid inventing, extrapolating, or assuming information that is not explicitly supported by the context.
3. If the retrieved context does not contain sufficient information to answer the question, clearly and explicitly state:
   "The provided documents do not contain sufficient information to answer this question."
4. Provide a concise, direct, and understandable response.
5. Do not contradict the provided context.

=== RETRIEVED CONTEXT ===
{context_str}
=========================

User Question: {question}

Grounded Answer:"""
        return prompt

    def generate_answer(
        self,
        question: str,
        retrieved_chunks: List[RetrievalResult],
        timeout: int = 60
    ) -> RAGAnswer:
        """
        Passes retrieved context and question to Ollama and extracts answer and citations.
        """
        prompt = self.format_prompt(question, retrieved_chunks)

        # Build citations (deduplicated by doc name & page)
        seen_citations = set()
        citations = []
        for chunk in retrieved_chunks:
            pair = (chunk.document_name, chunk.page_number)
            if pair not in seen_citations:
                seen_citations.add(pair)
                citations.append({
                    "document_name": chunk.document_name,
                    "page_number": chunk.page_number
                })

        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual grounding
                        "top_p": 0.9
                    }
                },
                timeout=timeout
            )
            response.raise_for_status()
            answer_text = response.json().get("response", "").strip()
        except requests.exceptions.RequestException as e:
            answer_text = (
                f"[Error connecting to Ollama LLM service at {self.base_url}: {e}. "
                "Ensure Ollama is running with `ollama run mistral`.]"
            )

        return RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            raw_prompt=prompt,
            model_name=self.model_name
        )

    @staticmethod
    def format_citations_markdown(citations: List[Dict[str, Any]]) -> str:
        """
        Formats source citations as specified in Task 6:
        Sources:
        ● document_name.pdf, Page X
        """
        if not citations:
            return "Sources:\n● None (No relevant context retrieved)"
        lines = ["Sources:"]
        for cit in citations:
            lines.append(f"● {cit['document_name']}, Page {cit['page_number']}")
        return "\n".join(lines)
