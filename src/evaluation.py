"""
Task 8: Retrieval Quality Evaluation
Implements precision and recall metrics using explicit relevance ground truths:
- Precision = Relevant retrieved chunks / Total retrieved chunks
- Recall = Relevant retrieved chunks / Total relevant chunks in evaluation dataset
Evaluates both retrieval quality and LLM grounded answer correctness.
"""
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.pipeline import RAGPipeline


@dataclass
class QuestionEvalResult:
    question_id: int
    question: str
    expected_document: str
    expected_page: int
    expected_outcome: str  # "success" or "expected_failure"
    total_relevant_in_corpus: int
    retrieved_chunks: List[Dict[str, Any]]
    relevant_retrieved_count: int
    total_retrieved_count: int
    precision: float
    recall: Optional[float]
    retrieved_is_relevant: bool
    generated_answer: str
    answer_is_grounded: bool
    answer_refusal_correct: bool


@dataclass
class EvaluationSummary:
    total_questions: int
    success_questions: int
    failure_edge_case_questions: int
    mean_precision: float
    mean_recall: float
    grounded_answer_rate: float
    detailed_results: List[QuestionEvalResult] = field(default_factory=list)


class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline, eval_dataset_path: str):
        self.pipeline = pipeline
        self.eval_dataset_path = eval_dataset_path
        with open(eval_dataset_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

    def evaluate(self, top_k: int = 3, run_llm: bool = True) -> EvaluationSummary:
        results: List[QuestionEvalResult] = []
        precisions = []
        recalls = []
        grounded_count = 0

        for item in self.dataset:
            qid = item["id"]
            question = item["question"]
            expected_doc = item["expected_document"]
            expected_page = item["expected_page"]
            expected_outcome = item.get("expected_outcome", "success")
            total_relevant = item.get("total_relevant_chunks_in_corpus", 1)

            # Retrieve chunks
            retrieval_res = self.pipeline.retrieve(question, top_k=top_k)

            relevant_retrieved = 0
            retrieved_details = []

            for r in retrieval_res:
                # Chunk is relevant if it matches the expected document and expected page
                is_rel = (
                    expected_doc != "None (Out of Domain / Absent)" and
                    r.document_name.lower() == expected_doc.lower() and
                    r.page_number == expected_page
                )
                if is_rel:
                    relevant_retrieved += 1

                retrieved_details.append({
                    "chunk_id": r.chunk_id,
                    "document": r.document_name,
                    "page": r.page_number,
                    "score": r.score,
                    "is_relevant": is_rel,
                    "text_snippet": r.text[:100] + "..."
                })

            total_retrieved = len(retrieval_res)
            prec = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0

            if total_relevant > 0:
                rec = min(1.0, relevant_retrieved / total_relevant)
                recalls.append(rec)
            else:
                # For out-of-domain questions with 0 relevant chunks in corpus:
                # If 0 were retrieved as relevant, recall is defined as 1.0 (all relevant retrieved)
                rec = 1.0 if relevant_retrieved == 0 else 0.0
                recalls.append(rec)

            precisions.append(prec)

            retrieved_is_relevant = relevant_retrieved > 0

            # Evaluate LLM answer
            generated_answer = ""
            answer_is_grounded = False
            answer_refusal_correct = False

            if run_llm:
                rag_ans = self.pipeline.llm.generate_answer(question, retrieval_res)
                generated_answer = rag_ans.answer

                lower_ans = generated_answer.lower()
                is_refusal = (
                    "not contain sufficient information" in lower_ans or
                    "does not contain" in lower_ans or
                    "insufficient information" in lower_ans or
                    "no information" in lower_ans
                )

                if expected_outcome == "expected_failure":
                    if is_refusal:
                        answer_refusal_correct = True
                        answer_is_grounded = True
                        grounded_count += 1
                else:
                    if retrieved_is_relevant and not is_refusal:
                        answer_is_grounded = True
                        grounded_count += 1
                    elif is_refusal and not retrieved_is_relevant:
                        # LLM accurately noted that retrieval failed
                        answer_is_grounded = True
                        grounded_count += 1

            results.append(QuestionEvalResult(
                question_id=qid,
                question=question,
                expected_document=expected_doc,
                expected_page=expected_page,
                expected_outcome=expected_outcome,
                total_relevant_in_corpus=total_relevant,
                retrieved_chunks=retrieved_details,
                relevant_retrieved_count=relevant_retrieved,
                total_retrieved_count=total_retrieved,
                precision=round(prec, 4),
                recall=round(rec, 4),
                retrieved_is_relevant=retrieved_is_relevant,
                generated_answer=generated_answer,
                answer_is_grounded=answer_is_grounded,
                answer_refusal_correct=answer_refusal_correct
            ))

        mean_p = sum(precisions) / len(precisions) if precisions else 0.0
        mean_r = sum(recalls) / len(recalls) if recalls else 0.0
        grounded_rate = grounded_count / len(self.dataset) if self.dataset else 0.0

        return EvaluationSummary(
            total_questions=len(self.dataset),
            success_questions=sum(1 for r in results if r.expected_outcome == "success"),
            failure_edge_case_questions=sum(1 for r in results if r.expected_outcome == "expected_failure"),
            mean_precision=round(mean_p, 4),
            mean_recall=round(mean_r, 4),
            grounded_answer_rate=round(grounded_rate, 4),
            detailed_results=results
        )
