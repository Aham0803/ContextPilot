"""
Evaluation Execution Script
Runs the 12-question evaluation benchmark suite and saves results to data/evaluation_results.json.
"""
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline
from src.evaluation import RAGEvaluator


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, "data", "sample_documents")
    eval_dataset = os.path.join(base_dir, "data", "eval_dataset.json")

    pipeline = RAGPipeline(chunk_size=400, chunk_overlap=50)
    pipeline.load_documents(docs_dir)

    print(f"Loaded {len(pipeline.chunks)} chunks from {len(pipeline.pages)} pages across {len(pipeline.ingestion_report.document_summaries)} documents.")

    evaluator = RAGEvaluator(pipeline, eval_dataset)
    print("Executing evaluation on dataset (retrieval + LLM grounding check)...")
    summary = evaluator.evaluate(top_k=3, run_llm=True)

    print("\n" + "=" * 60)
    print("               RETRIEVAL & RAG EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Questions Evaluated       : {summary.total_questions}")
    print(f"Standard In-Domain Questions    : {summary.success_questions}")
    print(f"Negative/Edge-Case Questions    : {summary.failure_edge_case_questions}")
    print(f"Mean Retrieval Precision @ K=3  : {summary.mean_precision:.4f} ({summary.mean_precision*100:.1f}%)")
    print(f"Mean Retrieval Recall @ K=3     : {summary.mean_recall:.4f} ({summary.mean_recall*100:.1f}%)")
    print(f"Grounded Answer / Refusal Rate  : {summary.grounded_answer_rate:.4f} ({summary.grounded_answer_rate*100:.1f}%)")
    print("=" * 60)

    # Serialize results
    eval_results_file = os.path.join(base_dir, "data", "evaluation_results.json")
    results_dict = {
        "summary": {
            "total_questions": summary.total_questions,
            "mean_precision": summary.mean_precision,
            "mean_recall": summary.mean_recall,
            "grounded_answer_rate": summary.grounded_answer_rate
        },
        "questions": [
            {
                "id": r.question_id,
                "question": r.question,
                "expected_document": r.expected_document,
                "expected_page": r.expected_page,
                "expected_outcome": r.expected_outcome,
                "precision": r.precision,
                "recall": r.recall,
                "retrieved_is_relevant": r.retrieved_is_relevant,
                "answer_is_grounded": r.answer_is_grounded,
                "answer_refusal_correct": r.answer_refusal_correct,
                "generated_answer": r.generated_answer,
                "retrieved_chunks": r.retrieved_chunks
            }
            for r in summary.detailed_results
        ]
    }

    with open(eval_results_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Evaluation report JSON saved to: {eval_results_file}")


if __name__ == "__main__":
    main()
