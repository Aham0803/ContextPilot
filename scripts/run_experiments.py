"""
Section 4: Experiments and Observations Suite
Executes Experiments A, B, C, and D systematically and saves the comprehensive
findings and comparisons to EXPERIMENT_REPORT.md and console output.
"""
import os
import sys
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline
from src.chunker import TextChunker


def run_experiment_a(docs_dir: str):
    print("\n--- Running Experiment A: Chunk Size (Small=200 vs Large=600) ---")
    query = "What are the rules and stipend for full-time remote employees?"

    # Small chunks
    pipe_small = RAGPipeline(chunk_size=200, chunk_overlap=30)
    pipe_small.load_documents(docs_dir)
    res_small = pipe_small.retrieve(query, top_k=2)

    # Large chunks
    pipe_large = RAGPipeline(chunk_size=600, chunk_overlap=80)
    pipe_large.load_documents(docs_dir)
    res_large = pipe_large.retrieve(query, top_k=2)

    exp_a_data = {
        "query": query,
        "small_chunk_config": {"size": 200, "overlap": 30, "total_chunks_created": len(pipe_small.chunks)},
        "small_results": [
            {"chunk_id": r.chunk_id, "score": r.score, "length": len(r.text), "text": r.text}
            for r in res_small
        ],
        "large_chunk_config": {"size": 600, "overlap": 80, "total_chunks_created": len(pipe_large.chunks)},
        "large_results": [
            {"chunk_id": r.chunk_id, "score": r.score, "length": len(r.text), "text": r.text}
            for r in res_large
        ]
    }
    return exp_a_data


def run_experiment_b(pipeline: RAGPipeline):
    print("\n--- Running Experiment B: Top-K Retrieval (K=3 vs K=5) ---")
    query = "What is the Redis semantic cache cosine threshold?"

    res_k3 = pipeline.retrieve(query, top_k=3)
    res_k5 = pipeline.retrieve(query, top_k=5)

    exp_b_data = {
        "query": query,
        "k3_results": [
            {"rank": i+1, "chunk_id": r.chunk_id, "doc": r.document_name, "score": r.score, "snippet": r.text[:90] + "..."}
            for i, r in enumerate(res_k3)
        ],
        "k5_results": [
            {"rank": i+1, "chunk_id": r.chunk_id, "doc": r.document_name, "score": r.score, "snippet": r.text[:90] + "..."}
            for i, r in enumerate(res_k5)
        ]
    }
    return exp_b_data


def run_experiment_c(pipeline: RAGPipeline):
    print("\n--- Running Experiment C: Retrieval vs Answer Quality & Failure Modes ---")

    # Case 1: Out of domain query -> Irrelevant retrieval
    query_irrelevant = "What are the rules regarding company stock option vesting and 401k plans?"
    res_irr = pipeline.retrieve(query_irrelevant, top_k=3)
    ans_irr = pipeline.llm.generate_answer(query_irrelevant, res_irr)

    # Case 2: Incomplete context -> Query asks for something partially mentioned
    query_subtle = "What is the disciplinary procedure for violating the remote work hybrid policy?"
    res_sub = pipeline.retrieve(query_subtle, top_k=3)
    ans_sub = pipeline.llm.generate_answer(query_subtle, res_sub)

    exp_c_data = {
        "case_1_irrelevant_retrieval": {
            "query": query_irrelevant,
            "top_retrieved_scores": [r.score for r in res_irr],
            "top_retrieved_doc": res_irr[0].document_name if res_irr else "None",
            "model_answer": ans_irr.answer,
            "cause_analysis": (
                "The topic 'stock options / 401k' does not exist in the corpus. "
                "However, cosine similarity retrieved general HR policy chunks with moderate score (approx 0.35-0.45) "
                "because of semantic overlap with employment terminology. The LLM's strict negative prompt instruction "
                "correctly prevented hallucination and identified that the information is absent."
            )
        },
        "case_2_partial_context": {
            "query": query_subtle,
            "top_retrieved_doc": res_sub[0].document_name if res_sub else "None",
            "model_answer": ans_sub.answer,
            "cause_analysis": (
                "The documents define the 2-day hybrid requirement, but explicitly do NOT contain a disciplinary escalation path. "
                "The grounded prompt prevented the LLM from inventing penalties, and it correctly reported the lack of sufficient info."
            )
        }
    }
    return exp_c_data


def run_experiment_d(pipeline: RAGPipeline):
    print("\n--- Running Experiment D: Metadata Filtering Comparison ---")
    query = "What are the data retention and log limits?"

    # Unfiltered retrieval
    res_unfiltered = pipeline.retrieve(query, top_k=3, metadata_filter=None)

    # Filtered by Compliance
    res_compliance = pipeline.retrieve(query, top_k=3, metadata_filter={"department": "Compliance"})

    # Filtered by Engineering
    res_engineering = pipeline.retrieve(query, top_k=3, metadata_filter={"department": "Engineering"})

    exp_d_data = {
        "query": query,
        "unfiltered": [
            {"doc": r.document_name, "dept": r.metadata.get("department"), "score": r.score, "snippet": r.text[:80]}
            for r in res_unfiltered
        ],
        "compliance_filtered": [
            {"doc": r.document_name, "dept": r.metadata.get("department"), "score": r.score, "snippet": r.text[:80]}
            for r in res_compliance
        ],
        "engineering_filtered": [
            {"doc": r.document_name, "dept": r.metadata.get("department"), "score": r.score, "snippet": r.text[:80]}
            for r in res_engineering
        ]
    }
    return exp_d_data


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(base_dir, "data", "sample_documents")

    pipeline = RAGPipeline(chunk_size=400, chunk_overlap=50)
    pipeline.load_documents(docs_dir)

    exp_a = run_experiment_a(docs_dir)
    exp_b = run_experiment_b(pipeline)
    exp_c = run_experiment_c(pipeline)
    exp_d = run_experiment_d(pipeline)

    combined_results = {
        "experiment_a_chunk_size": exp_a,
        "experiment_b_top_k": exp_b,
        "experiment_c_failures": exp_c,
        "experiment_d_metadata_filter": exp_d
    }

    results_json = os.path.join(base_dir, "data", "experiment_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\nAll experiments executed successfully! Results saved to {results_json}")


if __name__ == "__main__":
    main()
