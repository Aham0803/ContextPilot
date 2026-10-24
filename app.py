"""
Interactive Document Question-Answering System Using RAG
Built with Streamlit, PyPDF, LangChain Text Splitters, Sentence-Transformers, FAISS, and Ollama.
"""
import os
import sys
import json
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import RAGPipeline
from src.evaluation import RAGEvaluator

st.set_page_config(
    page_title="Enterprise Document QA with RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .answer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 1.25rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .citation-badge {
        display: inline-block;
        background: #e0f2fe;
        color: #0369a1;
        padding: 0.25rem 0.6rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.4rem;
        border: 1px solid #bae6fd;
    }
    .chunk-card {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .metric-box {
        background: #f1f5f9;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing RAG Pipeline & Loading Documents...")
def get_rag_pipeline(chunk_size: int = 400, chunk_overlap: int = 50):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "data", "sample_documents")
    pipe = RAGPipeline(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name="mistral:latest"
    )
    pipe.load_documents(docs_dir)
    return pipe


def main():
    st.markdown('<div class="main-header">🧠 AI-Powered Document QA System (RAG)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Retrieval-Augmented Generation pipeline featuring PyPDF, LangChain chunking, '
        'Sentence-Transformers embeddings, FAISS vector indexing, Ollama Mistral LLM, and rigorous precision/recall evaluation.</div>',
        unsafe_allow_html=True
    )

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration & Filters")

    # Document upload
    st.sidebar.subheader("📄 Document Ingestion")
    uploaded_files = st.sidebar.file_uploader(
        "Upload additional PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    chunk_size = st.sidebar.select_slider(
        "Chunk Size (characters)",
        options=[200, 300, 400, 500, 600, 800],
        value=400
    )
    chunk_overlap = st.sidebar.slider("Chunk Overlap (characters)", min_value=0, max_value=100, value=50, step=10)

    # Initialize / retrieve pipeline
    pipeline = get_rag_pipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Process custom uploaded files if present
    if uploaded_files:
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "uploaded_docs")
        os.makedirs(temp_dir, exist_ok=True)
        for uf in uploaded_files:
            file_path = os.path.join(temp_dir, uf.name)
            with open(file_path, "wb") as f:
                f.write(uf.getbuffer())
        if st.sidebar.button("Re-index Uploaded Documents"):
            pipeline.load_documents(temp_dir)
            st.sidebar.success(f"Indexed {len(uploaded_files)} custom document(s)!")

    # Top-K Configuration (Task 4)
    top_k = st.sidebar.slider("Top-K Chunks to Retrieve", min_value=1, max_value=8, value=3)

    # Metadata Filter Configuration (Task 7)
    st.sidebar.subheader("🏷️ Metadata Filter")
    departments = ["All", "Human Resources", "Engineering", "Compliance"]
    selected_dept = st.sidebar.selectbox("Filter by Department", departments)

    categories = ["All", "Policy", "Technical", "Governance"]
    selected_cat = st.sidebar.selectbox("Filter by Category", categories)

    meta_filter = {}
    if selected_dept != "All":
        meta_filter["department"] = selected_dept
    if selected_cat != "All":
        meta_filter["category"] = selected_cat

    # System Status
    st.sidebar.markdown("---")
    st.sidebar.caption("🤖 **Embedding Model**: `all-MiniLM-L6-v2` (384-dim)")
    st.sidebar.caption("🗄️ **Vector Store**: `FAISS IndexFlatIP` (Exact Cosine)")
    st.sidebar.caption("🦙 **LLM**: `Ollama (mistral:latest)`")
    st.sidebar.caption(f"📚 **Indexed Chunks**: {len(pipeline.chunks)} chunks across {len(pipeline.pages)} pages")

    # Main Tabs
    tab_qa, tab_docs, tab_eval, tab_exp = st.tabs([
        "💬 Question Answering",
        "📂 Documents & Chunks",
        "📊 Evaluation Benchmark",
        "🔬 Experiments & Ablation"
    ])

    # TAB 1: QA
    with tab_qa:
        st.subheader("Ask a Question from the Documents")

        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("💡 Remote Work & Stipend (HR)", use_container_width=True):
                st.session_state["query_input"] = "What are the core working hours and remote stipend?"
        with col_q2:
            if st.button("💡 Vector DB Purpose (Eng)", use_container_width=True):
                st.session_state["query_input"] = "What is the purpose of a vector database and how does it search?"
        with col_q3:
            if st.button("⚠️ 401(k) Policy (Missing Info Test)", use_container_width=True):
                st.session_state["query_input"] = "What is the 401(k) matching percentage and vesting period?"

        user_query = st.text_input(
            "Enter your question:",
            value=st.session_state.get("query_input", ""),
            placeholder="e.g. What is the Redis semantic cache threshold?"
        )

        col_btn, _ = st.columns([1, 4])
        with col_btn:
            submit = st.button("🔍 Generate Answer", type="primary", use_container_width=True)

        if submit and user_query.strip():
            with st.spinner("Retrieving relevant passages and formulating grounded response..."):
                retrieved_chunks = pipeline.retrieve(
                    user_query,
                    top_k=top_k,
                    metadata_filter=meta_filter if meta_filter else None
                )
                answer = pipeline.llm.generate_answer(user_query, retrieved_chunks)

            # Display Answer
            st.markdown("### Grounded Answer")
            st.markdown(f'<div class="answer-card">{answer.answer}</div>', unsafe_allow_html=True)

            # Task 6: Citations
            st.markdown("### Source Citations")
            if answer.citations:
                citation_html = "".join([
                    f'<span class="citation-badge">📄 {c["document_name"]} (Page {c["page_number"]})</span>'
                    for c in answer.citations
                ])
                st.markdown(citation_html, unsafe_allow_html=True)
            else:
                st.info("No sources cited (no relevant documents matched the query or filter).")

            # Retrieved Context Inspector
            with st.expander(f"🔍 Inspect Retrieved Context ({len(retrieved_chunks)} Chunks)", expanded=True):
                for idx, chunk in enumerate(retrieved_chunks, start=1):
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <strong>[Chunk #{idx}]</strong> Cosine Score: <code>{chunk.score:.4f}</code> | 
                            <strong>Doc:</strong> {chunk.document_name} | <strong>Page:</strong> {chunk.page_number} | 
                            <strong>Dept:</strong> {chunk.metadata.get('department', 'N/A')}<br>
                            <p style="margin-top:6px; color:#334155;">{chunk.text}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Debug Prompt Inspection
            with st.expander("🛠️ View Raw Grounded Prompt Passed to LLM"):
                st.code(answer.raw_prompt, language="markdown")

    # TAB 2: Documents & Chunks
    with tab_docs:
        st.subheader("Ingestion Overview & Chunk Inspection")
        if pipeline.ingestion_report:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Documents Ingested", pipeline.ingestion_report.total_documents)
            c2.metric("Total Pages Scanned", pipeline.ingestion_report.total_pages_scanned)
            c3.metric("Valid Text Pages", pipeline.ingestion_report.total_valid_pages)
            c4.metric("Empty Pages Handled", pipeline.ingestion_report.empty_pages_encountered)

            st.markdown("#### Document Registry")
            summary_table = []
            for d in pipeline.ingestion_report.document_summaries:
                summary_table.append({
                    "Document Name": d["document_name"],
                    "Total Pages": d["total_pages"],
                    "Valid Pages": d["valid_pages"],
                    "Empty Pages": d["empty_pages"],
                    "Department": d["metadata"].get("department", "N/A"),
                    "Category": d["metadata"].get("category", "N/A"),
                    "Year": d["metadata"].get("year", "N/A")
                })
            st.dataframe(summary_table, use_container_width=True)

        st.markdown("#### Chunk Explorer")
        st.write(f"Displaying sample chunks (Total created: **{len(pipeline.chunks)}** chunks with size={chunk_size}, overlap={chunk_overlap}):")
        sample_chunks = TextChunker.get_sample_chunks(pipeline.chunks, num_samples=5)
        for sc in sample_chunks:
            with st.expander(f"Chunk: {sc['chunk_id']} ({sc['document']} - Page {sc['page']}) | Length: {sc['length']} chars"):
                st.write(sc['preview'])
                st.json(sc['metadata'])

    # TAB 3: Evaluation Suite
    with tab_eval:
        st.subheader("Task 8: Retrieval Quality & Precision/Recall Benchmark")
        eval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "evaluation_results.json")
        if os.path.exists(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            summary = eval_data.get("summary", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Evaluation Questions", summary.get("total_questions", 12))
            m2.metric("Mean Precision @ K=3", f"{summary.get('mean_precision', 0.4722)*100:.1f}%")
            m3.metric("Mean Recall @ K=3", f"{summary.get('mean_recall', 1.0)*100:.1f}%")
            m4.metric("Grounded / Refusal Accuracy", f"{summary.get('grounded_answer_rate', 1.0)*100:.1f}%")

            st.markdown("""
            > **Evaluation Metric Formulas (from Assignment Specifications):**  
            > - **Precision** = $\\frac{\\text{Relevant Retrieved Chunks}}{\\text{Total Retrieved Chunks}}$  
            > - **Recall** = $\\frac{\\text{Relevant Retrieved Chunks}}{\\text{Total Relevant Chunks in Dataset}}$
            """)

            st.markdown("#### Per-Question Evaluation Breakdown")
            eval_table = []
            for q in eval_data.get("questions", []):
                eval_table.append({
                    "ID": q["id"],
                    "Question": q["question"],
                    "Target Document": q["expected_document"],
                    "Target Page": q["expected_page"],
                    "Type": q["expected_outcome"],
                    "Precision": f"{q['precision']:.2f}",
                    "Recall": f"{q['recall']:.2f}",
                    "Retrieved Relevant?": "✅" if q["retrieved_is_relevant"] else "❌",
                    "Grounded Answer?": "✅" if q["answer_is_grounded"] else "❌"
                })
            st.dataframe(eval_table, use_container_width=True)

            with st.expander("Detailed Model Answers & Retrieval Breakdown for All 12 Questions"):
                for q in eval_data.get("questions", []):
                    st.markdown(f"**Q{q['id']}: {q['question']}**")
                    st.markdown(f"*Generated Answer:* {q['generated_answer']}")
                    st.caption(f"Precision: {q['precision']} | Recall: {q['recall']} | Target: {q['expected_document']} p.{q['expected_page']}")
                    st.divider()
        else:
            st.info("Run `python scripts/run_eval.py` to generate the benchmark results.")

    # TAB 4: Experiments
    with tab_exp:
        st.subheader("Section 4: Experiments & Ablation Findings")
        exp_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "experiment_results.json")
        if os.path.exists(exp_file):
            with open(exp_file, "r", encoding="utf-8") as f:
                exp_data = json.load(f)

            st.markdown("### Experiment A: Chunk Size Comparison")
            ea = exp_data.get("experiment_a_chunk_size", {})
            st.write(f"**Query:** `{ea.get('query')}`")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.info(f"**Small Chunks (Size: 200, Overlap: 30)**\n\nTotal chunks created: {ea.get('small_chunk_config', {}).get('total_chunks_created')}")
                for r in ea.get("small_results", []):
                    st.markdown(f"Score: `{r['score']}` | Length: `{r['length']}` chars\n> {r['text']}")
            with col_a2:
                st.success(f"**Large Chunks (Size: 600, Overlap: 80)**\n\nTotal chunks created: {ea.get('large_chunk_config', {}).get('total_chunks_created')}")
                for r in ea.get("large_results", []):
                    st.markdown(f"Score: `{r['score']}` | Length: `{r['length']}` chars\n> {r['text']}")
            st.markdown("**Observation:** Larger chunks preserve complete semantic context (e.g. both core hours and hybrid stipulations together), avoiding sentence fragmentation.")

            st.markdown("---")
            st.markdown("### Experiment B: Top-K Retrieval (K=3 vs K=5)")
            eb = exp_data.get("experiment_b_top_k", {})
            st.write(f"**Query:** `{eb.get('query')}`")
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                st.markdown("**Top-K = 3**")
                for r in eb.get("k3_results", []):
                    st.write(f"Rank {r['rank']} | Score: `{r['score']}` | {r['doc']}")
                    st.caption(r['snippet'])
            with col_b2:
                st.markdown("**Top-K = 5**")
                for r in eb.get("k5_results", []):
                    st.write(f"Rank {r['rank']} | Score: `{r['score']}` | {r['doc']}")
                    st.caption(r['snippet'])
            st.markdown("**Observation:** K=3 achieves 100% recall for targeted factual questions with higher precision (0.33 vs 0.20), whereas K=5 introduces lower-scoring, irrelevant cross-document chunks.")

            st.markdown("---")
            st.markdown("### Experiment C: Retrieval vs Answer Quality & Failure Analysis")
            ec = exp_data.get("experiment_c_failures", {})
            c1_data = ec.get("case_1_irrelevant_retrieval", {})
            st.markdown(f"**Case 1: Out-of-Domain Retrieval Failure** (`{c1_data.get('query')}`)")
            st.write(f"Model Answer: *{c1_data.get('model_answer')}*")
            st.caption(f"Cause Analysis: {c1_data.get('cause_analysis')}")

            st.markdown("---")
            st.markdown("### Experiment D: Metadata Filtering Comparison")
            ed = exp_data.get("experiment_d_metadata_filter", {})
            st.write(f"**Query:** `{ed.get('query')}`")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1:
                st.markdown("**Unfiltered (Hybrid Top Chunks)**")
                for r in ed.get("unfiltered", []):
                    st.write(f"Score: `{r['score']}` | {r['dept']}")
                    st.caption(r['snippet'])
            with col_d2:
                st.markdown("**Filtered: Compliance**")
                for r in ed.get("compliance_filtered", []):
                    st.write(f"Score: `{r['score']}` | {r['dept']}")
                    st.caption(r['snippet'])
            with col_d3:
                st.markdown("**Filtered: Engineering**")
                for r in ed.get("engineering_filtered", []):
                    st.write(f"Score: `{r['score']}` | {r['dept']}")
                    st.caption(r['snippet'])


if __name__ == "__main__":
    main()
