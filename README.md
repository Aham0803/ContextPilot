<div align="center">

# 🧠 ContextPilot

### Enterprise-Grade Document Question-Answering System Using RAG

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Vector DB](https://img.shields.io/badge/Vector%20DB-FAISS-green)](https://github.com/facebookresearch/faiss)
[![Embeddings](https://img.shields.io/badge/Embeddings-Sentence--Transformers-orange)](https://www.sbert.net/)
[![LLM](https://img.shields.io/badge/LLM-Ollama%20%28Mistral%29-purple?logo=ollama&logoColor=white)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<p align="center">
  A production-ready Retrieval-Augmented Generation (RAG) platform featuring multi-page PDF ingestion, 
  recursive semantic chunking, dense vector retrieval with metadata filtering, strict hallucination-resistant 
  LLM grounding, automatic source citations, and an integrated evaluation benchmark suite.
</p>

[Quick Start](#-quick-start) • [Architecture](#-architecture) • [Key Features](#-key-features) • [Evaluation Benchmark](#-evaluation-benchmark) • [Experiments & Ablation](#-ablation-studies)

</div>

---

## 🌟 Key Features

- **Multi-Document PDF Ingestion**: Robust parser built on `PyPDF` that handles empty/whitespace pages, extracts clean text, and attaches structured metadata (`document_name`, `page_number`, `department`, `category`, `year`).
- **Recursive Semantic Chunking**: Configurable `RecursiveCharacterTextSplitter` respecting paragraphs and sentence boundaries to avoid context fragmentation.
- **Dense Vector Search with FAISS**: Normalized 384-dimensional embeddings via `SentenceTransformer("all-MiniLM-L6-v2")` indexed with `faiss.IndexFlatIP` for exact cosine similarity search in sub-millisecond time.
- **Structured Metadata Filtering**: Restrict searches by department (*Human Resources*, *Engineering*, *Compliance*), category, or year to eliminate cross-domain false positives.
- **Strict Grounding & Hallucination Prevention**: Prompts engineered to force Ollama (`mistral:latest`) to answer exclusively using retrieved passages. If facts are missing, the system explicitly refuses rather than hallucinating.
- **Granular Source Citations**: Every generated response includes verifiable provenance:  
  `Sources: ● employee_handbook_2024.pdf, Page 1`
- **Full-Featured Streamlit Interface**: Four dedicated interactive dashboards:
  1. 💬 **Question Answering**: Interactive search with quick-query suggestions, Top-K slider, and context inspector.
  2. 📂 **Documents & Chunks**: Scanned page statistics, empty-page tracking, and chunk explorer.
  3. 📊 **Evaluation Benchmark**: Visual Precision & Recall dashboards across 12 ground-truth questions.
  4. 🔬 **Experiments & Ablation**: Interactive comparisons for chunk sizes, Top-K depths, and filtering impact.

---

## 🏗️ Architecture & Pipeline Flow

```
                        ┌──────────────────────────────┐
                        │   Multi-Page PDF Documents   │
                        └──────────────┬───────────────┘
                                       │
                         [PyPDF Page-by-Page Ingestion]
                                       ▼
                        ┌──────────────────────────────┐
                        │ Cleaned Pages + Metadata     │
                        │ (doc_name, page, dept, year) │
                        └──────────────┬───────────────┘
                                       │
                      [LangChain Recursive Splitter]
                                       ▼
                        ┌──────────────────────────────┐
                        │ Chunks (size=400, overlap=50)│
                        └──────────────┬───────────────┘
                                       │
                     [all-MiniLM-L6-v2 Embeddings]
                                       ▼
                        ┌──────────────────────────────┐
                        │ FAISS Vector Store           │
                        │ (IndexFlatIP Inner Product)  │
                        └──────────────┬───────────────┘
                                       │
 [User Query] ──> [Query Embedding] ───┼──> [Semantic Search + Metadata Filter]
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  Top-K Retrieved Passages    │
                        └──────────────┬───────────────┘
                                       │
                  [Grounded Prompting + Mistral 7B LLM]
                                       ▼
                        ┌──────────────────────────────┐
                        │  Grounded Answer + Citations │
                        │  (document_name, Page X)     │
                        └──────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10+ (compatible with Python 3.11, 3.12, 3.13, 3.14)
- [Ollama](https://ollama.ai/) installed locally:
  ```bash
  ollama run mistral
  ```

### 2. Clone and Install
```bash
git clone https://github.com/Aham0803/ContextPilot.git
cd ContextPilot

# Install dependencies
python -m pip install -r requirements.txt
```

### 3. Launch the Application
```bash
python -m streamlit run app.py
```
Open your browser to: **`http://localhost:8501`**

---

## 📊 Evaluation Benchmark

ContextPilot includes an automated evaluation harness testing 12 distinct questions (10 in-domain targets + 2 negative controls) with ground truth relevance labels:

$$\text{Precision} = \frac{\text{Relevant Retrieved Chunks}}{\text{Total Retrieved Chunks}}$$

$$\text{Recall} = \frac{\text{Relevant Retrieved Chunks}}{\text{Total Relevant Chunks in Corpus}}$$

### Observed Benchmark Scores ($K = 3$):
| Metric | Result | Notes |
| :--- | :---: | :--- |
| **Mean Retrieval Recall** | **100.0%** | Target passages successfully captured in 100% of in-domain queries |
| **Mean Retrieval Precision** | **47.2%** | High density (1-2 target chunks per question across $K=3$) |
| **Grounded Refusal Accuracy** | **100.0%** | Zero hallucinations; out-of-domain queries cleanly rejected |

Run the benchmark anytime from the CLI:
```bash
python scripts/run_eval.py
```
Detailed results: [`EVALUATION_REPORT.md`](EVALUATION_REPORT.md)

---

## 🔬 Ablation Studies

ContextPilot includes empirical experiments evaluating core RAG hyperparameters:

- **Experiment A (Chunk Size)**: Tested 200 chars vs. 600 chars. 600 chars preserved complete multi-part policy clauses without breaking context across chunk boundaries.
- **Experiment B (Top-K Retrieval)**: Compared $K=3$ vs. $K=5$. $K=3$ delivered optimal 100% recall while avoiding lower-scoring cross-document distractors.
- **Experiment C (Failure Modes)**: Confirmed prompt engineering prevents hallucination on unrepresented subjects (e.g. stock options / 401k).
- **Experiment D (Metadata Filtering)**: Proved pre-filtering by department eliminates cross-domain polysemy between Compliance and Engineering.

Run all experiments from the CLI:
```bash
python scripts/run_experiments.py
```
Detailed observations and conceptual Q&A: [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)

---

## 📂 Repository Structure

```
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Production Python dependencies
├── README.md                      # Project documentation and guide
├── LICENSE                        # MIT Open Source License
├── CONTRIBUTING.md                # Contribution guidelines
├── .env.example                   # Environment configuration template
├── EVALUATION_REPORT.md           # Full benchmark metrics matrix
├── EXPERIMENT_REPORT.md           # Ablation experiments & conceptual analysis
├── src/                           # Modular core library
│   ├── __init__.py
│   ├── ingestion.py               # Document loading & empty-page handling
│   ├── chunker.py                 # Recursive semantic chunking
│   ├── embeddings.py              # Dense vector embedding generation
│   ├── vector_store.py            # FAISS vector database & metadata filtering
│   ├── llm.py                     # Ollama client & source citation generator
│   ├── pipeline.py                # End-to-end RAG orchestrator
│   └── evaluation.py              # Precision, recall, and grounding metrics
├── data/
│   ├── sample_documents/          # Ingestion PDFs (Employee Handbook, Cloud Guide, AI Policy)
│   ├── eval_dataset.json          # 12-question labelled ground-truth dataset
│   ├── evaluation_results.json    # Serialized evaluation outputs
│   └── experiment_results.json    # Serialized ablation metrics
├── scripts/
│   ├── create_sample_docs.py      # Multi-page PDF generator
│   ├── run_eval.py                # Benchmark evaluation CLI
│   └── run_experiments.py         # Ablation experiment runner
└── docs/
    └── specifications.pdf         # Original architectural specifications
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
