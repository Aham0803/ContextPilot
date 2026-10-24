# AI-Powered Document Question-Answering System Using RAG

An enterprise-grade Retrieval-Augmented Generation (RAG) system built with **Python**, **PyPDF**, **LangChain Text Splitters**, **Sentence-Transformers**, **FAISS**, and **Ollama (Mistral)**. The application extracts content from multi-page PDFs, chunks text with metadata preservation, indexes dense embeddings, performs semantic search with metadata filtering, produces strictly grounded answers with source citations, and includes an evaluation and ablation suite.

---

## 🏗️ Architecture & Pipeline Overview

```
                        ┌──────────────────────────────┐
                        │   Multi-Page PDF Documents   │
                        └──────────────┬───────────────┘
                                       │
                      [Task 1: PyPDF Text Extraction]
                                       ▼
                        ┌──────────────────────────────┐
                        │ Cleaned Pages + Metadata     │
                        │ (doc_name, page, dept, year) │
                        └──────────────┬───────────────┘
                                       │
                   [Task 2: Recursive Character Splitter]
                                       ▼
                        ┌──────────────────────────────┐
                        │ Chunks (size=400, overlap=50)│
                        └──────────────┬───────────────┘
                                       │
                     [Task 3: all-MiniLM-L6-v2 Embeddings]
                                       ▼
                        ┌──────────────────────────────┐
                        │ FAISS Vector Store           │
                        │ (IndexFlatIP Inner Product)  │
                        └──────────────┬───────────────┘
                                       │
 [User Query] ──> [Query Embedding] ───┼──> [Task 4 & 7: Semantic Retrieval + Metadata Filter]
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  Top-K Retrieved Passages    │
                        └──────────────┬───────────────┘
                                       │
               [Task 5 & 6: Grounded Prompting + Mistral LLM]
                                       ▼
                        ┌──────────────────────────────┐
                        │  Grounded Answer + Citations │
                        │  (doc_name.pdf, Page X)      │
                        └──────────────────────────────┘
```

---

## 🚀 Key Features & Task Implementation

| Task | Feature | Implementation Details |
| :--- | :--- | :--- |
| **Task 1: Ingestion** | Multi-PDF Ingestion Pipeline | Reads multiple PDFs using `pypdf`, handles empty/whitespace pages without crashing, tracks pages scanned vs. valid, and attaches structured metadata. |
| **Task 2: Chunking** | Configurable LangChain Chunking | Implements `RecursiveCharacterTextSplitter` with configurable chunk size (default 400) and overlap (default 50), preserving lineage and chunk IDs. |
| **Task 3: Embeddings** | Dense Semantic Representations | Generates $L_2$-normalized 384-dimensional embeddings via `SentenceTransformer("all-MiniLM-L6-v2")`. |
| **Task 4: Vector Store**| FAISS Vector Search | Fast, exact cosine similarity retrieval using `faiss.IndexFlatIP` with configurable Top-K (1 to 10). |
| **Task 5: Generation**  | Grounded Answering | Formulates strictly grounded responses via local Ollama (`mistral:latest`), rejecting hallucinations when information is absent. |
| **Task 6: Citations**   | Automatic Source Attribution | Formats citations for every answer: `Sources: ● document.pdf, Page X`. |
| **Task 7: Filtering**   | Structured Metadata Filtering| Enables pre/post-retrieval filtering by `Department`, `Category`, or `Year`. |
| **Task 8: Evaluation**  | Quantitative Evaluation Suite| 12-question benchmark measuring **Precision**, **Recall**, and **Grounded Refusal Rate**. |

---

## 🛠️ Technology Stack Choices

- **Language**: Python 3.10+ / 3.14
- **PDF Extraction**: `pypdf` (lightweight, zero external C dependencies, robust page-level text extraction)
- **Text Splitting**: `langchain-text-splitters` (`RecursiveCharacterTextSplitter` respecting paragraphs and sentence boundaries)
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`: fast, high semantic accuracy, 384 dimensions)
- **Vector Database**: `faiss-cpu` (`IndexFlatIP` for exact inner product / cosine similarity on normalized vectors)
- **LLM Engine**: `Ollama` with `mistral:latest` (local, private, high reasoning capability, zero API token costs)
- **Interface**: `Streamlit` (responsive web app with interactive query input, top-k sliders, chunk explorer, and evaluation dashboards)

---

## 📁 Repository Structure

```
├── app.py                         # Streamlit interactive web application
├── requirements.txt               # Python package dependencies
├── README.md                      # Architecture and execution guide
├── EXPERIMENT_REPORT.md           # Experiments A-D and answers to Section 8 questions
├── EVALUATION_REPORT.md           # Benchmark evaluation results matrix
├── src/                           # Modular source code
│   ├── __init__.py
│   ├── ingestion.py               # Task 1: Document loading & empty page handling
│   ├── chunker.py                 # Task 2: LangChain recursive text chunking
│   ├── embeddings.py              # Task 3: SentenceTransformers dense embeddings
│   ├── vector_store.py            # Task 4 & 7: FAISS vector database & metadata filtering
│   ├── llm.py                     # Task 5 & 6: Ollama client, grounded prompt & citations
│   ├── pipeline.py                # End-to-end RAG orchestrator
│   └── evaluation.py              # Task 8: Precision, recall, and grounding evaluation
├── data/
│   ├── sample_documents/          # Ingestion PDFs (Employee Handbook, Cloud Guide, AI Policy)
│   ├── eval_dataset.json          # 12-question labelled ground truth dataset
│   ├── evaluation_results.json    # JSON evaluation benchmark outputs
│   └── experiment_results.json    # JSON experiment metrics (Ablation A-D)
└── scripts/
    ├── create_sample_docs.py      # Generates multi-page test PDFs with metadata
    ├── run_eval.py                # Runs quantitative evaluation suite
    └── run_experiments.py         # Runs Experiments A, B, C, D
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10 or higher
- Ollama installed and running with `mistral`:
  ```bash
  ollama run mistral
  ```

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Generate Sample PDFs (if not already present)
```bash
python scripts/create_sample_docs.py
```

---

## 💻 Running the Application

### Launch Streamlit Web UI:
```bash
streamlit run app.py
```
Open your browser to `http://localhost:8501`.

The web application contains four interactive tabs:
1. **💬 Question Answering**: Interactive search with quick-query suggestions, Top-K slider, metadata filters, grounded answers, citations, and retrieved chunk inspection.
2. **📂 Documents & Chunks**: Overview of scanned pages, empty page handling, and chunk inspection.
3. **📊 Evaluation Benchmark**: Visual metrics cards and per-question precision/recall table.
4. **🔬 Experiments & Ablation**: Results and comparisons for Experiments A, B, C, and D.

---

## 🧪 Running Benchmarks & Experiments from CLI

### Run the Evaluation Benchmark:
```bash
python scripts/run_eval.py
```
Outputs mean Precision, Recall, and Grounding metrics across the 12 evaluation questions.

### Run Experiments A, B, C, and D:
```bash
python scripts/run_experiments.py
```
Executes:
- **Experiment A**: Chunk size comparison (200 vs 600 characters)
- **Experiment B**: Top-K retrieval comparison (K=3 vs K=5)
- **Experiment C**: Retrieval vs answer quality and failure analysis
- **Experiment D**: Metadata filtering comparison (Unfiltered vs Compliance vs Engineering)
