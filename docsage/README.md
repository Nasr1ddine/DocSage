# DocSage — Internal Docs Chatbot

## Performance Metrics (measured on [N] documents, [X] MB total)

> Run the evaluation pipeline (`python -m docsage.src.evaluate` and the `notebooks/eval_analysis.ipynb` notebook), then replace the placeholder values below with your actual numbers.


| Metric                      | Value     |
| --------------------------- | --------- |
| Avg chat latency (p50)      | XXXms     |
| Avg chat latency (p95)      | XXXms     |
| Hallucination rate (naive)  | XX%       |
| Hallucination rate (tuned)  | XX%       |
| Hallucination reduction     | XX% fewer |
| Storage per 100-page doc    | X.X MB    |
| Chunks indexed (tuned)      | XXXX      |
| Retrieval precision (tuned) | 0.XX      |


## Architecture

```text
          ┌───────────────────────────────────┐
          │           PDF Corpus              │
          │       data/raw_pdfs/*.pdf         │
          └───────────────────────────────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │  Ingestion (CLI) │
                 │  src/ingest.py   │
                 └──────────────────┘
                    │            │
        naive 1000c │            │ tuned 512t + meta
                    ▼            ▼
           ┌──────────────┐  ┌──────────────┐
           │ Qdrant        │  │ Qdrant        │
           │ docsage_naive │  │ docsage_tuned │
           └──────────────┘  └──────────────┘
                    │            │
                    └────┬───────┘
                         ▼
               ┌─────────────────────┐
               │ Hybrid Retriever    │
               │ src/retriever.py    │
               │                     │
               │ 1. dense (OpenAI)   │
               │ 2. BM25 (rank_bm25) │
               │ 3. RRF fusion       │
               │ 4. CrossEncoder     │
               └─────────┬───────────┘
                         ▼
               ┌─────────────────────┐
               │ LlamaIndex Pipeline │
               │ src/pipeline.py     │
               │  gpt-4o-mini, T=0   │
               └─────────┬───────────┘
                         ▼
               ┌─────────────────────────┐
               │ FastAPI Backend          │
               │ src/api.py              │
               │ /, /chat, /chat/stream, │
               │ /ingest, /health        │
               └─────────┬───────────────┘
                         ▼
               ┌─────────────────────┐
               │ Frontend (HTML)     │
               │ frontend/index.html │
               │ served at /         │
               │ chat UI + SSE       │
               └─────────────────────┘
```

## Quickstart

1. **Clone and install dependencies**
   ```bash
   cd docsage
   .\.venv\Scripts\Activate.ps1   # Activate existing venv (Windows)
   python -m venv .venv           # Or create a new virtual environment
   source .venv/bin/activate      # Linux/Mac activation
   pip install -r requirements.txt
   ```
2. **Configure environment**
   ```bash
   cp .env.example .env
   # edit .env, set OPENAI_API_KEY and (optionally) QDRANT_URL / QDRANT_API_KEY
   ```
3. **Start Qdrant**
   ```bash
   docker-compose up -d
   ```
4. **Add PDFs**
   Download 5–10 public PDFs (e.g., SEC 10-Ks, arXiv ML papers, Fed reports) into `data/raw_pdfs/`.
5. **Ingest (naive, then tuned)**
   ```bash
   # naive baseline
   python -m docsage.src.ingest --strategy naive

   # tuned sentence-aware chunking + metadata
   python -m docsage.src.ingest --strategy tuned
   ```
6. **Run API** (from the project root, one level above `docsage/`)
   ```bash
   uvicorn docsage.src.api:app --reload --port 8000
   ```
7. **Open frontend**
   Navigate to **http://localhost:8000/** in your browser. The API serves the frontend automatically. Use the **Naive / Tuned** toggle to compare retrieval quality live.

## Why Hybrid Search + Reranking?

Dense vector search is excellent at capturing semantic similarity, but it is surprisingly brittle on exact-match queries: ticker symbols, specific dates, and precise numeric values often have weak cosine similarity despite being the correct answer span. This means a purely dense retriever can silently miss obviously correct chunks while still returning fluent answers, which is a classic path to subtle hallucinations in production RAG systems.

Hybrid retrieval combines dense search with BM25 lexical scoring on the same corpus, then merges the two ranked lists via Reciprocal Rank Fusion (RRF). This preserves the semantic recall benefits of dense embeddings while giving BM25 a chance to surface exact matches that dense might overlook. A cross-encoder reranker sits on top to rescore candidates using the full query–chunk pair, so the final top-k set is both semantically aligned and lexically grounded, which is critical for financial figures and technical references.

## Failure Case Study

See `notebooks/eval_analysis.ipynb` for a fully worked failure case where the tuned pipeline still retrieved the wrong chunk, how RAGAS metrics (especially faithfulness) exposed the issue, and what change you applied (e.g., adjusting chunk size/overlap or metadata) to fix it. The notebook records the **question**, **incorrect retrieved chunk**, **detection method**, **root cause analysis**, and **before/after metrics** so you can reason about real-world RAG failure modes rather than anecdotal successes.

## Eval Methodology

DocSage uses a manually curated ground-truth set of 20 questions stored in `data/eval_dataset.json`:

- 7 factual questions whose answers contain specific numbers or dates
- 7 conceptual questions requiring multi-sentence explanations
- 3 out-of-scope questions whose answers are not present in the corpus
- 3 trick questions that are easy to misattribute across similar sections

For each question, the evaluation pipeline runs the RAG query against both the **naive** and **tuned + hybrid + rerank** collections and collects:
`faithfulness`, `answer_relevancy`, `context_precision`, and `context_recall` via **RAGAS**. The notebook aggregates these into the required comparison table and derives hallucination rates (`1 - faithfulness`) so you can quantify and visualize how much the tuned pipeline reduces hallucinations relative to the naive baseline.