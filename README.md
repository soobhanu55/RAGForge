# RAGForge

A RAG pipeline with a built-in optimizer that grid-searches retrieval strategies (dense, BM25, hybrid) and chunking configs against RAGAS metrics, then picks the best-performing one automatically.

![Retrieval evaluation, terminal recording](docs/demo.gif)

## The finding that matters

| Strategy | Recall@1 | Recall@3 |
|---|---|---|
| BM25 | 75.0% | 95.0% |
| **Dense (BGE)** | **100.0%** | 100.0% |
| Hybrid | 95.0% | 100.0% |

**Hybrid scored *worse* than pure Dense at Recall@1.** The default 50/50 blend let BM25's weaker ranking pull a wrong document above the correct one, even though Dense alone had it right. Kept in as-is — real evidence that combining retrieval strategies isn't automatically an improvement, blend weighting needs tuning per corpus.

## How it works

```
Documents → Chunker (fixed/semantic/sliding) → BGE Embedder → Qdrant
User query → {Dense | BM25 | Hybrid} retriever → Cross-encoder reranker → LLM → Answer
Eval dataset → Optimizer grid-search → best config auto-selected (scored via RAGAS)
```

## Run it

```bash
docker run -p 6333:6333 qdrant/qdrant
pip install -r requirements.txt
uvicorn backend.main:app --reload        # API: localhost:8000/docs
streamlit run dashboard/streamlit_app.py # UI: localhost:8501
```

**Known gap:** CI/CD workflows exist but the test/eval steps are still placeholder `echo` commands — stated here rather than left for someone to discover by opening the file.

Full architecture diagram, feature list, and setup details in [`docs/DETAILS.md`](docs/DETAILS.md).
