"""Hand-labeled retrieval evaluation: BM25 vs. Dense (BGE) vs. Hybrid.

Runs fully locally -- no Qdrant server and no paid API required. The dense
retriever's Qdrant-backed vector store is swapped for an in-memory numpy
cosine-similarity index (InMemoryDenseRetriever below) using the *same*
real BGEEmbedder class the production DenseRetriever uses; the real,
unmodified BM25Retriever and HybridRetriever classes are used as-is.

This deliberately does not touch RAGAS (faithfulness/answer-relevancy),
since those require an LLM-as-judge and a paid OpenAI key. It evaluates
the retrieval layer only: does the right document get found for a query.

Run: python evaluation/eval_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from ingestion.embedder import BGEEmbedder
from rag.retrievers.bm25 import BM25Retriever
from rag.retrievers.hybrid import HybridRetriever

CORPUS = [
    {"id": "1", "text": "The quarterly board meeting is scheduled for the first Tuesday of every month at 10am."},
    {"id": "2", "text": "To reset your password, click 'Forgot Password' on the login page and check your email."},
    {"id": "3", "text": "Employees are entitled to 25 days of paid annual leave per calendar year."},
    {"id": "4", "text": "The API rate limit is 100 requests per minute per API key, with a burst allowance of 20."},
    {"id": "5", "text": "Refunds are processed within 5 to 7 business days after the return is received."},
    {"id": "6", "text": "The office building has bicycle parking available in the underground garage, level B1."},
    {"id": "7", "text": "Database backups run automatically every night at 2am and are retained for 30 days."},
    {"id": "8", "text": "New employees complete a 3-day onboarding program covering company policy and tools."},
    {"id": "9", "text": "The mobile app supports biometric login via fingerprint or face recognition on supported devices."},
    {"id": "10", "text": "Customer support is available via chat and email, with phone support for enterprise accounts."},
    {"id": "11", "text": "The product roadmap for next quarter prioritizes performance improvements over new features."},
    {"id": "12", "text": "All servers must be patched within 14 days of a critical security advisory being published."},
    {"id": "13", "text": "The company matches employee retirement contributions up to 4% of base salary."},
    {"id": "14", "text": "Two-factor authentication is required for all accounts with administrative privileges."},
    {"id": "15", "text": "The warehouse ships orders same-day if placed before 2pm local time on business days."},
    {"id": "16", "text": "Code review requires at least one approval before merging into the main branch."},
    {"id": "17", "text": "The cafeteria offers vegetarian and vegan options every day, with a rotating weekly menu."},
    {"id": "18", "text": "Incident response follows a documented runbook with a 15-minute initial response SLA."},
    {"id": "19", "text": "Remote employees receive a one-time home office stipend of 500 euros."},
    {"id": "20", "text": "The recommendation engine re-trains weekly on the last 90 days of user interaction data."},
]

EVAL_QUERIES = [
    {"query": "when is the board meeting", "correct_id": "1"},
    {"query": "how do I recover a forgotten password", "correct_id": "2"},
    {"query": "how many vacation days do employees get", "correct_id": "3"},
    {"query": "what is the API throughput limit", "correct_id": "4"},
    {"query": "how long until I get my money back after a return", "correct_id": "5"},
    {"query": "where can I park my bike at the office", "correct_id": "6"},
    {"query": "how often is the database backed up", "correct_id": "7"},
    {"query": "what does new hire onboarding cover", "correct_id": "8"},
    {"query": "can I log in with my fingerprint", "correct_id": "9"},
    {"query": "how do I contact support", "correct_id": "10"},
    {"query": "what's the focus of the next product release", "correct_id": "11"},
    {"query": "how fast must critical vulnerabilities be patched", "correct_id": "12"},
    {"query": "does the company contribute to my pension", "correct_id": "13"},
    {"query": "is 2FA mandatory for admin accounts", "correct_id": "14"},
    {"query": "can I get same-day shipping", "correct_id": "15"},
    {"query": "what's required before merging code", "correct_id": "16"},
    {"query": "is there vegan food at the office", "correct_id": "17"},
    {"query": "what's the SLA for responding to an incident", "correct_id": "18"},
    {"query": "do remote workers get a home office allowance", "correct_id": "19"},
    {"query": "how frequently does the recommender model update", "correct_id": "20"},
]


class InMemoryDenseRetriever:
    """Same real BGEEmbedder the production DenseRetriever uses, with an
    in-memory numpy cosine-similarity index instead of a live Qdrant server.
    """

    def __init__(self, documents: list[dict], embedder: BGEEmbedder):
        self.documents = documents
        self.embedder = embedder
        texts = [d["text"] for d in documents]
        vectors = np.array(embedder.embed_documents(texts))
        self.vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        qvec = np.array(self.embedder.embed_query(query))
        qvec = qvec / np.linalg.norm(qvec)
        sims = self.vectors @ qvec
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [
            {"id": self.documents[i]["id"], "score": float(sims[i]), "text": self.documents[i]["text"]}
            for i in top_idx
        ]


def recall_at_k(retriever, queries: list[dict], k: int) -> float:
    hits = 0
    for q in queries:
        results = retriever.retrieve(q["query"], top_k=k)
        retrieved_ids = [r["id"] for r in results]
        if q["correct_id"] in retrieved_ids:
            hits += 1
    return hits / len(queries)


def main() -> None:
    print("Loading BGE embedder (real production embedder)...")
    embedder = BGEEmbedder()

    bm25 = BM25Retriever(CORPUS)
    dense = InMemoryDenseRetriever(CORPUS, embedder)
    hybrid = HybridRetriever(dense, bm25, alpha=0.5)

    results = {}
    for name, retriever in [("BM25", bm25), ("Dense (BGE)", dense), ("Hybrid", hybrid)]:
        r1 = recall_at_k(retriever, EVAL_QUERIES, k=1)
        r3 = recall_at_k(retriever, EVAL_QUERIES, k=3)
        results[name] = (r1, r3)
        print(f"{name:15s} Recall@1={r1*100:5.1f}%  Recall@3={r3*100:5.1f}%")

    print(f"\n{len(EVAL_QUERIES)} hand-labeled queries against a {len(CORPUS)}-document corpus.")


if __name__ == "__main__":
    main()
