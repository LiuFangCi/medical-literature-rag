"""
Phase 4: Hybrid retrieval — combines dense (semantic) and BM25 (keyword)
search using Reciprocal Rank Fusion (RRF).

This file does NOT reimplement retrieval logic. It imports and reuses the
dense and BM25 searchers already built in Phase 3a / this Phase, and just
fuses their rankings. This is exactly the "same code, different config"
pattern described in the comparison write-up: one shared pipeline, retrieval
strategy chosen as a parameter, not as a separate copy of the code.

Usage:
    python retrieval/hybrid_retriever.py --query "..." --top_k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.bm25_retriever import search as bm25_search  # noqa: E402
from retrieval.dense_retriever import search as dense_search  # noqa: E402


def reciprocal_rank_fusion(dense_results: list, bm25_results: list, k: int = 60, top_k: int = 5) -> list:
    """
    RRF score for a chunk = sum, over each ranking list it appears in, of
    1 / (k + rank). A chunk that ranks near the top of BOTH lists ends up
    with the highest fused score. k=60 is the constant most RAG-Fusion
    implementations use; it just softens the effect of rank 1 vs rank 2.
    """
    scores = {}
    chunk_lookup = {}

    for rank, r in enumerate(dense_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunk_lookup[cid] = r

    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
        chunk_lookup[cid] = r

    ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
    return [{**chunk_lookup[cid], "rrf_score": scores[cid]} for cid in ranked_ids]


def search(query: str, top_k: int = 5, candidate_k: int = 20) -> list:
    """candidate_k: how many results to pull from EACH retriever before
    fusing — wider than top_k so RRF has enough overlap to work with."""
    dense_results = dense_search(query, top_k=candidate_k)
    bm25_results = bm25_search(query, top_k=candidate_k)
    return reciprocal_rank_fusion(dense_results, bm25_results, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Hybrid (dense + BM25) search via RRF.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top_k)

    print(f"\nTop {len(results)} hybrid (RRF) results for: {args.query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] rrf_score={r['rrf_score']:.4f}  pmid={r['pmid']}  ({r['year']})")
        print(f"    {r['title']}")
        print(f"    {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
