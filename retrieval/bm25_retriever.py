"""
Phase 4: BM25 (keyword-based) retrieval.

Reuses the same data/processed/chunks_meta.json that embedder.py already
produced — no re-embedding needed. BM25 doesn't understand meaning like
dense embeddings do, but it's very good at exact term matches (drug names,
PMIDs, acronyms, dosages) that embeddings sometimes miss.

Usage:
    python retrieval/bm25_retriever.py --query "..." --top_k 5
"""

import argparse
import json
import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def load_chunks() -> list:
    with open("data/processed/chunks_meta.json", "r", encoding="utf-8") as f:
        return json.load(f)


def search(query: str, top_k: int = 5) -> list:
    chunks = load_chunks()
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in ranked_idx:
        results.append({**chunks[idx], "score": float(scores[idx])})
    return results


def main():
    parser = argparse.ArgumentParser(description="BM25 keyword search over the chunk corpus.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top_k)

    print(f"\nTop {len(results)} BM25 results for: {args.query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] score={r['score']:.3f}  pmid={r['pmid']}  ({r['year']})")
        print(f"    {r['title']}")
        print(f"    {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
