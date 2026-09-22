"""
Phase 3a: Query the FAISS index built by embedding/embedder.py.

Given a natural-language question, embeds it with the same model used to
build the index, and returns the top-k most similar chunks along with
their source paper metadata (pmid, title, year).

Usage:
    python retrieval/dense_retriever.py --query "What are the risk factors for mortality after kidney transplantation?" --top_k 5
"""

import argparse
import json

import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def search(query: str, top_k: int = 5):
    index = faiss.read_index("data/processed/faiss.index")
    with open("data/processed/chunks_meta.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        results.append({**chunk, "score": float(score)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Search the FAISS index for relevant chunks.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top_k)

    print(f"\nTop {len(results)} results for: {args.query!r}\n")
    for rank, r in enumerate(results, start=1):
        print(f"[{rank}] score={r['score']:.3f}  pmid={r['pmid']}  ({r['year']})")
        print(f"    {r['title']}")
        print(f"    {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
