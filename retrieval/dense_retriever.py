"""
Phase 3a (revised): Query the FAISS index using MedCPT's Query Encoder.

Must use the encoder that MATCHES the one used to build the index
(embedding/embedder.py uses the Article Encoder). MedCPT's query and
article encoders are trained together to share the same vector space —
encoding the query with a different model would land it in an unrelated
space and produce meaningless similarity scores.

Usage:
    python retrieval/dense_retriever.py --query "What are the risk factors for mortality after kidney transplantation?" --top_k 5
"""

import argparse
import json

import faiss
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
MAX_LENGTH = 64  # MedCPT's query encoder is trained on short queries

_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def embed_query(query: str):
    tokenizer, model = get_model()
    with torch.no_grad():
        encoded = tokenizer(
            [query],
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MAX_LENGTH,
        )
        embeds = model(**encoded).last_hidden_state[:, 0, :]
    return embeds.numpy().astype("float32")


def search(query: str, top_k: int = 5) -> list:
    index = faiss.read_index("data/processed/faiss.index")
    with open("data/processed/chunks_meta.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    query_vec = embed_query(query)
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
    parser = argparse.ArgumentParser(description="Search the FAISS index using MedCPT.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    results = search(args.query, args.top_k)

    print(f"\nTop {len(results)} results for: {args.query!r}\n")
    for rank, r in enumerate(results, start=1):
        section = r.get("section", "")
        print(f"[{rank}] score={r['score']:.3f}  pmid={r['pmid']}  section={section}  ({r['year']})")
        print(f"    {r['title']}")
        print(f"    {r['text'][:200]}...\n")


if __name__ == "__main__":
    main()
