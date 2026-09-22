"""
Phase 3a: Turn chunked text into vector embeddings and build a FAISS index.

Reads data/processed/chunks.json (from preprocessing/chunking.py), embeds
each chunk's text with a local Sentence-Transformers model (no API key
needed), and saves:
  - a FAISS index of the embeddings (data/processed/faiss.index)
  - the chunk metadata in the same order as the index, so a search result
    (an index position) can be mapped back to its chunk_id / pmid / text
    (data/processed/chunks_meta.json)

Usage:
    python embedding/embedder.py
"""

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

# Small, fast, general-purpose model to start with. Later you can compare
# this against a biomedical-specific model (e.g. PubMedBERT-based) as one
# of your Phase 5 experiments.
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    chunks_path = Path("data/processed/chunks.json")
    if not chunks_path.exists():
        print(f"{chunks_path} not found. Run preprocessing/chunking.py first.")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        print("No chunks found in chunks.json.")
        return

    print(f"Loading embedding model: {MODEL_NAME}")
    print("(first run downloads the model, ~90MB — may take a few minutes)")
    model = SentenceTransformer(MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")

    # normalize vectors so inner product search behaves like cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir = Path("data/processed")
    faiss.write_index(index, str(out_dir / "faiss.index"))

    # save chunk metadata in the same order as the vectors, so FAISS result
    # position i maps directly to chunks[i]
    with open(out_dir / "chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved FAISS index with {index.ntotal} vectors to data/processed/faiss.index")
    print("Saved matching metadata to data/processed/chunks_meta.json")


if __name__ == "__main__":
    main()
