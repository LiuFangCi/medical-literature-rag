"""
Phase 3a (revised): Embed chunks with MedCPT (medical-domain specific)
and build a FAISS index.

Uses ncbi/MedCPT-Article-Encoder, trained on 255M PubMed query-article
pairs — chosen over a general-purpose model because published
comparisons show it performing better specifically on PubMed retrieval
tasks (see the project's method-comparison write-up for citations).

IMPORTANT: MedCPT has TWO separate encoders that must be used together.
This file uses the Article Encoder to embed the corpus; the matching
Query Encoder is used in retrieval/dense_retriever.py to embed the
user's question. Using a different model — or the wrong MedCPT encoder —
on either side puts the two vectors in unrelated coordinate spaces and
silently makes the whole search meaningless.

Usage:
    python embedding/embedder.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
MAX_LENGTH = 512
BATCH_SIZE = 16


def embed_articles(pairs: list, tokenizer, model) -> np.ndarray:
    """pairs: list of [title, section_text]. Returns an (N, dim) float32 array."""
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(pairs), BATCH_SIZE):
            batch = pairs[i : i + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=MAX_LENGTH,
            )
            # MedCPT uses the [CLS] token's last hidden state as the embedding
            embeds = model(**encoded).last_hidden_state[:, 0, :]
            all_embeds.append(embeds.numpy())
            print(f"  Embedded {min(i + BATCH_SIZE, len(pairs))}/{len(pairs)}", end="\r")
    print()
    return np.vstack(all_embeds).astype("float32")


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

    print(f"Loading MedCPT Article Encoder ({MODEL_NAME})")
    print("(first run downloads the model — may take a few minutes)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    # MedCPT's article encoder was trained on [title, section_text] pairs,
    # not a single concatenated string, so we keep them as two fields.
    pairs = [[c.get("title", ""), c["text"]] for c in chunks]

    print(f"Embedding {len(pairs)} chunks...")
    embeddings = embed_articles(pairs, tokenizer, model)

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir = Path("data/processed")
    faiss.write_index(index, str(out_dir / "faiss.index"))

    with open(out_dir / "chunks_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nSaved FAISS index with {index.ntotal} vectors to data/processed/faiss.index")
    print("Saved matching metadata to data/processed/chunks_meta.json")


if __name__ == "__main__":
    main()
