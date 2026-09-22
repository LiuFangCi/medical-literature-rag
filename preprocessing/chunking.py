"""
Phase 2: Text cleaning and chunking for the Medical Literature RAG project.

Reads the raw JSON files produced by ingestion/pubmed_loader.py (Phase 1),
cleans the text, splits each paper's title + abstract into overlapping
chunks, and saves everything as a single processed JSON file ready for
embedding in Phase 3.

Usage:
    python preprocessing/chunking.py --chunk_size 500 --chunk_overlap 100
"""

import argparse
import json
import re
from pathlib import Path


def clean_text(text: str) -> str:
    """Basic text cleaning: collapse whitespace, strip stray characters."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)  # collapse multiple spaces / newlines / tabs
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list:
    """
    Split text into overlapping chunks by character count.

    chunk_size / chunk_overlap are in characters for this first, simple
    version (matches the 500 / 100 starting point from the project plan).
    You can later swap this for a token-aware splitter if you want chunk
    sizes measured in LLM tokens instead of characters.
    """
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def process_papers(raw_dir: Path, chunk_size: int, chunk_overlap: int) -> list:
    """Load all raw JSON files, clean + chunk each paper, attach metadata."""
    processed_chunks = []

    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {raw_dir}. Run ingestion/pubmed_loader.py first.")
        return processed_chunks

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            papers = json.load(f)

        for paper in papers:
            pmid = paper.get("pmid", "unknown")
            title = clean_text(paper.get("title", ""))
            abstract = clean_text(paper.get("abstract", ""))

            # combine title + abstract so title context isn't lost during retrieval
            full_text = f"{title}. {abstract}" if title else abstract
            if not full_text.strip():
                continue

            chunks = chunk_text(full_text, chunk_size, chunk_overlap)

            for i, chunk in enumerate(chunks):
                processed_chunks.append(
                    {
                        "chunk_id": f"{pmid}_{i}",
                        "pmid": pmid,
                        "chunk_index": i,
                        "text": chunk,
                        "title": title,
                        "year": paper.get("year", ""),
                        "authors": paper.get("authors", []),
                        "journal": paper.get("journal", ""),
                        "source": paper.get("source", "pubmed"),
                    }
                )

    return processed_chunks


def main():
    parser = argparse.ArgumentParser(description="Clean and chunk raw papers for the RAG pipeline.")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--chunk_overlap", type=int, default=100)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = process_papers(raw_dir, args.chunk_size, args.chunk_overlap)

    out_path = out_dir / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Processed chunks: {len(chunks)}")
    print(f"Saved to {out_path}")

    if chunks:
        print("\n--- Milestone check: first chunk ---")
        first = chunks[0]
        print(f"chunk_id: {first['chunk_id']}")
        print(f"pmid:     {first['pmid']}")
        print(f"text:     {first['text'][:200]}...")


if __name__ == "__main__":
    main()
