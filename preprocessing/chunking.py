"""
Phase 2 (revised again): Structure-aware chunking using PubMed's actual
section labels.

The first version of this script tried to GUESS section boundaries by
searching the flattened abstract text for words like "BACKGROUND:" — but
NLM stores that information as an XML attribute (Label), not as text
inside the abstract, so the guess mostly failed. ingestion/pubmed_loader.py
now preserves that attribute as an "abstract_sections" field per paper;
this script uses it directly instead of guessing. Papers with no such
structure (a single free-text abstract, common for many journals) fall
back to the same sliding-window chunking as before.

Requires data/raw/*.json to have been fetched with the updated
pubmed_loader.py (re-run it if your raw files predate this change —
files fetched before will simply fall back to sliding-window chunking
for every paper, same as last time).

Usage:
    python preprocessing/chunking.py
"""

import argparse
import json
import re
from pathlib import Path

MIN_SECTIONS_TO_TRUST = 2
FALLBACK_CHUNK_SIZE = 500
FALLBACK_CHUNK_OVERLAP = 100


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(label: str) -> str:
    if not label:
        return "Unlabeled"
    return label.strip().title()


def sliding_window_chunks(text: str, chunk_size: int = FALLBACK_CHUNK_SIZE, chunk_overlap: int = FALLBACK_CHUNK_OVERLAP) -> list:
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


def chunk_paper(title: str, abstract: str, abstract_sections: list) -> list:
    """
    Returns [(section_label, chunk_text), ...] for one paper. Uses
    PubMed's real section labels when there are enough of them to be
    meaningful; falls back to sliding-window chunking over the whole
    title+abstract otherwise.
    """
    real_sections = [
        (normalize_label(s.get("label", "")), clean_text(s.get("text", "")))
        for s in (abstract_sections or [])
        if clean_text(s.get("text", ""))
    ]

    if len(real_sections) >= MIN_SECTIONS_TO_TRUST:
        result = []
        for label, content in real_sections:
            if len(content) > FALLBACK_CHUNK_SIZE:
                for i, sub in enumerate(sliding_window_chunks(content)):
                    sub_label = f"{label} ({i + 1})" if i > 0 else label
                    result.append((sub_label, sub))
            else:
                result.append((label, content))
        return result

    # fallback: no usable structure, chunk the whole title+abstract
    full_text = f"{title}. {abstract}" if title else abstract
    return [("Unstructured", c) for c in sliding_window_chunks(full_text)]


def process_papers(raw_dir: Path) -> list:
    processed_chunks = []
    structured_count = 0
    unstructured_count = 0

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
            abstract_sections = paper.get("abstract_sections", [])

            if not abstract.strip():
                continue

            chunks = chunk_paper(title, abstract, abstract_sections)

            if chunks and chunks[0][0] != "Unstructured":
                structured_count += 1
            else:
                unstructured_count += 1

            for i, (label, chunk_text_value) in enumerate(chunks):
                processed_chunks.append(
                    {
                        "chunk_id": f"{pmid}_{i}",
                        "pmid": pmid,
                        "chunk_index": i,
                        "section": label,
                        "text": chunk_text_value,
                        "title": title,
                        "year": paper.get("year", ""),
                        "authors": paper.get("authors", []),
                        "journal": paper.get("journal", ""),
                        "source": paper.get("source", "pubmed"),
                    }
                )

    print(f"Structured abstracts (real PubMed section labels): {structured_count}")
    print(f"Unstructured abstracts (fell back to sliding-window): {unstructured_count}")

    return processed_chunks


def main():
    parser = argparse.ArgumentParser(description="Structure-aware chunking of raw papers.")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = process_papers(raw_dir)

    out_path = out_dir / "chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nProcessed chunks: {len(chunks)}")
    print(f"Saved to {out_path}")

    if chunks:
        print("\n--- Milestone check: first chunk ---")
        first = chunks[0]
        print(f"chunk_id: {first['chunk_id']}")
        print(f"section:  {first['section']}")
        print(f"text:     {first['text'][:200]}...")


if __name__ == "__main__":
    main()
