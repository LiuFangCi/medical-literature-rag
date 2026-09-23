"""
One-off check: how many papers ended up structured vs unstructured after
chunking.py — counted at the PAPER level (unique PMID), not the chunk
level (searching "Unstructured" in chunks.json counts chunks, and one
unstructured paper can produce several chunks via the sliding window).

Usage:
    python count_structured.py
"""
import json
from pathlib import Path

with open("data/processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# a paper's chunks are either ALL "Unstructured" or none are, so the
# first chunk seen for a given pmid tells us which bucket it's in
pmid_section = {}
for c in chunks:
    pmid_section.setdefault(c["pmid"], c["section"])

unstructured = sum(1 for s in pmid_section.values() if s == "Unstructured")
structured = len(pmid_section) - unstructured

print(f"Structured papers:   {structured}")
print(f"Unstructured papers: {unstructured}")
print(f"Total papers:        {len(pmid_section)}")
print(f"Total chunks:        {len(chunks)}")
