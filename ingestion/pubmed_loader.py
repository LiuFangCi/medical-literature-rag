"""
PubMed literature loader for the Medical Literature RAG project (Phase 1).

Fetches paper metadata (title, abstract, PMID, year, authors, journal) from
PubMed using the NCBI E-utilities API, and saves the results as a JSON file
that later phases (chunking -> embedding -> vector DB) will consume.

Usage:
    python pubmed_loader.py --query "chronic kidney disease AND kidney transplantation" --max_results 50

Environment variables (put these in a .env file, see .env.example):
    NCBI_API_KEY   optional, raises the rate limit from 3 req/s to 10 req/s
    NCBI_EMAIL     required by NCBI's usage policy, identifies who is calling
"""

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL = f"{EUTILS_BASE}/esearch.fcgi"
EFETCH_URL = f"{EUTILS_BASE}/efetch.fcgi"

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
NCBI_EMAIL = os.environ.get("NCBI_EMAIL", "your_email@example.com")


def search_pmids(query: str, max_results: int = 50) -> list:
    """Search PubMed and return a list of matching PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(ESEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["esearchresult"]["idlist"]


def fetch_details(pmids: list) -> list:
    """Fetch title / abstract / year / authors / journal for a batch of PMIDs."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(EFETCH_URL, params=params, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(part.itertext()).strip() for part in abstract_parts)

        year_el = article.find(".//PubDate/Year")
        if year_el is None:
            # some records only have a free-text MedlineDate, e.g. "2020 Jan-Feb"
            medline_date = article.find(".//PubDate/MedlineDate")
            year = medline_date.text[:4] if medline_date is not None else ""
        else:
            year = year_el.text

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None:
                name = last.text
                if fore is not None:
                    name = f"{fore.text} {name}"
                authors.append(name)

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "year": year,
                "authors": authors,
                "journal": journal,
                "source": "pubmed",
            }
        )

    return records


def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed articles for the RAG data pipeline.")
    parser.add_argument("--query", required=True, help='e.g. "chronic kidney disease AND dialysis"')
    parser.add_argument("--max_results", type=int, default=50)
    parser.add_argument("--out_dir", default="data/raw")
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching PubMed for: {args.query!r}")
    pmids = search_pmids(args.query, args.max_results)
    print(f"Found {len(pmids)} PMIDs")

    all_records = []
    for i in range(0, len(pmids), args.batch_size):
        batch = pmids[i : i + args.batch_size]
        print(f"Fetching details for PMIDs {i + 1}-{i + len(batch)} of {len(pmids)}")
        records = fetch_details(batch)
        all_records.extend(records)
        # be polite to NCBI's servers
        time.sleep(0.4 if NCBI_API_KEY else 1.0)

    safe_name = args.query[:40].replace(" ", "_").replace("/", "_")
    out_path = out_dir / f"{safe_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_records)} records to {out_path}")

    if all_records:
        first = all_records[0]
        print("\n--- Milestone check: first record ---")
        print(f"PMID:     {first['pmid']}")
        print(f"Title:    {first['title']}")
        print(f"Abstract: {first['abstract'][:200]}...")


if __name__ == "__main__":
    main()
