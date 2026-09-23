"""
Adds a "reference_answer" field to every entry in data/evaluation/eval_set.json,
required for the RAGAS Context Recall metric.

For each labeled question, pulls the full abstract(s) of its relevant_pmids
from data/raw/, and asks an LLM to synthesize a short reference answer
grounded ONLY in those abstracts. This is an LLM-ASSISTED draft, not a
substitute for your own judgment — spot-check a handful of the results
before trusting them as ground truth, the same way you spot-checked
which PMIDs were relevant earlier.

Requires OPENAI_API_KEY in .env. Cost is small: ~25 short LLM calls.

Usage:
    python evaluation/build_reference_answers.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_NAME = "gpt-5-mini"

SYSTEM_PROMPT = (
    "You write short, factual reference answers for a research-QA evaluation set. "
    "Use ONLY the provided abstract(s). Write 2-4 sentences covering the key "
    "findings relevant to the question. Do not add outside knowledge, and do not "
    "hedge or editorialize — state the findings directly, as a reference answer "
    "a grader would compare other answers against."
)


def load_papers_by_pmid() -> dict:
    papers = {}
    for f in Path("data/raw").glob("*.json"):
        with open(f, "r", encoding="utf-8") as fh:
            for paper in json.load(fh):
                papers[paper.get("pmid")] = paper
    return papers


def build_reference(client: OpenAI, question: str, abstracts: list) -> str:
    context = "\n\n".join(f"Abstract {i + 1}: {a}" for i, a in enumerate(abstracts))
    user_prompt = f"Question: {question}\n\n{context}\n\nWrite the reference answer now."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in .env.")
        return
    client = OpenAI(api_key=api_key)

    eval_path = Path("data/evaluation/eval_set.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    papers = load_papers_by_pmid()

    for i, item in enumerate(eval_set, start=1):
        if item.get("reference_answer"):
            continue  # already has one, don't waste a call re-writing it

        abstracts = [
            papers[pmid]["abstract"]
            for pmid in item["relevant_pmids"]
            if pmid in papers and papers[pmid].get("abstract")
        ]
        if not abstracts:
            print(f"[{i}] Skipping (no abstracts found): {item['question']}")
            continue

        print(f"[{i}/{len(eval_set)}] {item['question']}")
        ref = build_reference(client, item["question"], abstracts)
        item["reference_answer"] = ref
        print(f"    -> {ref[:150]}...")

        # save after every question, same pattern as build_eval_set.py
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_set, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved to {eval_path}")


if __name__ == "__main__":
    main()
