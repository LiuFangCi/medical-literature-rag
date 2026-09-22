"""
Phase 3b: Generate an answer using retrieved chunks + an LLM (OpenAI API).

This ties Phase 3a's retriever together with an LLM call: given a question,
it retrieves the most relevant chunks, builds a prompt that asks the model
to answer using ONLY the provided context, and prints the answer along with
the papers (PMIDs) it was based on. This completes the first working
"Naive RAG" milestone from the project plan.

Usage:
    python generation/answer_generator.py --query "What are the risk factors for mortality after kidney transplantation?" --top_k 5

Requires OPENAI_API_KEY in your .env file (see .env.example).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# reuse the retriever built in Phase 3a
sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.dense_retriever import search  # noqa: E402

load_dotenv()

MODEL_NAME = "gpt-5-mini"  # cheap + fast, good enough for this task

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions strictly based on "
    "the provided excerpts from medical literature. Only use information "
    "found in the excerpts. If the excerpts don't contain enough information "
    "to answer, say so clearly instead of guessing or using outside knowledge."
)


def build_context(chunks: list) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}] PMID: {c['pmid']} | Year: {c['year']} | Title: {c['title']}\n{c['text']}"
        )
    return "\n\n".join(parts)


def generate_answer(query: str, chunks: list) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file (see .env.example).")

    client = OpenAI(api_key=api_key)
    context = build_context(chunks)

    user_prompt = (
        f"Context (excerpts from medical papers):\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context above. "
        "Cite the PMID inline whenever you use a fact, like (PMID: 12345678)."
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


def main():
    parser = argparse.ArgumentParser(description="Generate an answer from retrieved chunks.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    print(f"Retrieving top {args.top_k} chunks for: {args.query!r}")
    chunks = search(args.query, args.top_k)

    if not chunks:
        print("No chunks retrieved — did you run embedding/embedder.py first?")
        return

    print("Generating answer...\n")
    answer = generate_answer(args.query, chunks)

    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("SOURCES USED FOR RETRIEVAL")
    print("=" * 60)
    for c in chunks:
        print(f"- PMID {c['pmid']} ({c['year']}): {c['title']}")


if __name__ == "__main__":
    main()
