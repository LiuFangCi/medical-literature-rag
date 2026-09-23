"""
Generate an answer using retrieved chunks + an LLM (OpenAI API).

Retrieval strategy is a PARAMETER, not hardcoded — same "same code,
different config" pattern used in retrieval_eval.py / run_comparison.py.
Defaults to Dense + Reranker (D_dense_rerank), matching this project's
own statistical findings: Reranker showed a significant positive main
effect on Recall@5 (p = 0.018), and Dense+Rerank scored numerically
highest among the four configs in the latest run. Retrieval type
(dense vs hybrid) did NOT show a significant difference, so either is
defensible — pass --retrieval hybrid to use Hybrid instead.

Usage:
    python generation/answer_generator.py --query "What are the risk factors for mortality after kidney transplantation?" --top_k 5
    python generation/answer_generator.py --query "..." --retrieval hybrid --no_rerank

Requires OPENAI_API_KEY in your .env file (see .env.example).
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.dense_retriever import search as dense_search  # noqa: E402
from retrieval.hybrid_retriever import search as hybrid_search  # noqa: E402
from retrieval.reranker import rerank  # noqa: E402

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


def retrieve_chunks(query: str, top_k: int, retrieval: str, use_rerank: bool) -> list:
    """The exact A/B/C/D logic from retrieval_eval.py's run_config(),
    reused here instead of duplicated, so "which config is production"
    is a one-line decision, not a second copy of this logic to keep in sync."""
    candidate_k = 20 if use_rerank else top_k

    if retrieval == "dense":
        results = dense_search(query, top_k=candidate_k)
    elif retrieval == "hybrid":
        results = hybrid_search(query, top_k=candidate_k)
    else:
        raise ValueError(f"Unknown retrieval mode: {retrieval}")

    return rerank(query, results, top_k=top_k) if use_rerank else results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Generate an answer from retrieved chunks.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--retrieval", choices=["dense", "hybrid"], default="dense",
                         help="Dense and Hybrid showed no significant difference in this project's "
                              "stats; default is Dense since it scored numerically highest with Reranker on.")
    parser.add_argument("--no_rerank", action="store_true",
                         help="Turn off the Reranker (on by default — it's the one config change "
                              "with a statistically significant effect in this project's evaluation).")
    args = parser.parse_args()
    use_rerank = not args.no_rerank

    print(f"Retrieving top {args.top_k} chunks for: {args.query!r} "
          f"(retrieval={args.retrieval}, rerank={use_rerank})")
    chunks = retrieve_chunks(args.query, args.top_k, args.retrieval, use_rerank)

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
