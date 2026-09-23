"""
RAGAS-style generation-face evaluation, implemented directly (not via the
`ragas` package) so every step stays visible and matches the plain-language
explanation of these four metrics: Faithfulness, Answer Relevancy, Context
Precision, Context Recall.

For each config (A/B/C/D) x each question, this runs the FULL pipeline
(retrieve -> generate), then uses an LLM as a judge to score the four
metrics, and MedCPT's Query Encoder for the embedding-similarity part of
Answer Relevancy.

Requires:
  - data/evaluation/eval_set.json entries to have a "reference_answer"
    field (run build_reference_answers.py first)
  - OPENAI_API_KEY in .env

Cost note: ~4 LLM judge calls per (config, question) pair. With 4 configs
x 25 questions, that's roughly 400 small calls — cheap on gpt-5-mini
(a few dollars at most), but not free; this script will ask for
confirmation before starting.

Usage:
    python evaluation/ragas_eval.py
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.dense_retriever import embed_query  # noqa: E402
from retrieval.dense_retriever import search as dense_search  # noqa: E402
from retrieval.hybrid_retriever import search as hybrid_search  # noqa: E402
from retrieval.reranker import rerank  # noqa: E402
from generation.answer_generator import generate_answer  # noqa: E402

load_dotenv()
MODEL_NAME = "gpt-5-mini"

CONFIGS = {
    "A_dense_only": {"retrieval": "dense", "reranker": False},
    "B_hybrid": {"retrieval": "hybrid", "reranker": False},
    "C_hybrid_rerank": {"retrieval": "hybrid", "reranker": True},
    "D_dense_rerank": {"retrieval": "dense", "reranker": True},
}


def run_config(config: dict, query: str, top_k: int = 5) -> list:
    candidate_k = 20 if config["reranker"] else top_k
    results = dense_search(query, top_k=candidate_k) if config["retrieval"] == "dense" else hybrid_search(query, top_k=candidate_k)
    return rerank(query, results, top_k=top_k) if config["reranker"] else results[:top_k]


def ask_json(client: OpenAI, prompt: str) -> dict:
    """Send a prompt that asks for JSON-only output, and parse it. Strips
    markdown code fences if the model wraps its answer in them anyway."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def cosine_sim(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_and_verify_claims(client: OpenAI, source_text: str, evidence_text: str):
    """Shared by Faithfulness (source=answer) and Context Recall
    (source=reference_answer): decompose source_text into claims, check
    each against evidence_text. Returns (score, n_claims)."""
    prompt = f"""Evidence:
{evidence_text}

Text to check:
{source_text}

List each distinct factual claim in "Text to check" as a short sentence, then judge whether that claim is directly supported by "Evidence".
Respond ONLY with JSON in this exact shape: {{"claims": [{{"claim": "...", "supported": true}}]}}"""
    result = ask_json(client, prompt)
    claims = result.get("claims", [])
    if not claims:
        return None, 0
    supported = sum(1 for c in claims if c.get("supported"))
    return supported / len(claims), len(claims)


def answer_relevancy(client: OpenAI, question: str, answer_text: str, n: int = 3) -> float:
    prompt = f"""Answer:
{answer_text}

Write {n} different questions that this answer could be a response to.
Respond ONLY with JSON: {{"questions": ["...", "...", "..."]}}"""
    result = ask_json(client, prompt)
    generated_qs = result.get("questions", [])
    if not generated_qs:
        return None

    q_vec = embed_query(question)[0]
    sims = []
    for gq in generated_qs:
        gq_vec = embed_query(gq)[0]
        sims.append(cosine_sim(q_vec, gq_vec))
    return sum(sims) / len(sims)


def context_precision(client: OpenAI, question: str, contexts: list) -> float:
    numbered = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(contexts))
    prompt = f"""Question: {question}

Contexts (numbered):
{numbered}

For each numbered context, is it relevant to answering the Question?
Respond ONLY with JSON: {{"relevant": [true, false, ...]}} — same order and count as the contexts."""
    result = ask_json(client, prompt)
    relevant = result.get("relevant", [])
    if len(relevant) != len(contexts) or not any(relevant):
        return 0.0

    precisions = []
    hits = 0
    for k, is_rel in enumerate(relevant, start=1):
        if is_rel:
            hits += 1
        precisions.append((hits / k) * (1 if is_rel else 0))
    return sum(precisions) / hits


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not found in .env.")
        return
    client = OpenAI(api_key=api_key)

    eval_path = Path("data/evaluation/eval_set.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    missing_ref = [i for i, item in enumerate(eval_set) if not item.get("reference_answer")]
    if missing_ref:
        print(f"{len(missing_ref)} question(s) have no reference_answer yet.")
        print("Run evaluation/build_reference_answers.py first.")
        return

    n_calls = len(eval_set) * len(CONFIGS) * 4
    print(f"This will make roughly {n_calls} LLM judge calls (~{len(eval_set)} questions x {len(CONFIGS)} configs x 4 metrics).")
    confirm = input("Continue? [y/N] ").strip().lower()
    if confirm != "y":
        print("Cancelled.")
        return

    rows = []
    for qid, item in enumerate(eval_set, start=1):
        question = item["question"]
        reference = item["reference_answer"]

        for config_name, config in CONFIGS.items():
            print(f"[{qid}/{len(eval_set)}] {config_name}: {question[:60]}...")

            contexts = run_config(config, question)
            answer = generate_answer(question, contexts)
            contexts_text = "\n\n".join(c["text"] for c in contexts)

            faithfulness, _ = extract_and_verify_claims(client, answer, contexts_text)
            recall, _ = extract_and_verify_claims(client, reference, contexts_text)
            relevancy = answer_relevancy(client, question, answer)
            precision = context_precision(client, question, contexts)

            rows.append(
                {
                    "question_id": qid,
                    "question": question,
                    "config": config_name,
                    "retrieval": config["retrieval"],
                    "reranker": config["reranker"],
                    "faithfulness": faithfulness,
                    "answer_relevancy": relevancy,
                    "context_precision": precision,
                    "context_recall": recall,
                }
            )

    out_path = Path("data/evaluation/ragas_scores.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved RAGAS scores to {out_path}")

    print(f"\n{'Config':<20} {'Faithfulness':<14} {'AnswerRel':<12} {'CtxPrec':<10} {'CtxRecall':<10}")
    for config_name in CONFIGS:
        sub = [r for r in rows if r["config"] == config_name]

        def avg(key):
            vals = [r[key] for r in sub if r[key] is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        print(
            f"{config_name:<20} {avg('faithfulness'):<14.3f} {avg('answer_relevancy'):<12.3f} "
            f"{avg('context_precision'):<10.3f} {avg('context_recall'):<10.3f}"
        )


if __name__ == "__main__":
    main()
