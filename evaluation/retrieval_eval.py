"""
Phase 5: Retrieval evaluation.

Turns "eyeballing which PMIDs each config found" into two standard,
comparable numbers: Recall@k and MRR (Mean Reciprocal Rank).

Requires data/evaluation/eval_set.json: a small hand-labeled set of
{question, relevant_pmids} pairs. This file does NOT come from an API —
YOU decide, by actually reading the candidate papers, which PMIDs truly
answer each question. That human judgment is the "ground truth" every
score below is measured against; the numbers are only as good as this
labeling.

Usage:
    python evaluation/retrieval_eval.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.dense_retriever import search as dense_search  # noqa: E402
from retrieval.hybrid_retriever import search as hybrid_search  # noqa: E402
from retrieval.reranker import rerank  # noqa: E402

# Same CONFIGS shape as experiments/run_comparison.py — kept identical on
# purpose so results from the two scripts line up.
CONFIGS = {
    "A_dense_only": {"retrieval": "dense", "reranker": False},
    "B_hybrid": {"retrieval": "hybrid", "reranker": False},
    "C_hybrid_rerank": {"retrieval": "hybrid", "reranker": True},
    "D_dense_rerank": {"retrieval": "dense", "reranker": True},
}


def run_config(config: dict, query: str, top_k: int = 5) -> list:
    candidate_k = 20 if config["reranker"] else top_k
    if config["retrieval"] == "dense":
        results = dense_search(query, top_k=candidate_k)
    else:
        results = hybrid_search(query, top_k=candidate_k)
    if config["reranker"]:
        results = rerank(query, results, top_k=top_k)
    else:
        results = results[:top_k]
    return results


def unique_pmids_in_order(results: list) -> list:
    """Chunks are per-paragraph, so the same paper can appear several
    times in a row (as you saw with C_hybrid_rerank). Collapse to one
    entry per paper, keeping the first (highest-ranked) occurrence —
    otherwise a config could look artificially strong just by repeating
    one paper five times."""
    seen = []
    for r in results:
        if r["pmid"] not in seen:
            seen.append(r["pmid"])
    return seen


def recall_at_k(retrieved_pmids: list, relevant_pmids: list):
    """Of the papers that SHOULD have been found, what fraction actually
    showed up in the top-k? 1.0 = every relevant paper was retrieved."""
    if not relevant_pmids:
        return None
    hits = len(set(retrieved_pmids) & set(relevant_pmids))
    return hits / len(relevant_pmids)


def reciprocal_rank(retrieved_pmids: list, relevant_pmids: list) -> float:
    """1 / (rank of the FIRST relevant paper). 1.0 = it was rank 1;
    0.5 = it was rank 2; 0.0 = no relevant paper was found at all."""
    for rank, pmid in enumerate(retrieved_pmids, start=1):
        if pmid in relevant_pmids:
            return 1 / rank
    return 0.0


def main():
    eval_path = Path("data/evaluation/eval_set.json")
    if not eval_path.exists():
        print(f"{eval_path} not found.")
        return

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    scores = {name: {"recall": [], "mrr": []} for name in CONFIGS}
    long_rows = []  # one row per (question, config) — the raw data statistical tests need

    for qid, item in enumerate(eval_set, start=1):
        question = item["question"]
        relevant = item["relevant_pmids"]

        print(f"\nQuestion: {question}")
        for config_name, config in CONFIGS.items():
            results = run_config(config, question)
            retrieved = unique_pmids_in_order(results)

            r = recall_at_k(retrieved, relevant)
            mrr = reciprocal_rank(retrieved, relevant)

            scores[config_name]["recall"].append(r)
            scores[config_name]["mrr"].append(mrr)
            long_rows.append(
                {
                    "question_id": qid,
                    "question": question,
                    "config": config_name,
                    "retrieval": config["retrieval"],
                    "reranker": config["reranker"],
                    "recall_at_5": r,
                    "mrr": mrr,
                }
            )
            print(f"  {config_name:<20} recall@5={r:.2f}  mrr={mrr:.2f}  retrieved={retrieved}")

    print(f"\n{'=' * 55}")
    print(f"{'Config':<20} {'Recall@5 (avg)':<18} {'MRR (avg)':<12}")
    print("-" * 55)
    for config_name, s in scores.items():
        avg_recall = sum(s["recall"]) / len(s["recall"])
        avg_mrr = sum(s["mrr"]) / len(s["mrr"])
        print(f"{config_name:<20} {avg_recall:<18.3f} {avg_mrr:<12.3f}")

    # save per-question raw scores — this is the file statistical_analysis.py reads
    out_dir = Path("data/evaluation")
    out_path = out_dir / "retrieval_scores.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        writer.writeheader()
        writer.writerows(long_rows)
    print(f"\nSaved per-question scores to {out_path} (used by statistical_analysis.py)")


if __name__ == "__main__":
    main()
