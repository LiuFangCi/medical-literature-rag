"""
Interactive helper for building data/evaluation/eval_set.json.

For each candidate question, this pools candidates from EVERY config
being compared (dense, hybrid, dense+rerank, hybrid+rerank) — not just
one of them — then shows the merged, de-duplicated list for you to
label. This matters: if the candidate list only came from one retriever
(e.g. hybrid+rerank), the ground truth would be structurally biased
toward whatever that retriever tends to find, unfairly penalizing the
other configs in the comparison. This is the standard "pooling" fix used
in IR evaluation (e.g. TREC) for exactly this problem.

Progress is saved after every question, so you can stop and resume any
time — questions already labeled in a previous run are skipped
automatically.

Usage:
    python evaluation/build_eval_set.py
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from retrieval.dense_retriever import search as dense_search  # noqa: E402
from retrieval.hybrid_retriever import search as hybrid_search  # noqa: E402
from retrieval.reranker import rerank  # noqa: E402

# Seed list of candidate questions for a CKD / kidney-transplant corpus.
# EDIT THIS: remove any question that clearly has no good match once you
# see its candidates (type 's' to skip it below, or delete the line),
# and add your own questions freely — this is meant to be a starting
# point, not a fixed list. Aim for a spread of easy (one clear answer
# paper) and hard (several relevant papers, or none) questions; a mix is
# more informative for the ablation than 25 near-identical questions.
CANDIDATE_QUESTIONS = [
    "What are the risk factors for mortality after kidney transplantation?",
    "How does survival compare between deceased donor transplantation and continued dialysis?",
    "What factors affect long-term kidney graft survival?",
    "How does donor age affect transplant outcomes?",
    "What is the impact of expanded criteria donors on transplant outcomes?",
    "How does recipient age affect the survival benefit of transplantation?",
    "What are the complications of kidney transplantation?",
    "How does diabetes affect outcomes after kidney transplantation?",
    "What is the role of immunosuppression in preventing graft rejection?",
    "How does chronic kidney disease progress before dialysis is needed?",
    "What biomarkers are used to monitor kidney function after transplant?",
    "How does quality of life change after kidney transplantation compared to dialysis?",
    "What are the risk factors for chronic kidney disease progression?",
    "How does cardiovascular disease relate to chronic kidney disease outcomes?",
    "What is the effect of dialysis modality on patient outcomes?",
    "How does donation after circulatory death compare to donation after brain death?",
    "What factors predict acute kidney injury after transplantation?",
    "How does delayed graft function affect long-term outcomes?",
    "What is the relationship between proteinuria and kidney disease progression?",
    "How effective are different immunosuppressive regimens at preventing rejection?",
    "What role does patient adherence play in transplant outcomes?",
    "How does living donor transplantation compare to deceased donor transplantation?",
    "What are the predictors of early hospital readmission after kidney transplant?",
    "How does hypertension management affect chronic kidney disease progression?",
    "What is the incidence of post-transplant malignancy in kidney transplant recipients?",
    # --- 補充候選:針對截圖裡出現過、還沒被當成答案的論文各自設計的問題 ---
    "How does cognitive function differ across different kidney replacement therapies?",
    "What surgical techniques improve efficiency in paired kidney transplantation?",
    "What is the audiological or hearing health profile of transplant candidates?",
    "How does a Mediterranean diet affect nutrition in chronic kidney disease patients?",
    "How does ambient temperature or climate affect chronic kidney disease risk?",
    "What is the relationship between anemia and iron regulation in chronic kidney disease?",
    "How accurate is combining cystatin C and creatinine for estimating kidney function decline?",
    "What is the prevalence of periodontitis among kidney transplant recipients?",
    "What treatment options exist for osteoporosis in kidney transplant recipients?",
    "How is native kidney nephrectomy managed in polycystic kidney disease transplant candidates?",
    "What are the outcomes of dual versus single kidney transplantation using hard-to-place donor kidneys?",
    "How is hepatitis C managed in kidney transplant candidates in the era of direct-acting antivirals?",
]


def get_pooled_candidates(question: str, pool_size: int = 6) -> list:
    """
    Pull top candidates from every config in the comparison, then merge
    and de-duplicate by PMID. This "pooling" is what keeps the ground
    truth fair: a paper only needs to be found by ONE of the four
    configs to have a chance of being labeled relevant, instead of only
    papers the hybrid+rerank pipeline happens to surface.
    """
    dense_top = dense_search(question, top_k=pool_size)
    hybrid_top = hybrid_search(question, top_k=pool_size)
    dense_reranked = rerank(question, dense_search(question, top_k=20), top_k=pool_size)
    hybrid_reranked = rerank(question, hybrid_search(question, top_k=20), top_k=pool_size)

    pooled = {}
    for results in (dense_top, hybrid_top, dense_reranked, hybrid_reranked):
        for r in results:
            if r["pmid"] not in pooled:
                pooled[r["pmid"]] = r

    return list(pooled.values())


def load_existing() -> list:
    path = Path("data/evaluation/eval_set.json")
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save(eval_set: list):
    path = Path("data/evaluation/eval_set.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)


def main():
    eval_set = load_existing()
    already_labeled = {item["question"] for item in eval_set}

    for question in CANDIDATE_QUESTIONS:
        if question in already_labeled:
            continue

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print("=" * 70)

        candidates = get_pooled_candidates(question)

        # collapse to one entry per paper so you're not shown the same
        # paper's chunks as if they were different candidates
        seen = {}
        for c in candidates:
            if c["pmid"] not in seen:
                seen[c["pmid"]] = c
        options = list(seen.values())

        for i, c in enumerate(options, start=1):
            print(f"\n[{i}] PMID {c['pmid']} ({c['year']})")
            print(f"    {c['title']}")
            print(f"    {c['text'][:220]}...")

        print("\nWhich of the above actually answer this question?")
        print("Numbers separated by commas (e.g. 1,3) | 's' = skip this question | 'q' = quit and save")
        answer = input("> ").strip()

        if answer.lower() == "q":
            break
        if answer.lower() == "s" or answer == "":
            continue

        try:
            picked = [int(x.strip()) for x in answer.split(",")]
            relevant_pmids = [options[i - 1]["pmid"] for i in picked if 1 <= i <= len(options)]
        except (ValueError, IndexError):
            print("Couldn't parse that input — skipping this question.")
            continue

        if not relevant_pmids:
            print("No valid picks — skipping this question.")
            continue

        eval_set.append({"question": question, "relevant_pmids": relevant_pmids})
        save(eval_set)
        print(f"Saved. {len(eval_set)} questions labeled so far.")

    print(f"\nStopped — {len(eval_set)} questions currently in data/evaluation/eval_set.json")


if __name__ == "__main__":
    main()
