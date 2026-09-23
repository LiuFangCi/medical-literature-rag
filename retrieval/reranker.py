"""
Phase 4 (revised): Cross-encoder reranking using MedCPT's Cross-Encoder.

Switched from the general-purpose ms-marco-MiniLM-L-6-v2 model to
ncbi/MedCPT-Cross-Encoder — NCBI trained this cross-encoder for exactly
this task (ranking PubMed articles against a query), matching the
domain-specific choice already made for the embedding model.

Takes a batch of candidate chunks (from dense, BM25, or hybrid search)
and re-scores them by feeding the query and each chunk's text into the
model TOGETHER, rather than comparing two pre-computed vectors like the
retrievers do. Normally run AFTER retrieval, on a wider candidate list
(e.g. top 20), to pick the true top 5.

Used as a library function by evaluation/retrieval_eval.py and
evaluation/build_eval_set.py — no command-line entry point of its own,
since it always needs a candidate list from a retriever first.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"

_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def rerank(query: str, candidates: list, top_k: int = 5) -> list:
    if not candidates:
        return []

    tokenizer, model = get_model()
    pairs = [[query, c["text"]] for c in candidates]

    with torch.no_grad():
        encoded = tokenizer(
            pairs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )
        logits = model(**encoded).logits.squeeze(dim=1)

    for c, score in zip(candidates, logits.tolist()):
        c["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
