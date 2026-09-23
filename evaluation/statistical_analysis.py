"""
Statistical significance testing for the A/B/C/D comparison.

Two parallel routes, run on the same data, so they can be compared as
agreeing (stronger conclusion) or disagreeing (worth discussing):

  (1) Two-way repeated-measures ANOVA (retrieval type x reranker), with
      a normality check first. Primary metric: Recall@5. MRR is run
      through the same pipeline afterward as a robustness check.
  (2) Friedman test across all four configs, then Wilcoxon signed-rank
      for specific pairwise comparisons (e.g. A vs D isolates the
      Reranker effect).

Note on sphericity: pingouin does not compute a sphericity test for
two-way repeated-measures designs, but this doesn't matter here —
sphericity is a property of a factor with 3+ levels, and both factors in
this design (retrieval: dense/hybrid; reranker: off/on) have exactly 2
levels, where sphericity is automatically satisfied. rm_anova still
reports the Greenhouse-Geisser corrected p-value alongside the
uncorrected one as a routine safeguard.

Effect sizes are intentionally not computed or reported here.

Usage:
    python evaluation/statistical_analysis.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pingouin as pg


def find_col(df: pd.DataFrame, candidates: list) -> str:
    """pingouin's exact column naming (p-unc vs p_unc, p-val vs p-value,
    etc.) has drifted slightly across versions — try each candidate name
    in turn instead of hardcoding one and risking a KeyError."""
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of {candidates} found in columns: {list(df.columns)}")


def load_scores() -> pd.DataFrame:
    path = Path("data/evaluation/retrieval_scores.csv")
    if not path.exists():
        print(f"{path} not found. Run evaluation/retrieval_eval.py first.")
        sys.exit(1)
    return pd.read_csv(path)


def run_normality_check(df: pd.DataFrame, metric: str):
    print(f"\n--- Normality check ({metric}), by config (Shapiro-Wilk) ---")
    result = pg.normality(data=df, dv=metric, group="config")
    print(result[["W", "pval", "normal"]])
    if not result["normal"].all():
        print(
            f"NOTE: at least one group's {metric} scores depart from normality. "
            "The ANOVA result below should be read alongside the Friedman/Wilcoxon "
            "result rather than on its own."
        )


def run_two_way_anova(df: pd.DataFrame, metric: str):
    print(f"\n--- Two-way repeated-measures ANOVA ({metric}) ---")
    aov = pg.rm_anova(
        data=df,
        dv=metric,
        within=["retrieval", "reranker"],
        subject="question_id",
        detailed=True,
        correction=True,
    )
    # print every column pingouin returned (so no p-value column gets
    # silently dropped due to a naming mismatch), except effect-size
    # columns (ng2/np2/n2), which were explicitly excluded from this
    # analysis
    effsize_cols = {"ng2", "np2", "n2"}
    show_cols = [c for c in aov.columns if c not in effsize_cols]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(aov[show_cols].round(4))


def run_friedman_and_wilcoxon(df: pd.DataFrame, metric: str):
    print(f"\n--- Friedman test across A/B/C/D ({metric}) ---")
    fr = pg.friedman(data=df, dv=metric, within="config", subject="question_id")
    print(fr.round(4))

    p_col = find_col(fr, ["p-unc", "p_unc", "p-value", "pval"])
    p_value = fr[p_col].iloc[0]
    if p_value >= 0.05:
        print(f"Friedman p = {p_value:.4f} (not significant at alpha=0.05) — "
              "pairwise Wilcoxon tests below are exploratory, not confirmatory.")
    else:
        print(f"Friedman p = {p_value:.4f} (significant at alpha=0.05) — "
              "proceeding to pairwise Wilcoxon tests.")

    wide = df.pivot(index="question_id", columns="config", values=metric)

    pairs = [
        ("A_dense_only", "D_dense_rerank", "Reranker effect (Dense)"),
        ("B_hybrid", "C_hybrid_rerank", "Reranker effect (Hybrid)"),
        ("A_dense_only", "B_hybrid", "Retrieval effect (no Reranker)"),
        ("D_dense_rerank", "C_hybrid_rerank", "Retrieval effect (with Reranker)"),
    ]

    print(f"\n--- Pairwise Wilcoxon signed-rank tests ({metric}) ---")
    for a, b, label in pairs:
        if a not in wide.columns or b not in wide.columns:
            continue
        res = pg.wilcoxon(wide[a], wide[b])
        wp_col = find_col(res, ["p-val", "p_val", "p-value", "pval"])
        p = res[wp_col].iloc[0]
        print(f"{label:<35} {a} vs {b}: p = {p:.4f}")


class Tee:
    """Duplicates every print() to both the terminal and a file, so the
    full statistical report survives after you close the terminal —
    previously this only ever existed on-screen and in your screenshots."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    out_path = Path("data/evaluation/statistical_analysis_report.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as report_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, report_file)
        try:
            print(f"Statistical analysis report — generated {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

            df = load_scores()
            print(f"Loaded {len(df)} rows ({df['question_id'].nunique()} questions x {df['config'].nunique()} configs).")

            for metric in ["recall_at_5", "mrr"]:
                print(f"\n{'=' * 70}\nMETRIC: {metric}\n{'=' * 70}")
                run_normality_check(df, metric)
                run_two_way_anova(df, metric)
                run_friedman_and_wilcoxon(df, metric)
        finally:
            sys.stdout = original_stdout

    print(f"\nSaved full report to {out_path}")


if __name__ == "__main__":
    main()
