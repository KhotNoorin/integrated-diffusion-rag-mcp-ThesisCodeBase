"""
evaluation/metrics_reporter.py

Aggregates and formats all metric results from experiments for reporting.

Generates:
  - Summary tables (CSV, Markdown, LaTeX)
  - Mean/Std metrics per experiment
  - Ranking of experiments by metric improvement

Use this before writing your thesis evaluation section.
"""

from __future__ import annotations
import os
import pandas as pd
from utils.logging_utils import get_logger

logger = get_logger("metrics_reporter")

RESULTS_CSV = "experiments/results/metrics_logs.csv"
OUTPUT_DIR = "evaluation/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def summarize_metrics():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"Metrics file not found: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    if df.empty:
        raise ValueError("metrics_logs.csv is empty — run experiments first.")

    logger.info(f"Loaded {len(df)} records from metrics_logs.csv")

    # Aggregate by experiment and metric
    summary = (
        df.groupby(["experiment_name", "metric"])
        .agg({"baseline_value": ["mean", "std"], "improved_value": ["mean", "std"], "delta_percent": "mean"})
        .reset_index()
    )
    summary.columns = ["experiment", "metric", "base_mean", "base_std", "improved_mean", "improved_std", "delta_mean"]
    summary["delta_mean"] = summary["delta_mean"].round(2)
    summary["improved_mean"] = summary["improved_mean"].round(3)
    summary["base_mean"] = summary["base_mean"].round(3)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    summary.to_csv(csv_path, index=False)
    logger.info(f"Saved summary CSV: {csv_path}")

    # Save Markdown
    md_path = os.path.join(OUTPUT_DIR, "metrics_summary.md")
    summary.to_markdown(md_path, index=False)
    logger.info(f"Saved Markdown table: {md_path}")

    # Save LaTeX
    tex_path = os.path.join(OUTPUT_DIR, "metrics_summary.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(summary.to_latex(index=False, caption="Quantitative Evaluation Summary"))
    logger.info(f"Saved LaTeX table: {tex_path}")

    # Ranking by delta
    ranking = (
        summary.groupby("experiment")["delta_mean"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    rank_path = os.path.join(OUTPUT_DIR, "ranking.csv")
    ranking.to_csv(rank_path, index=False)
    logger.info(f"Saved ranking CSV: {rank_path}")

    print("\n=== 🧠 Top Experiment Improvements ===")
    print(ranking.head(5))
    return summary, ranking


if __name__ == "__main__":
    summarize_metrics()