"""
experiments/results/comparison_tables/generate_tables.py

Generates formatted tables (CSV, Markdown, and LaTeX) comparing:
  - Baseline vs Improved models across metrics
  - Constraint sweeps summary
"""

import os
import pandas as pd

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "../metrics_logs.csv")
OUTPUT_DIR = os.path.dirname(__file__)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_comparison_table():
    df = pd.read_csv(RESULTS_CSV)
    grouped = (
        df.groupby(["experiment_name", "metric"])
        .agg({"baseline_value": "mean", "improved_value": "mean", "delta_percent": "mean"})
        .reset_index()
    )

    # Round and sort for readability
    grouped["baseline_value"] = grouped["baseline_value"].round(3)
    grouped["improved_value"] = grouped["improved_value"].round(3)
    grouped["delta_percent"] = grouped["delta_percent"].round(2)

    # Save as CSV
    csv_path = os.path.join(OUTPUT_DIR, "comparison_summary.csv")
    grouped.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")

    # Save as Markdown
    md_path = os.path.join(OUTPUT_DIR, "comparison_summary.md")
    grouped.to_markdown(md_path, index=False)
    print(f"✅ Saved: {md_path}")

    # Save as LaTeX (for thesis)
    latex_path = os.path.join(OUTPUT_DIR, "comparison_summary.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(grouped.to_latex(index=False, caption="Experiment Comparison Summary"))
    print(f"✅ Saved: {latex_path}")

if __name__ == "__main__":
    make_comparison_table()