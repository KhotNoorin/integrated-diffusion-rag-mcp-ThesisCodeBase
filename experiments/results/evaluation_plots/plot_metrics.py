"""
experiments/results/evaluation_plots/plot_metrics.py

Generates plots from `metrics_logs.csv` for thesis visualization:
  - Metric improvements (accuracy, CLIPScore, FID)
  - Constraint sweep curves
  - Comparison of baseline vs improved models
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "../metrics_logs.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_metric_improvements():
    df = pd.read_csv(RESULTS_CSV)
    sns.set(style="whitegrid", font_scale=1.2)

    # Group by experiment and metric
    for metric_name in df["metric"].unique():
        subset = df[df["metric"] == metric_name]
        plt.figure(figsize=(8, 5))
        sns.barplot(x="experiment_name", y="delta_percent", data=subset, palette="viridis")
        plt.title(f"Improvement in {metric_name} (%)")
        plt.ylabel("Δ Percent (Improved vs Baseline)")
        plt.xlabel("Experiment")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        path = os.path.join(OUTPUT_DIR, f"{metric_name}_improvements.png")
        plt.savefig(path)
        print(f"✅ Saved: {path}")

def plot_constraint_sweep():
    df = pd.read_csv(RESULTS_CSV)
    if "constraint_weight_sweep" not in df["experiment_name"].unique():
        print("⚠️ No constraint sweep data found.")
        return

    sweep_df = df[df["experiment_name"] == "constraint_weight_sweep"]
    plt.figure(figsize=(8, 5))
    sns.lineplot(x="metric", y="delta_percent", hue="epoch", data=sweep_df, marker="o")
    plt.title("Constraint Sweep Impact on Accuracy and Control")
    plt.xlabel("Constraint Type / Metric")
    plt.ylabel("Δ Percent vs Baseline")
    plt.legend(title="Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "constraint_sweep_analysis.png"))
    print("✅ Saved: constraint_sweep_analysis.png")

if __name__ == "__main__":
    plot_metric_improvements()
    plot_constraint_sweep()