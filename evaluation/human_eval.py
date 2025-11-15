"""
evaluation/human_eval.py

Generates a structured human evaluation interface for assessing:
  - Factual accuracy
  - Visual realism
  - Creativity or style control
  - Overall preference

Outputs:
  - A CSV file to distribute to annotators (or convert to Google Form)
  - Aggregates responses from multiple annotators into summary statistics
"""

from __future__ import annotations
import os
import csv
import json
import pandas as pd
from typing import List, Dict, Optional
from utils.logging_utils import get_logger

logger = get_logger("human_eval")

RESULTS_DIR = "experiments/results"
EVAL_DIR = "evaluation/human_study"
os.makedirs(EVAL_DIR, exist_ok=True)


# ------------------------------------------------------------
# 🧩 Step 1: Prepare the evaluation dataset
# ------------------------------------------------------------
def prepare_human_eval_dataset(
    exp_names: List[str],
    num_samples: int = 5,
    output_file: str = "human_eval_samples.csv"
):
    """
    Collects a subset of generated outputs for human evaluation.

    Each row contains:
        - Experiment name
        - Prompt / caption
        - Baseline image path
        - Improved image path
        - Annotator rating fields
    """
    records = []

    for exp in exp_names:
        summary_path = os.path.join(RESULTS_DIR, f"{exp}_summary.json")
        if not os.path.exists(summary_path):
            logger.warning(f"Missing summary for {exp}")
            continue

        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        improved = data.get("improved_runs", [])
        baseline = data.get("baseline_runs", [])
        num = min(num_samples, len(improved))

        for i in range(num):
            prompt = improved[i].get("prompt", "")
            imp_imgs = improved[i].get("outputs", {}).get("images", [])
            base_imgs = baseline[i].get("outputs", {}).get("images", []) if i < len(baseline) else []

            record = {
                "experiment": exp,
                "sample_id": f"{exp}_{i+1}",
                "prompt": prompt,
                "baseline_image": base_imgs[0] if base_imgs else "",
                "improved_image": imp_imgs[0] if imp_imgs else "",
                "rating_factual": "",
                "rating_realism": "",
                "rating_style": "",
                "rating_overall": "",
                "comments": "",
            }
            records.append(record)

    # Save CSV for annotation
    output_path = os.path.join(EVAL_DIR, output_file)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"✅ Prepared human evaluation dataset: {output_path}")
    return output_path


# ------------------------------------------------------------
# 🧩 Step 2: Aggregate responses from annotators
# ------------------------------------------------------------
def aggregate_human_ratings(
    input_files: List[str],
    output_file: str = "human_eval_summary.csv"
):
    """
    Aggregates multiple annotator CSVs into a unified summary with averages.
    """
    all_data = []
    for fpath in input_files:
        if not os.path.exists(fpath):
            logger.warning(f"File not found: {fpath}")
            continue
        df = pd.read_csv(fpath)
        df["annotator"] = os.path.basename(fpath).replace(".csv", "")
        all_data.append(df)

    if not all_data:
        raise ValueError("No annotation files found.")

    combined = pd.concat(all_data, ignore_index=True)
    numeric_cols = ["rating_factual", "rating_realism", "rating_style", "rating_overall"]

    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    summary = (
        combined.groupby(["experiment"])
        [numeric_cols].mean()
        .round(2)
        .reset_index()
    )

    output_path = os.path.join(EVAL_DIR, output_file)
    summary.to_csv(output_path, index=False)
    logger.info(f"✅ Saved aggregated human ratings summary: {output_path}")
    print(summary)

    return summary


# ------------------------------------------------------------
# 🧩 Step 3: Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    # Step 1: Prepare evaluation samples for annotators
    experiments = ["ablation_diffusion", "full_model_integration", "constraint_weight_sweep"]
    dataset_path = prepare_human_eval_dataset(experiments, num_samples=5)

    # (Annotators fill the generated CSV manually or through a Google Form)
    # After collecting multiple CSVs (e.g., human_eval_annotator1.csv, etc.)

    # Step 2: Aggregate results
    example_files = [
        os.path.join(EVAL_DIR, "human_eval_annotator1.csv"),
        os.path.join(EVAL_DIR, "human_eval_annotator2.csv"),
    ]
    if all(os.path.exists(f) for f in example_files):
        aggregate_human_ratings(example_files)