"""
experiments/run_experiment.py

Central experiment runner for:
  - Ablation studies (RAG-only, Diffusion-only, etc.)
  - Full model integration tests
  - Constraint weight sweeps

Loads config YAML files, executes pipelines, evaluates results,
and logs all metrics to experiments/results/metrics_logs.csv.

Usage:
  python experiments/run_experiment.py --config experiments/configs/full_model.yaml
"""

from __future__ import annotations
import os
import sys
import yaml
import csv
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any

import torch

# Local imports from project structure
from pipelines.text_generation import TextGenerationPipeline
from pipelines.image_generation import ImageGenerationPipeline
from pipelines.multimodal_generation import MultimodalGenerationPipeline
from pipelines.evaluation_pipeline import EvaluationPipeline
from utils.logging_utils import get_logger

logger = get_logger("run_experiment")

RESULTS_DIR = "experiments/results"
LOG_CSV = os.path.join(RESULTS_DIR, "metrics_logs.csv")


# ------------------------------------------------------------
# 📄 Helper Functions
# ------------------------------------------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def log_metrics_csv(row: Dict[str, Any]):
    """
    Appends one experiment's metrics to metrics_logs.csv
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(LOG_CSV)

    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_name",
                "epoch",
                "metric",
                "baseline_value",
                "improved_value",
                "delta_percent",
                "timestamp",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_metrics(summary: Dict[str, Any], experiment_name: str):
    """
    Logs summary metrics to CSV and prints formatted results.
    """
    agg_base = summary.get("agg_baseline", {})
    agg_improved = summary.get("agg_improved", {})
    deltas = summary.get("deltas_percent", {})
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for metric, delta in deltas.items():
        log_metrics_csv({
            "experiment_name": experiment_name,
            "epoch": summary.get("epoch", 1),
            "metric": metric,
            "baseline_value": agg_base.get(metric, 0.0),
            "improved_value": agg_improved.get(metric, 0.0),
            "delta_percent": delta,
            "timestamp": timestamp,
        })

    logger.info(f"✅ Logged {len(deltas)} metrics for experiment: {experiment_name}")


# ------------------------------------------------------------
# 🧠 Experiment Runner Class
# ------------------------------------------------------------
class ExperimentRunner:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.exp_name = self.config["experiment"]["name"]
        self.output_dir = self.config["experiment"].get("output_dir", RESULTS_DIR)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"🧩 Loaded config for experiment: {self.exp_name}")

    def run(self):
        model_type = self.config["model"]["type"]
        logger.info(f"🚀 Starting experiment: {self.exp_name} ({model_type})")

        # Choose pipeline
        if model_type == "text_only":
            pipeline = TextGenerationPipeline()
        elif model_type == "diffusion_only":
            pipeline = ImageGenerationPipeline()
        elif model_type == "multimodal":
            pipeline = MultimodalGenerationPipeline()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Evaluation setup
        evaluation = EvaluationPipeline(prompts=self._get_eval_prompts(), out_dir=self.output_dir)

        # Constraints setup
        constraints = self.config.get("constraints", {})
        rag_k = self.config.get("retrieval", {}).get("top_k", 3)

        # Run evaluation
        summary = evaluation.run_experiment(
            constraints_baseline={"style": None, "factual": False, "ethical": False, "diversity": False},
            constraints_improved=constraints,
            num_images=1,
            rag_k=rag_k,
            multimodal=model_type == "multimodal",
        )

        summarize_metrics(summary, self.exp_name)
        self._save_summary(summary)

        logger.info(f"✅ Experiment '{self.exp_name}' completed successfully!")

    # --------------------------------------------------------
    def _get_eval_prompts(self):
        prompts = self.config.get("evaluation", {}).get("eval_prompts")
        if prompts:
            return prompts
        # Fallback to generic ones
        return [
            "A photorealistic portrait of a scientist working in a futuristic lab.",
            "A cinematic illustration of an astronaut reading a book on Mars.",
        ]

    def _save_summary(self, summary: Dict[str, Any]):
        out_path = os.path.join(self.output_dir, f"{self.exp_name}_summary.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary JSON: {out_path}")


# ------------------------------------------------------------
# 🧪 Main CLI Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multimodal experiment from config file")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML (e.g., experiments/configs/full_model.yaml)",
    )
    args = parser.parse_args()

    runner = ExperimentRunner(args.config)
    runner.run()

    # Optional: Auto-generate plots & tables after run
    try:
        os.system("python experiments/results/evaluation_plots/plot_metrics.py")
        os.system("python experiments/results/comparison_tables/generate_tables.py")
    except Exception as e:
        logger.warning(f"Post-processing failed: {e}")
