"""
experiments/run_ablation.py

Automatically discover and run ablation experiment configs:
  - scans experiments/configs/ for files matching ablation_*.yaml
  - runs each experiment via ExperimentRunner from run_experiment.py
  - collects per-experiment summary and writes aggregated results to:
      experiments/results/ablation_summary.json
      experiments/results/ablation_summary.csv

Usage:
  # run all ablations discovered
  python experiments/run_ablation.py

  # run a named ablation only
  python experiments/run_ablation.py --names ablation_rag,ablation_diffusion

  # run a single config path
  python experiments/run_ablation.py --paths experiments/configs/ablation_rag.yaml
"""
from __future__ import annotations
import os
import sys
import argparse
import glob
import json
import csv
from datetime import datetime
from typing import List, Dict, Any

# Ensure experiments package path is resolvable
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from run_experiment import ExperimentRunner  # type: ignore
from utils.logging_utils import get_logger

logger = get_logger("run_ablation")

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
AGG_JSON = os.path.join(RESULTS_DIR, "ablation_summary.json")
AGG_CSV = os.path.join(RESULTS_DIR, "ablation_summary.csv")


def discover_ablation_configs() -> List[str]:
    """Find all ablation_*.yaml config files in experiments/configs/"""
    pattern = os.path.join(CONFIG_DIR, "ablation_*.yaml")
    files = sorted(glob.glob(pattern))
    logger.info(f"Discovered {len(files)} ablation configs.")
    return files


def run_configs(config_paths: List[str]) -> List[Dict[str, Any]]:
    """Run ExperimentRunner on each config and collect summaries"""
    summaries = []
    for cfg in config_paths:
        try:
            logger.info(f"Running ablation config: {cfg}")
            runner = ExperimentRunner(cfg)
            runner.run()
            # The ExperimentRunner saves a summary JSON in its output_dir;
            # load it if available:
            exp_name = runner.config["experiment"]["name"]
            summary_path = os.path.join(runner.output_dir, f"{exp_name}_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            else:
                # Fallback: minimal summary
                summary = {
                    "experiment": exp_name,
                    "config_path": cfg,
                    "timestamp": datetime.now().isoformat(),
                    "note": "Summary file missing - runner may have logged results elsewhere."
                }
            summaries.append(summary)
        except Exception as e:
            logger.exception(f"Failed to run config {cfg}: {e}")
            summaries.append({
                "experiment": os.path.basename(cfg),
                "config_path": cfg,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
    return summaries


def save_aggregated_json(summaries: List[Dict[str, Any]]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(AGG_JSON, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(), "summaries": summaries}, f, indent=2)
    logger.info(f"Saved aggregated JSON: {AGG_JSON}")


def save_aggregated_csv(summaries: List[Dict[str, Any]]):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Build CSV rows: experiment, key, value for flattened metrics where possible
    rows = []
    for s in summaries:
        exp = s.get("experiment") or s.get("experiment_name") or s.get("config_path")
        # Flatten evaluation metrics if present
        agg_baseline = s.get("agg_baseline", {})
        agg_improved = s.get("agg_improved", {})
        deltas = s.get("deltas_percent", {})

        # If deltas empty, try nested runs
        if not deltas and "improved_runs" in s:
            # attempt to compute simple deltas per top-level metric
            # else store placeholder
            deltas = s.get("deltas_percent", {})

        if deltas:
            for metric, delta in deltas.items():
                rows.append({
                    "experiment": exp,
                    "metric": metric,
                    "baseline_mean": agg_baseline.get(metric, ""),
                    "improved_mean": agg_improved.get(metric, ""),
                    "delta_percent": delta,
                    "timestamp": s.get("timestamp", datetime.now().isoformat()),
                })
        else:
            # fallback row
            rows.append({
                "experiment": exp,
                "metric": "N/A",
                "baseline_mean": "",
                "improved_mean": "",
                "delta_percent": "",
                "timestamp": s.get("timestamp", datetime.now().isoformat()),
            })

    # Write CSV
    keys = ["experiment", "metric", "baseline_mean", "improved_mean", "delta_percent", "timestamp"]
    with open(AGG_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info(f"Saved aggregated CSV: {AGG_CSV}")


def parse_args():
    p = argparse.ArgumentParser(description="Run all ablation experiments and aggregate results.")
    p.add_argument("--paths", type=str, default=None,
                   help="Comma-separated paths to config files to run (overrides discovery).")
    p.add_argument("--names", type=str, default=None,
                   help="Comma-separated ablation names (e.g., ablation_rag,ablation_diffusion). Resolved under experiments/configs/")
    return p.parse_args()


def main():
    args = parse_args()

    if args.paths:
        config_list = [p.strip() for p in args.paths.split(",") if p.strip()]
    elif args.names:
        names = [n.strip() for n in args.names.split(",") if n.strip()]
        config_list = []
        for n in names:
            candidate = os.path.join(CONFIG_DIR, f"{n}.yaml")
            if os.path.exists(candidate):
                config_list.append(candidate)
            else:
                logger.warning(f"Named config not found: {candidate}")
    else:
        config_list = discover_ablation_configs()

    if not config_list:
        logger.error("No ablation configs found. Exiting.")
        return

    summaries = run_configs(config_list)
    save_aggregated_json(summaries)
    save_aggregated_csv(summaries)
    logger.info("All ablation experiments finished.")


if __name__ == "__main__":
    main()