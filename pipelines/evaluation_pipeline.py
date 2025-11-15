"""
pipelines/evaluation_pipeline.py

End-to-end evaluation runner for comparing baseline vs improved pipelines.

Features:
  - Run multiple prompts through two pipeline configurations
  - Collect per-prompt evaluation metrics (BLEU, CLIPScore, FID, CSR, Constraint_* scores)
  - Aggregate metrics (mean) and compute percent-delta improvements
  - Save results to disk (JSON + CSV)
"""

from __future__ import annotations
import os
import json
import csv
import time
from typing import List, Dict, Any, Callable, Optional
from statistics import mean

from utils.logging_utils import get_logger
from utils.timer import Timer

from models import Evaluator
from pipelines.multimodal_generation import MultimodalGenerationPipeline
from pipelines.image_generation import ImageGenerationPipeline
from pipelines.text_generation import TextGenerationPipeline

logger = get_logger("evaluation_pipeline")


class EvaluationPipeline:
    """
    Runs evaluation experiments comparing two systems (baseline vs improved)
    """

    def __init__(
        self,
        prompts: Optional[List[str]] = None,
        out_dir: str = "evaluation_results",
    ):
        self.prompts = prompts or []
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.evaluator = Evaluator()
        logger.info(f"EvaluationPipeline initialized (out_dir={out_dir})")

    # ------------------------------------------------------------
    # Convenience: baseline & improved runner factories
    # ------------------------------------------------------------
    def _baseline_runner(self):
        """
        Baseline: minimal RAG, no constraints, default diffusion.
        Use TextGenerationPipeline / ImageGenerationPipeline or MultimodalGenerationPipeline
        depending on your experiment.
        """
        # For multimodal baseline we use MultimodalGenerationPipeline with constraints off
        return MultimodalGenerationPipeline()

    def _improved_runner(self):
        """
        Improved: full RAG + MCP + fusion enabled.
        You can replace components here for ablation studies.
        """
        # Same pipeline but you can instantiate/configure components differently if needed
        return MultimodalGenerationPipeline()

    # ------------------------------------------------------------
    # Run single prompt on a runner (callable object with `generate`)
    # ------------------------------------------------------------
    def _run_prompt(
        self,
        runner: Any,
        prompt: str,
        constraints: Optional[Dict[str, Any]] = None,
        multimodal: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute runner.generate() and return evaluated metrics.
        """
        try:
            with Timer("single_prompt"):
                if multimodal:
                    out = runner.generate(user_query=prompt, constraints=constraints, **kwargs)
                else:
                    out = runner.generate(prompt, constraints=constraints, **kwargs)
        except Exception as e:
            logger.exception(f"Runner failed for prompt: {prompt} -> {e}")
            out = {"images": [], "captions": [], "constraints_eval": {}}

        # Evaluate using Evaluator (outputs expected: images + captions)
        eval_scores = self.evaluator.evaluate(out, reference_texts=None, constraint_scores=out.get("constraints_eval"))
        # merge stored info
        out_record = {
            "prompt": prompt,
            "outputs": out,
            "evaluation": eval_scores,
            "timestamp": time.time(),
        }
        return out_record

    # ------------------------------------------------------------
    # Run experiments across all prompts
    # ------------------------------------------------------------
    def run_experiment(
        self,
        constraints_baseline: Optional[Dict[str, Any]] = None,
        constraints_improved: Optional[Dict[str, Any]] = None,
        num_images: int = 1,
        rag_k: int = 3,
        multimodal: bool = True,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs baseline and improved systems on all prompts and returns aggregated results.
        """
        baseline_runner = self._baseline_runner()
        improved_runner = self._improved_runner()

        baseline_results = []
        improved_results = []

        logger.info("Starting evaluation experiment...")
        for idx, prompt in enumerate(self.prompts):
            logger.info(f"[{idx+1}/{len(self.prompts)}] Evaluating prompt: {prompt[:80]}")

            # Baseline (no constraints by default)
            b_constraints = constraints_baseline or {}
            b_out = self._run_prompt(
                baseline_runner,
                prompt,
                constraints=b_constraints,
                multimodal=multimodal,
                num_images=num_images,
                rag_k=rag_k,
                seed=seed,
            )
            baseline_results.append(b_out)

            # Improved (with constraints)
            i_constraints = constraints_improved or {}
            i_out = self._run_prompt(
                improved_runner,
                prompt,
                constraints=i_constraints,
                multimodal=multimodal,
                num_images=num_images,
                rag_k=rag_k,
                seed=seed,
            )
            improved_results.append(i_out)

            # Save interim results
            self._save_run_json(baseline_results, improved_results, f"interim_{idx+1}.json")

        # Aggregate metrics
        agg_baseline = self._aggregate_results(baseline_results)
        agg_improved = self._aggregate_results(improved_results)
        deltas = self._compute_deltas(agg_baseline, agg_improved)

        summary = {
            "prompts": self.prompts,
            "baseline_runs": baseline_results,
            "improved_runs": improved_results,
            "agg_baseline": agg_baseline,
            "agg_improved": agg_improved,
            "deltas_percent": deltas,
        }

        # Save final summary
        self._save_run_json(baseline_results, improved_results, "final_results.json")
        self._save_summary_csv(summary, "summary.csv")

        logger.info("Evaluation experiment complete.")
        return summary

    # ------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------
    def _aggregate_results(self, runs: List[Dict[str, Any]]) -> Dict[str, float]:
        metrics_acc: Dict[str, List[float]] = {}
        for run in runs:
            evals = run.get("evaluation", {})
            for k, v in evals.items():
                if isinstance(v, (int, float)):
                    metrics_acc.setdefault(k, []).append(v)

        agg = {k: mean(vs) if vs else 0.0 for k, vs in metrics_acc.items()}
        logger.info(f"Aggregated metrics: {agg}")
        return agg

    def _compute_deltas(self, base: Dict[str, float], improved: Dict[str, float]) -> Dict[str, float]:
        deltas: Dict[str, float] = {}
        keys = set(list(base.keys()) + list(improved.keys()))
        for k in keys:
            b = base.get(k, 0.0)
            i = improved.get(k, 0.0)
            # relative percent change (i - b) / (abs(b) + eps)
            eps = 1e-8
            try:
                delta = ((i - b) / (abs(b) + eps)) * 100.0
            except Exception:
                delta = 0.0
            deltas[k] = round(delta, 3)
        logger.info(f"Computed metric deltas: {deltas}")
        return deltas

    # ------------------------------------------------------------
    # Output saving utilities
    # ------------------------------------------------------------
    def _save_run_json(self, baseline_runs: List[Dict[str, Any]], improved_runs: List[Dict[str, Any]], fname: str):
        path = os.path.join(self.out_dir, fname)
        payload = {"baseline": baseline_runs, "improved": improved_runs}
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            logger.info(f"Saved run JSON: {path}")
        except Exception as e:
            logger.warning(f"Failed to save run JSON: {e}")

    def _save_summary_csv(self, summary: Dict[str, Any], fname: str):
        path = os.path.join(self.out_dir, fname)
        try:
            agg_b = summary.get("agg_baseline", {})
            agg_i = summary.get("agg_improved", {})
            deltas = summary.get("deltas_percent", {})

            keys = sorted(set(list(agg_b.keys()) + list(agg_i.keys()) + list(deltas.keys())))
            with open(path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                header = ["metric", "baseline_mean", "improved_mean", "delta_percent"]
                writer.writerow(header)
                for k in keys:
                    writer.writerow([k, agg_b.get(k, 0.0), agg_i.get(k, 0.0), deltas.get(k, 0.0)])
            logger.info(f"Saved summary CSV: {path}")
        except Exception as e:
            logger.warning(f"Failed to save summary CSV: {e}")


# ------------------------------------------------------------
# Quick demo run
# ------------------------------------------------------------
if __name__ == "__main__":
    prompts = [
        "A photorealistic portrait of a scientist holding a test tube in a lab.",
        "An infographic poster explaining solar energy for high school students.",
        "A cinematic scene of a small robot cooking in a modern kitchen.",
    ]
    pipeline = EvaluationPipeline(prompts=prompts, out_dir="evaluation_results_demo")
    summary = pipeline.run_experiment(
        constraints_baseline={"style": None, "factual": False, "ethical": False, "diversity": False},
        constraints_improved={"style": "photorealistic", "factual": True, "ethical": True, "diversity": True},
        num_images=1,
        rag_k=3,
        multimodal=True,
        seed=42,
    )
    print("\n=== Evaluation Summary ===")
    print(json.dumps(summary["deltas_percent"], indent=2))
