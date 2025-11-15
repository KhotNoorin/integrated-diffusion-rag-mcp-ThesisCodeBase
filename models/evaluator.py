"""
models/evaluator.py

Evaluation utilities for multimodal content generation.

Supports:
  - Text quality metrics (BLEU, ROUGE)
  - Image quality metrics (FID, IS)
  - Cross-modal alignment metrics (CLIPScore, CSR)
  - Constraint satisfaction metrics (factuality, ethics, diversity)

Used in:
  - EvaluationPipeline
  - MultimodalGenerator post-hoc validation
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any
from utils.logging_utils import get_logger
from utils.metrics import (
    compute_bleu,
    compute_clipscore,
    compute_fid,
    compute_csr,
)

logger = get_logger("evaluator")


class Evaluator:
    """
    Unified evaluation class for multimodal generation outputs.
    """

    def __init__(self, enable_fid: bool = True, enable_clip: bool = True):
        self.enable_fid = enable_fid
        self.enable_clip = enable_clip
        logger.info(f"Evaluator initialized (FID={enable_fid}, CLIPScore={enable_clip})")

    # ------------------------------------------------------------
    # 🔍 Main Evaluation Function
    # ------------------------------------------------------------
    def evaluate(
        self,
        outputs: Dict[str, Any],
        reference_texts: Optional[List[str]] = None,
        constraint_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Compute all available evaluation metrics.

        Args:
            outputs: dict with "images" and "captions"
            reference_texts: list of reference captions
            constraint_scores: dict from ConstraintManager.evaluate()

        Returns:
            dict of aggregated metrics
        """
        results = {}
        captions = outputs.get("captions", [])
        images = outputs.get("images", [])

        # --------------------------------------------------------
        # Text Quality
        # --------------------------------------------------------
        if reference_texts:
            bleu_score = compute_bleu(captions, reference_texts)
            results["BLEU"] = bleu_score
            logger.info(f"BLEU Score: {bleu_score:.3f}")

        # --------------------------------------------------------
        # Cross-modal Alignment
        # --------------------------------------------------------
        if self.enable_clip:
            try:
                clip_score = compute_clipscore(images, captions)
                results["CLIPScore"] = clip_score
                logger.info(f"CLIPScore: {clip_score:.3f}")
            except Exception as e:
                logger.warning(f"CLIPScore failed: {e}")

        # --------------------------------------------------------
        # Image Quality
        # --------------------------------------------------------
        if self.enable_fid:
            try:
                fid_score = compute_fid(images)
                results["FID"] = fid_score
                logger.info(f"FID Score: {fid_score:.3f}")
            except Exception as e:
                logger.warning(f"FID computation failed: {e}")

        # --------------------------------------------------------
        # Semantic Relatedness
        # --------------------------------------------------------
        try:
            csr_score = compute_csr(captions)
            results["CSR"] = csr_score
            logger.info(f"Caption Semantic Relatedness (CSR): {csr_score:.3f}")
        except Exception as e:
            logger.warning(f"CSR metric failed: {e}")

        # --------------------------------------------------------
        # Constraint Satisfaction
        # --------------------------------------------------------
        if constraint_scores:
            results.update({
                f"Constraint_{k.capitalize()}": v for k, v in constraint_scores.items()
            })

        # --------------------------------------------------------
        # Aggregated Quality Score
        # --------------------------------------------------------
        try:
            valid_scores = [v for v in results.values() if isinstance(v, (int, float))]
            results["OverallScore"] = np.mean(valid_scores) if valid_scores else 0.0
        except Exception as e:
            logger.warning(f"Failed to compute OverallScore: {e}")
            results["OverallScore"] = 0.0

        logger.info(f"Final evaluation results: {results}")
        return results

    # ------------------------------------------------------------
    # 📊 Comparative Evaluation
    # ------------------------------------------------------------
    def compare_runs(
        self,
        baseline_results: Dict[str, float],
        improved_results: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compare evaluation metrics between baseline and improved models.

        Returns delta improvements (%).
        """
        deltas = {}
        for key in improved_results:
            if key in baseline_results:
                try:
                    base = baseline_results[key]
                    new = improved_results[key]
                    delta = ((new - base) / (base + 1e-8)) * 100.0
                    deltas[key] = round(delta, 2)
                except Exception:
                    deltas[key] = 0.0
        logger.info(f"Evaluation deltas: {deltas}")
        return deltas


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    dummy_output = {
        "images": ["img1.png", "img2.png"],  # placeholders
        "captions": ["A futuristic cityscape.", "A robot serving food."]
    }
    reference = ["A city in the future.", "A humanoid robot serving dinner."]

    evaluator = Evaluator()
    scores = evaluator.evaluate(dummy_output, reference_texts=reference, constraint_scores={
        "factual": 0.92, "ethical": 0.98, "style": 0.87, "diversity": 0.91
    })

    print("\n=== Evaluation Results ===")
    for k, v in scores.items():
        print(f"{k:20s}: {v:.3f}")