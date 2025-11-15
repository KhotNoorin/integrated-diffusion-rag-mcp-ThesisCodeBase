"""
models/constraints/constraint_manager.py

Main controller for Multi-Constraint Prompting (MCP).

Responsibilities:
  - Manage and apply multiple constraints (factuality, style, ethics, diversity)
  - Coordinate pre-generation (prompt modification)
  - Coordinate post-generation (evaluation of outputs)
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger
from utils.timer import Timer

# Submodules (lazy imports for performance)
try:
    from models.constraints.factuality_checker import FactualityChecker
    from models.constraints.style_controller import StyleController
    from models.constraints.ethical_filter import EthicalFilter
    from models.constraints.diversity_controller import DiversityController
except Exception:
    FactualityChecker = None
    StyleController = None
    EthicalFilter = None
    DiversityController = None

logger = get_logger("constraint_manager")


class ConstraintManager:
    """
    Unified interface for handling multiple constraints.
    """

    def __init__(self):
        self.factuality_checker = FactualityChecker() if FactualityChecker else None
        self.style_controller = StyleController() if StyleController else None
        self.ethical_filter = EthicalFilter() if EthicalFilter else None
        self.diversity_controller = DiversityController() if DiversityController else None

        logger.info("ConstraintManager initialized with available modules:")
        for name, module in {
            "FactualityChecker": self.factuality_checker,
            "StyleController": self.style_controller,
            "EthicalFilter": self.ethical_filter,
            "DiversityController": self.diversity_controller,
        }.items():
            logger.info(f"  - {name}: {'✅' if module else '❌'}")

    # ------------------------------------------------------------
    # 🧩 Apply Constraints Before Generation
    # ------------------------------------------------------------
    def apply(self, constraints: Dict[str, Any], prompt: str) -> str:
        """
        Apply all relevant constraint transformations to the prompt.

        Args:
            constraints: user-defined or system constraints
            prompt: raw text prompt
        Returns:
            modified prompt string
        """
        logger.info("Applying constraints to prompt...")

        updated_prompt = prompt

        if constraints.get("style") and self.style_controller:
            with Timer("style.apply"):
                updated_prompt = self.style_controller.apply_style(updated_prompt, constraints["style"])

        if constraints.get("ethical") and self.ethical_filter:
            with Timer("ethics.apply"):
                updated_prompt = self.ethical_filter.enforce_ethics(updated_prompt)

        if constraints.get("diversity") and self.diversity_controller:
            with Timer("diversity.apply"):
                updated_prompt = self.diversity_controller.promote_diversity(updated_prompt)

        if constraints.get("factual") and self.factuality_checker:
            with Timer("factuality.apply"):
                updated_prompt = self.factuality_checker.ensure_factuality(updated_prompt)

        logger.info("All applicable constraints applied successfully.")
        return updated_prompt

    # ------------------------------------------------------------
    # 🧮 Evaluate Constraints After Generation
    # ------------------------------------------------------------
    def evaluate(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated outputs for constraint satisfaction.

        Args:
            output: dict with "prompt", "images", "captions", and "constraints"
        Returns:
            dict of constraint satisfaction scores
        """
        results = {"factual": 0.0, "style": 0.0, "ethical": 0.0, "diversity": 0.0}

        try:
            prompt = output.get("prompt", "")
            captions = output.get("captions", [])
            images = output.get("images", [])

            if self.factuality_checker:
                results["factual"] = self.factuality_checker.evaluate(prompt, captions)
            if self.style_controller:
                results["style"] = self.style_controller.evaluate_style(prompt, captions)
            if self.ethical_filter:
                results["ethical"] = self.ethical_filter.evaluate_safety(prompt, images)
            if self.diversity_controller:
                results["diversity"] = self.diversity_controller.evaluate_diversity(images, captions)

            avg_score = sum(results.values()) / len(results)
            logger.info(f"Constraint satisfaction average: {avg_score:.3f}")
        except Exception as e:
            logger.warning(f"Constraint evaluation failed: {e}")

        return results

    # ------------------------------------------------------------
    # 🧠 Summary Helper
    # ------------------------------------------------------------
    def summary(self) -> Dict[str, bool]:
        return {
            "factuality": bool(self.factuality_checker),
            "style": bool(self.style_controller),
            "ethical": bool(self.ethical_filter),
            "diversity": bool(self.diversity_controller),
        }


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    cm = ConstraintManager()
    prompt = "Generate an image of a person standing on Mars."
    constraints = {
        "factual": True,
        "style": "cinematic",
        "ethical": True,
        "diversity": True,
    }

    updated = cm.apply(constraints, prompt)
    print("\nModified Prompt:\n", updated)

    fake_output = {
        "prompt": updated,
        "images": [],
        "captions": ["A cinematic astronaut standing on Mars."],
        "constraints": constraints,
    }
    scores = cm.evaluate(fake_output)
    print("\nConstraint Scores:", scores)