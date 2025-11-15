"""
models/constraints/__init__.py

Public API for the constraints subpackage.

This module exposes the entire Multi-Constraint Prompting (MCP)
framework components for multimodal generation control.

Modules included:
  - ConstraintManager       → Central controller that applies & evaluates all constraints
  - FactualityChecker       → Ensures knowledge-grounded correctness
  - StyleController         → Controls artistic or linguistic style
  - EthicalFilter           → Filters unsafe, NSFW, or biased content
  - DiversityController     → Promotes variation and avoids repetitive outputs

Example:
    from models.constraints import ConstraintManager

    cm = ConstraintManager()
    prompt = "Generate a photorealistic portrait of a scientist."
    modified = cm.apply(
        constraints={
            "factual": True,
            "style": "photorealistic",
            "ethical": True,
            "diversity": True
        },
        prompt=prompt
    )
    print(modified)
"""

from .constraint_manager import ConstraintManager  # type: ignore
from .factuality_checker import FactualityChecker  # type: ignore
from .style_controller import StyleController  # type: ignore
from .ethical_filter import EthicalFilter  # type: ignore
from .diversity_controller import DiversityController  # type: ignore

__all__ = [
    "ConstraintManager",
    "FactualityChecker",
    "StyleController",
    "EthicalFilter",
    "DiversityController",
]