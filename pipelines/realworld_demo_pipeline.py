"""
pipelines/realworld_demo_pipeline.py

Real-world demonstration pipeline for thesis showcase.

Integrates:
  - Retrieval-Augmented Generation (RAG)
  - Diffusion-based image generation
  - Multi-Constraint Prompting (MCP)
  - Evaluation metrics
  - Streamlit/Gradio-ready hooks

Use this pipeline in your thesis demo or Streamlit UI to show
how factual retrieval + ethical + stylistic constraints improve
real-world multimodal content generation.

Example use cases:
  - Educational illustrations
  - News-style factual visualization
  - Creative artwork with ethical filtering
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os
from datetime import datetime
from PIL import Image

from utils.logging_utils import get_logger
from models import (
    MultimodalGenerator,
    Evaluator,
)

logger = get_logger("realworld_demo_pipeline")


class RealWorldDemoPipeline:
    """
    High-level demonstration interface combining all components.
    """

    def __init__(self, output_dir: str = "demo_outputs"):
        self.generator = MultimodalGenerator()
        self.evaluator = Evaluator()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info("🚀 RealWorldDemoPipeline initialized successfully.")

    # ------------------------------------------------------------
    # 🌍 Run a real-world example
    # ------------------------------------------------------------
    def run_demo(
        self,
        scenario: str,
        constraints: Optional[Dict[str, Any]] = None,
        style_hint: Optional[str] = None,
        num_images: int = 1,
        height: int = 512,
        width: int = 512,
    ) -> Dict[str, Any]:
        """
        Run a real-world multimodal generation scenario.
        """
        constraints = constraints or {
            "style": style_hint or "realistic",
            "factual": True,
            "ethical": True,
            "diversity": True,
        }

        logger.info(f"Running real-world demo scenario: {scenario}")
        results = self.generator.generate(
            user_query=scenario,
            constraints=constraints,
            num_images=num_images,
            height=height,
            width=width,
            style_hint=style_hint,
        )

        # Evaluate
        eval_metrics = self.evaluator.evaluate(
            {"images": results["images"], "captions": results["captions"]},
            constraint_scores=results.get("constraints_eval"),
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        demo_dir = os.path.join(self.output_dir, f"demo_{timestamp}")
        os.makedirs(demo_dir, exist_ok=True)

        for idx, img in enumerate(results["images"]):
            if isinstance(img, Image.Image):
                img_path = os.path.join(demo_dir, f"demo_image_{idx}.png")
                img.save(img_path)
                logger.info(f"Saved demo image: {img_path}")

        # Save metadata
        meta_path = os.path.join(demo_dir, "demo_metadata.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"Scenario: {scenario}\n")
            f.write(f"Prompt: {results['prompt']}\n")
            f.write(f"Constraints: {constraints}\n")
            f.write(f"Evaluation: {eval_metrics}\n")
        logger.info(f"Saved demo metadata: {meta_path}")

        return {
            "scenario": scenario,
            "prompt": results["prompt"],
            "constraints": constraints,
            "evaluation": eval_metrics,
            "output_dir": demo_dir,
        }

    # ------------------------------------------------------------
    # 🧠 Example scenarios
    # ------------------------------------------------------------
    def predefined_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns a set of ready-to-run demo scenarios.
        """
        return {
            "education": {
                "description": "Explain climate change through visual storytelling.",
                "style": "scientific",
                "constraints": {"factual": True, "ethical": True, "diversity": True},
            },
            "art": {
                "description": "A surrealist painting of a city floating above the ocean.",
                "style": "abstract",
                "constraints": {"style": "abstract", "ethical": True, "diversity": True},
            },
            "news": {
                "description": "An infographic showing renewable energy adoption across continents.",
                "style": "infographic",
                "constraints": {"factual": True, "ethical": True},
            },
            "cinematic_ai": {
                "description": "A cinematic scene of a humanoid AI learning to paint.",
                "style": "cinematic",
                "constraints": {"style": "cinematic", "factual": True, "ethical": True, "diversity": True},
            },
        }


# ------------------------------------------------------------
# ✅ Example Test Run
# ------------------------------------------------------------
if __name__ == "__main__":
    demo = RealWorldDemoPipeline()
    scenarios = demo.predefined_scenarios()

    # Example 1: Educational scenario
    result = demo.run_demo(
        scenario=scenarios["education"]["description"],
        constraints=scenarios["education"]["constraints"],
        style_hint=scenarios["education"]["style"],
    )

    print("\n=== Real-World Demo Summary ===")
    print(f"Scenario: {result['scenario']}")
    print(f"Prompt: {result['prompt'][:300]}")
    print(f"Constraints: {result['constraints']}")
    print(f"Evaluation Metrics: {result['evaluation']}")
    print(f"Output Directory: {result['output_dir']}")