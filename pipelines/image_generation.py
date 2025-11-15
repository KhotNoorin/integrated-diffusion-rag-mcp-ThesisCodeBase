"""
pipelines/image_generation.py

Image generation pipeline using:
  - Retrieval Augmented Generation (RAG)
  - Multi-Constraint Prompting (MCP)
  - Diffusion model for visual synthesis

This pipeline demonstrates how factual retrieval and ethical,
style-aware constraints improve diffusion-based generation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from utils.logging_utils import get_logger
from utils.timer import Timer

from models import (
    Retriever,
    Reranker,
    ConstraintManager,
    PromptConstructor,
    BaseDiffusion,
    Evaluator,
)

logger = get_logger("image_generation_pipeline")


class ImageGenerationPipeline:
    """
    Generates images using RAG + Diffusion + MCP.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.constraints = ConstraintManager()
        self.prompt_builder = PromptConstructor()
        self.diffusion = BaseDiffusion()
        self.evaluator = Evaluator(enable_clip=True, enable_fid=True)

        logger.info("ImageGenerationPipeline initialized successfully.")

    # ------------------------------------------------------------
    # 🧠 Main image generation method
    # ------------------------------------------------------------
    def generate(
        self,
        user_query: str,
        constraints: Optional[Dict[str, Any]] = None,
        num_images: int = 1,
        rag_k: int = 3,
        use_reranker: bool = True,
        style_hint: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate images using the integrated pipeline.
        """
        constraints = constraints or {}

        logger.info(f"Starting image generation for: '{user_query}'")
        with Timer("image_generation_pipeline"):
            # --------------------------------------------------------
            # 1️⃣ Retrieve contextual evidence
            # --------------------------------------------------------
            retrieved_docs = self.retriever.retrieve(user_query, k=rag_k)
            if use_reranker:
                reranked = self.reranker.rerank(user_query, retrieved_docs)
                retrieved_docs = [doc for doc, _ in reranked]
            logger.info(f"Retrieved {len(retrieved_docs)} documents for context.")

            # --------------------------------------------------------
            # 2️⃣ Build a constraint-aware prompt
            # --------------------------------------------------------
            prompt = self.prompt_builder.build(user_query, retrieved_docs, constraints)
            if style_hint:
                prompt = self.prompt_builder.apply_style(prompt, style_hint)

            logger.debug(f"Final prompt:\n{prompt}")

            # --------------------------------------------------------
            # 3️⃣ Generate image via diffusion
            # --------------------------------------------------------
            diffusion_results = self.diffusion.generate(
                prompt=prompt,
                num_images=num_images,
                style_hint=style_hint,
                height=height,
                width=width,
                seed=seed,
            )

            images = diffusion_results if isinstance(diffusion_results, list) else diffusion_results.get("images", [])

            # --------------------------------------------------------
            # 4️⃣ Evaluate generation quality
            # --------------------------------------------------------
            constraint_scores = self.constraints.evaluate({
                "prompt": prompt,
                "images": images,
                "captions": [prompt] * len(images),
                "constraints": constraints,
            })

            eval_scores = self.evaluator.evaluate(
                {"images": images, "captions": [prompt] * len(images)},
                constraint_scores=constraint_scores,
            )

            logger.info(f"Evaluation complete for '{user_query}'")

            return {
                "query": user_query,
                "prompt": prompt,
                "retrieved_docs": retrieved_docs,
                "images": images,
                "constraints_eval": constraint_scores,
                "evaluation_metrics": eval_scores,
            }


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    pipeline = ImageGenerationPipeline()

    result = pipeline.generate(
        user_query="A cinematic portrait of an astronaut walking on Mars",
        constraints={
            "style": "cinematic",
            "factual": True,
            "ethical": True,
            "diversity": True,
        },
        num_images=1,
        style_hint="cinematic",
    )

    print("\n=== Image Generation Output ===")
    print("Prompt:", result["prompt"][:300])
    print("Constraints:", result["constraints_eval"])
    print("Evaluation Metrics:", result["evaluation_metrics"])