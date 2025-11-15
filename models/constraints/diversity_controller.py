"""
models/constraints/diversity_controller.py

Promotes diversity across multimodal generations.

Features:
  - Adds diversity-promoting directives in prompts
  - Evaluates text/image output diversity using embedding variance
  - Helps prevent repetitive or overly similar generations
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from utils.logging_utils import get_logger
from utils.timer import Timer
from models.retrieval.embedder import Embedder

logger = get_logger("diversity_controller")


class DiversityController:
    """
    Controls and measures diversity of generated content.
    """

    def __init__(self):
        self.embedder = Embedder(use_clip=True)
        logger.info("DiversityController initialized (using CLIP for embedding diversity).")

    # ------------------------------------------------------------
    # 🧩 Prompt Augmentation (Pre-generation)
    # ------------------------------------------------------------
    def promote_diversity(self, prompt: str) -> str:
        """
        Adds creative variation directive to the prompt.

        Example:
            Input  : "Generate an image of a city skyline."
            Output : "Generate an image of a city skyline with unique composition and creative diversity."
        """
        directive = (
            "Encourage diverse, non-repetitive, and creatively varied elements in the generation. "
            "Avoid duplication of visual or linguistic features."
        )
        if "diverse" not in prompt.lower():
            prompt = f"{prompt}\n\n[DIVERSITY NOTE]: {directive}"
        logger.info("Diversity directive added to prompt.")
        return prompt

    # ------------------------------------------------------------
    # 🎯 Evaluate Diversity (Post-generation)
    # ------------------------------------------------------------
    def evaluate_diversity(
        self,
        images: List[Any],
        captions: List[str],
    ) -> float:
        """
        Quantifies diversity using embedding variance.

        Steps:
          - Compute CLIP embeddings for images and captions
          - Compute average pairwise cosine similarity
          - Diversity score = 1 - avg_similarity
        Returns a score between 0 (identical) and 1 (highly diverse)
        """
        with Timer("diversity.evaluate"):
            try:
                image_score = self._evaluate_image_diversity(images)
                text_score = self._evaluate_text_diversity(captions)
                final_score = (image_score + text_score) / 2
                logger.info(f"Diversity score: {final_score:.3f}")
                return np.clip(final_score, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Diversity evaluation failed: {e}")
                return 0.5

    # ------------------------------------------------------------
    # 🖼️ Image Diversity
    # ------------------------------------------------------------
    def _evaluate_image_diversity(self, images: List[Any]) -> float:
        if not images:
            return 1.0

        try:
            embeds = self.embedder.encode_images(images)
            sims = self._pairwise_cosine_sim(embeds)
            avg_sim = np.mean(sims)
            return float(1.0 - avg_sim)
        except Exception as e:
            logger.debug(f"Image diversity skipped: {e}")
            return 1.0

    # ------------------------------------------------------------
    # 🔤 Text Diversity
    # ------------------------------------------------------------
    def _evaluate_text_diversity(self, captions: List[str]) -> float:
        if not captions or len(captions) < 2:
            return 1.0
        try:
            embeds = self.embedder.encode_texts(captions)
            sims = self._pairwise_cosine_sim(embeds)
            avg_sim = np.mean(sims)
            return float(1.0 - avg_sim)
        except Exception as e:
            logger.debug(f"Text diversity skipped: {e}")
            return 1.0

    # ------------------------------------------------------------
    # 🧮 Pairwise Cosine Similarity
    # ------------------------------------------------------------
    def _pairwise_cosine_sim(self, X: np.ndarray) -> np.ndarray:
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(Xn, Xn.T)
        n = len(Xn)
        upper = sim_matrix[np.triu_indices(n, k=1)]
        return upper


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    dc = DiversityController()
    prompt = "Generate futuristic cars in various styles."
    mod_prompt = dc.promote_diversity(prompt)
    print("\nModified Prompt:\n", mod_prompt)

    captions = [
        "A futuristic car with glowing wheels.",
        "A flying car in neon colors.",
        "A futuristic car with a streamlined metallic body."
    ]
    score = dc.evaluate_diversity([], captions)
    print(f"\nDiversity Score: {score:.3f}")