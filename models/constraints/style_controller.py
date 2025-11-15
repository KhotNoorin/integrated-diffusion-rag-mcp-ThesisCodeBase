"""
models/constraints/style_controller.py

Controls stylistic aspects of multimodal content generation.

Features:
  - Injects stylistic cues into prompts (e.g., "in cinematic style")
  - Evaluates stylistic alignment post-generation (semantic match)
  - Adaptable for both text and image generation
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from utils.logging_utils import get_logger
from utils.timer import Timer
from models.retrieval.embedder import Embedder

logger = get_logger("style_controller")


class StyleController:
    """
    Applies and evaluates style consistency in multimodal generation.
    """

    def __init__(self):
        self.embedder = Embedder(use_clip=True)
        logger.info("StyleController initialized (using CLIP for style evaluation).")

    # ------------------------------------------------------------
    # 🧩 Apply Style Pre-Generation
    # ------------------------------------------------------------
    def apply_style(self, prompt: str, style: str) -> str:
        """
        Appends stylistic description to prompt.

        Example:
            Input  : "A futuristic city skyline."
            Style  : "cyberpunk"
            Output : "A futuristic city skyline, in a detailed cyberpunk style."
        """
        if style.lower() in prompt.lower():
            return prompt

        stylistic_phrases = [
            f"in a {style} style",
            f"depicted in a {style} manner",
            f"with a {style} aesthetic",
        ]
        augmented = f"{prompt}, {np.random.choice(stylistic_phrases)}."
        logger.info(f"Applied style '{style}' to prompt.")
        return augmented

    # ------------------------------------------------------------
    # 🎯 Evaluate Style Consistency Post-Generation
    # ------------------------------------------------------------
    def evaluate_style(self, prompt: str, captions: List[str]) -> float:
        """
        Evaluates whether the generated captions align with the intended style.
        Uses CLIP-based semantic similarity between style word and caption text.
        """
        if not captions:
            return 0.0

        with Timer("style.evaluate"):
            try:
                style_terms = self._extract_style_terms(prompt)
                if not style_terms:
                    return 1.0  # No explicit style constraint, so skip

                scores = []
                for style in style_terms:
                    style_emb = self.embedder.encode_texts([style])[0]
                    caps_emb = self.embedder.encode_texts(captions)
                    sims = np.dot(caps_emb, style_emb) / (
                        np.linalg.norm(caps_emb, axis=1) * np.linalg.norm(style_emb) + 1e-8
                    )
                    scores.append(np.mean(sims))
                final_score = float(np.mean(scores))
                logger.info(f"Style consistency score: {final_score:.3f}")
                return np.clip(final_score, 0.0, 1.0)
            except Exception as e:
                logger.warning(f"Style evaluation failed: {e}")
                return 0.0

    # ------------------------------------------------------------
    # 🧠 Helper: Extract style words
    # ------------------------------------------------------------
    def _extract_style_terms(self, text: str) -> List[str]:
        """
        Extracts stylistic keywords from a prompt.
        """
        keywords = [
            "cinematic", "photorealistic", "watercolor", "anime", "digital art",
            "scientific", "illustration", "abstract", "minimalist", "3d render",
            "cartoon", "flat design", "oil painting", "vintage", "cyberpunk"
        ]
        found = [kw for kw in keywords if kw in text.lower()]
        return found


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    controller = StyleController()
    prompt = "Generate a portrait of a medieval knight."
    styled = controller.apply_style(prompt, "cinematic")
    print("\nStyled Prompt:\n", styled)

    captions = ["A cinematic portrait of a knight wearing armor."]
    score = controller.evaluate_style(styled, captions)
    print(f"\nStyle Score: {score:.3f}")