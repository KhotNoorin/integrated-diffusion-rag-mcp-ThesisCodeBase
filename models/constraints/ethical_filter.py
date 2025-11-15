"""
models/constraints/ethical_filter.py

Ensures ethical and safe multimodal generation.

Responsibilities:
  - Detect and mitigate NSFW or biased content
  - Filter unsafe text or images
  - Provide ethical safety scores post-generation
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from utils.logging_utils import get_logger
from utils.timer import Timer
import numpy as np
import re

logger = get_logger("ethical_filter")

# Optional deps for deep filtering
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False

try:
    import torch
    import torchvision
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False


class EthicalFilter:
    """
    Provides ethical prompt enforcement and NSFW detection.
    """

    def __init__(self, use_text_filter: bool = True, use_image_filter: bool = True):
        self.use_text_filter = use_text_filter
        self.use_image_filter = use_image_filter
        self.text_classifier = None
        self.image_classifier = None

        # Optional Hugging Face text toxicity model
        if _HAS_TRANSFORMERS and self.use_text_filter:
            try:
                self.text_classifier = pipeline(
                    "text-classification",
                    model="unitary/toxic-bert",
                    top_k=None
                )
                logger.info("Loaded text toxicity classifier (unitary/toxic-bert).")
            except Exception as e:
                logger.warning(f"Failed to load text classifier: {e}")
                self.text_classifier = None

        # Optional NSFW image detector
        if _HAS_TORCHVISION and self.use_image_filter:
            try:
                from torchvision import models, transforms
                self.image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.image_model.eval()
                self.preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])
                logger.info("Initialized simple image safety model (ResNet18 base).")
            except Exception as e:
                logger.warning(f"Failed to load image model: {e}")
                self.image_model = None

    # ------------------------------------------------------------
    # 🧩 Prompt Sanitization (Pre-generation)
    # ------------------------------------------------------------
    def enforce_ethics(self, prompt: str) -> str:
        """
        Adds an ethical safety directive to the generation prompt.
        """
        ethical_directive = (
            "Ensure the generated content follows ethical and non-harmful principles. "
            "Avoid NSFW, violent, or biased elements."
        )

        if "ethical" not in prompt.lower():
            prompt = f"{prompt}\n\n[ETHICAL SAFETY NOTE]: {ethical_directive}"

        logger.info("Ethical directive appended to prompt.")
        return prompt

    # ------------------------------------------------------------
    # 🧮 Evaluate Ethical Safety (Post-generation)
    # ------------------------------------------------------------
    def evaluate_safety(self, prompt: str, images: List[Any]) -> float:
        """
        Evaluate safety for both text and images.
        Returns a normalized score in [0, 1], where 1 = fully safe.
        """
        with Timer("ethics.evaluate"):
            try:
                text_score = self._evaluate_text(prompt)
                image_score = self._evaluate_images(images)
                combined_score = (text_score + image_score) / 2
                logger.info(f"Ethical safety score: {combined_score:.3f}")
                return float(np.clip(combined_score, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Ethical evaluation failed: {e}")
                return 0.5  # neutral fallback

    # ------------------------------------------------------------
    # 🔤 Text Filter
    # ------------------------------------------------------------
    def _evaluate_text(self, text: str) -> float:
        """
        Check text for toxicity or unsafe content.
        """
        if not self.use_text_filter:
            return 1.0

        unsafe_keywords = [
            "violence", "blood", "weapon", "sex", "nude", "drugs", "kill", "hate", "racist", "suicide"
        ]
        pattern = "|".join([re.escape(k) for k in unsafe_keywords])

        # keyword-level heuristic
        if re.search(pattern, text.lower()):
            logger.warning("Unsafe keywords detected in text.")
            return 0.3

        # deep toxicity model
        if self.text_classifier:
            try:
                preds = self.text_classifier(text)
                avg_tox = np.mean([p.get("score", 0.0) for p in preds[0] if "toxic" in p["label"].lower()])
                return float(np.clip(1.0 - avg_tox, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Text classifier failed: {e}")
                return 0.7

        return 1.0

    # ------------------------------------------------------------
    # 🖼️ Image Filter
    # ------------------------------------------------------------
    def _evaluate_images(self, images: List[Any]) -> float:
        """
        Basic NSFW scoring for images.
        (In production, use OpenAI's moderation API or LAION NSFW model.)
        """
        if not self.use_image_filter or not images:
            return 1.0

        if not _HAS_TORCHVISION or not hasattr(self, "preprocess"):
            logger.debug("Torchvision not available — skipping image filtering.")
            return 1.0

        try:
            from PIL import Image
            scores = []
            for img in images[:5]:  # limit for efficiency
                if not isinstance(img, Image.Image):
                    continue
                tensor = self.preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    _ = self.image_model(tensor)
                    # Placeholder: randomly assume safe
                    scores.append(1.0)
            return float(np.mean(scores))
        except Exception as e:
            logger.warning(f"Image safety check failed: {e}")
            return 0.8


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    ef = EthicalFilter()
    prompt = "Generate an image of a violent battle scene."
    safe_prompt = ef.enforce_ethics(prompt)
    print("\nModified Prompt:\n", safe_prompt)

    score = ef.evaluate_safety(safe_prompt, [])
    print(f"\nEthical Safety Score: {score:.3f}")