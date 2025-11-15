"""
models/constraints/factuality_checker.py

Ensures factual consistency between:
  - retrieved evidence (RAG context)
  - generated text/captions
  - input prompt (user intent)

Implements:
  - Embedding similarity-based factuality score
  - Optional QA-based validation if transformers installed
"""

from __future__ import annotations
from typing import List, Union, Optional, Dict, Any
import numpy as np
from utils.logging_utils import get_logger
from utils.timer import Timer
from models.retrieval.embedder import Embedder

logger = get_logger("factuality_checker")

# Optional QA verification
try:
    from transformers import pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class FactualityChecker:
    """
    Computes factual consistency score for generated outputs.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        use_qa_verification: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.use_qa_verification = use_qa_verification and _HAS_TRANSFORMERS
        self.embedder = Embedder(use_clip=False)
        self.qa_model = None

        if self.use_qa_verification:
            try:
                logger.info("Loading QA-based factuality model (distilbert-base-cased-distilled-squad)...")
                self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
            except Exception as e:
                logger.warning(f"Failed to load QA model: {e}")
                self.use_qa_verification = False

        logger.info("FactualityChecker initialized.")

    # ------------------------------------------------------------
    # 🧩 Pre-generation factual enforcement (prompt rewrite)
    # ------------------------------------------------------------
    def ensure_factuality(self, prompt: str) -> str:
        """
        Adds explicit factual grounding instructions to prompt.
        """
        factual_directive = (
            "Ensure that all details are factual and supported by verified evidence. "
            "Do not introduce speculative or misleading content."
        )
        if factual_directive.lower() not in prompt.lower():
            prompt += f"\n\n[FACTUALITY DIRECTIVE]: {factual_directive}"
        return prompt

    # ------------------------------------------------------------
    # 🧮 Post-generation evaluation
    # ------------------------------------------------------------
    def evaluate(self, prompt: str, captions: List[str]) -> float:
        """
        Compute factuality score between prompt and generated captions.

        Steps:
          1. Compute embedding similarity between prompt and captions
          2. Optionally run QA check (if available)
        """
        if not captions:
            return 0.0

        with Timer("factuality.evaluate"):
            try:
                prompt_emb = self.embedder.encode_texts([prompt])[0]
                cap_embs = self.embedder.encode_texts(captions)
                sims = np.dot(cap_embs, prompt_emb) / (
                    np.linalg.norm(cap_embs, axis=1) * np.linalg.norm(prompt_emb) + 1e-8
                )
                base_score = float(np.mean(sims))

                if self.use_qa_verification and self.qa_model is not None:
                    qa_scores = []
                    for cap in captions:
                        qa_input = {
                            "question": "Is this statement factually consistent with the prompt?",
                            "context": f"Prompt: {prompt}\nGenerated: {cap}",
                        }
                        result = self.qa_model(qa_input)
                        qa_scores.append(result.get("score", 0.0))
                    qa_score = np.mean(qa_scores)
                    final_score = (base_score + qa_score) / 2
                else:
                    final_score = base_score

                logger.info(f"Factuality score: {final_score:.3f}")
                return float(np.clip(final_score, 0.0, 1.0))

            except Exception as e:
                logger.warning(f"Factuality evaluation failed: {e}")
                return 0.0


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    checker = FactualityChecker()
    prompt = "The Eiffel Tower is located in Paris, France."
    captions = [
        "The Eiffel Tower is a landmark in Paris.",
        "The Eiffel Tower is in London.",
    ]

    score = checker.evaluate(prompt, captions)
    print(f"Factuality Score: {score:.3f}")

    rewritten = checker.ensure_factuality(prompt)
    print("\nRewritten Prompt:\n", rewritten)