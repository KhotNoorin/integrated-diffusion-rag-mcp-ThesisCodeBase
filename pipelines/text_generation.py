"""
pipelines/text_generation.py

Text-only generation pipeline:
  - Uses Retrieval Augmented Generation (RAG)
  - Applies Multi-Constraint Prompting (MCP)
  - Generates factual, ethical, stylistically controlled text outputs

This pipeline can be used for:
  - Text-to-text content creation (summaries, descriptions, answers)
  - Comparison baseline for multimodal generation
"""

from __future__ import annotations
from typing import Dict, Any
from utils.logging_utils import get_logger
from utils.timer import Timer

from models import (
    Retriever,
    Reranker,
    ConstraintManager,
    PromptConstructor,
)

logger = get_logger("text_generation_pipeline")


class TextGenerationPipeline:
    """
    Text-only RAG + MCP pipeline.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.constraints = ConstraintManager()
        self.prompt_builder = PromptConstructor()

        logger.info("TextGenerationPipeline initialized.")

    # ------------------------------------------------------------
    # 🧠 Generate Constrained Text
    # ------------------------------------------------------------
    def generate(
        self,
        user_query: str,
        constraints: Dict[str, Any] = None,
        rag_k: int = 3,
        use_reranker: bool = True,
    ) -> Dict[str, Any]:
        """
        Generates a factual, stylistically controlled text output.
        """
        constraints = constraints or {}

        with Timer("text_generation"):
            # 1️⃣ Retrieve knowledge context
            retrieved_docs = self.retriever.retrieve(user_query, k=rag_k)
            if use_reranker:
                reranked = self.reranker.rerank(user_query, retrieved_docs)
                retrieved_docs = [doc for doc, _ in reranked]

            # 2️⃣ Build constrained prompt
            prompt = self.prompt_builder.build(user_query, retrieved_docs, constraints)

            # 3️⃣ Simulated generation (replace with real LLM or API call)
            # For demonstration: echo with modifications
            generated_text = f"[Generated Text] {prompt}"

            # 4️⃣ Evaluate constraint satisfaction
            constraint_scores = self.constraints.evaluate({
                "prompt": prompt,
                "images": [],
                "captions": [generated_text],
                "constraints": constraints,
            })

            return {
                "query": user_query,
                "prompt": prompt,
                "retrieved_docs": retrieved_docs,
                "generated_text": generated_text,
                "constraints_eval": constraint_scores,
            }


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    pipeline = TextGenerationPipeline()
    result = pipeline.generate(
        "Explain how diffusion models generate images.",
        constraints={
            "style": "scientific",
            "factual": True,
            "ethical": True,
            "diversity": False,
        },
    )

    print("\n=== Text Generation Output ===")
    print("Prompt:", result["prompt"])
    print("Generated:", result["generated_text"])
    print("Constraint Scores:", result["constraints_eval"])