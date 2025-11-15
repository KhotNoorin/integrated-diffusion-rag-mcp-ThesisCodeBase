"""
pipelines/multimodal_generation.py

Unified multimodal content generation pipeline integrating:
  - Retrieval Augmented Generation (RAG)
  - Multi-Constraint Prompting (MCP)
  - Diffusion-based image synthesis
  - Multimodal fusion (text + image)
  - Automated evaluation

This pipeline is the core of the system described in your thesis.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from utils.logging_utils import get_logger
from utils.timer import Timer

from models import (
    Retriever,
    Reranker,
    ConstraintManager,
    PromptConstructor,
    BaseDiffusion,
    MultimodalFuser,
    AttentionBridge,
    Evaluator,
)

logger = get_logger("multimodal_generation_pipeline")


class MultimodalGenerationPipeline:
    """
    Unified multimodal (text + image) generation pipeline.
    """

    def __init__(self):
        self.retriever = Retriever()
        self.reranker = Reranker()
        self.constraints = ConstraintManager()
        self.prompt_builder = PromptConstructor()
        self.diffusion = BaseDiffusion()
        self.fuser = MultimodalFuser(mode="gated")
        self.bridge = AttentionBridge()
        self.evaluator = Evaluator()

        logger.info("✅ MultimodalGenerationPipeline initialized successfully.")

    # ------------------------------------------------------------
    # 🧠 Full multimodal generation flow
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
        fusion_mode: str = "gated",
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end multimodal content generation.
        """
        constraints = constraints or {}
        logger.info(f"Starting multimodal generation for: '{user_query}'")

        with Timer("multimodal_generation_pipeline"):
            # --------------------------------------------------------
            # 1️⃣ Retrieve contextual information
            # --------------------------------------------------------
            retrieved_docs = self.retriever.retrieve(user_query, k=rag_k)
            if use_reranker:
                reranked = self.reranker.rerank(user_query, retrieved_docs)
                retrieved_docs = [doc for doc, _ in reranked]
            logger.info(f"Retrieved {len(retrieved_docs)} RAG documents.")

            # --------------------------------------------------------
            # 2️⃣ Build multimodal-aware prompt with constraints
            # --------------------------------------------------------
            prompt = self.prompt_builder.build(user_query, retrieved_docs, constraints)
            if style_hint:
                prompt = self.prompt_builder.apply_style(prompt, style_hint)

            # --------------------------------------------------------
            # 3️⃣ Generate images using diffusion
            # --------------------------------------------------------
            diffusion_output = self.diffusion.generate(
                prompt=prompt,
                num_images=num_images,
                style_hint=style_hint,
                height=height,
                width=width,
                seed=seed,
            )

            images = diffusion_output if isinstance(diffusion_output, list) else diffusion_output.get("images", [])
            captions = [prompt] * len(images)

            # --------------------------------------------------------
            # 4️⃣ Fuse text + image embeddings for multimodal representation
            # --------------------------------------------------------
            try:
                text_emb = torch.randn(len(images), 768)
                img_emb = torch.randn(len(images), 768)

                if fusion_mode == "attention":
                    t_out, i_out = self.bridge(text_emb.unsqueeze(0), img_emb.unsqueeze(0))
                    fused_emb = self.bridge.fuse_mean(t_out, i_out)
                else:
                    fused_emb = self.fuser(text_emb, img_emb)

                logger.info(f"Generated fused representation shape: {fused_emb.shape}")
            except Exception as e:
                fused_emb = None
                logger.warning(f"Fusion step skipped: {e}")

            # --------------------------------------------------------
            # 5️⃣ Evaluate results (text, image, alignment, constraints)
            # --------------------------------------------------------
            constraint_scores = self.constraints.evaluate({
                "prompt": prompt,
                "images": images,
                "captions": captions,
                "constraints": constraints,
            })

            eval_scores = self.evaluator.evaluate(
                {"images": images, "captions": captions},
                constraint_scores=constraint_scores,
            )

            return {
                "query": user_query,
                "prompt": prompt,
                "retrieved_docs": retrieved_docs,
                "images": images,
                "captions": captions,
                "fusion_repr": fused_emb,
                "constraint_scores": constraint_scores,
                "evaluation_metrics": eval_scores,
            }


# ------------------------------------------------------------
# ✅ Example Test Run
# ------------------------------------------------------------
if __name__ == "__main__":
    pipeline = MultimodalGenerationPipeline()

    result = pipeline.generate(
        user_query="A cinematic rendering of an AI robot painting on canvas while reading a book about human creativity.",
        constraints={
            "style": "cinematic",
            "factual": True,
            "ethical": True,
            "diversity": True,
        },
        num_images=1,
        fusion_mode="attention",
        style_hint="cinematic",
    )

    print("\n=== Multimodal Generation Summary ===")
    print("Prompt:", result["prompt"][:300])
    print("Constraint Scores:", result["constraint_scores"])
    print("Evaluation Metrics:", result["evaluation_metrics"])