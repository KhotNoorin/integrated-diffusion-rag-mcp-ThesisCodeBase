"""
models/multimodal_generator.py

Central module integrating:
  - Retrieval Augmented Generation (RAG)
  - Multi-Constraint Prompting (MCP)
  - Diffusion-based multimodal generation
  - Multimodal fusion (text + image representations)
  - Automatic evaluation (optional)

This serves as the unified pipeline controller for multimodal content generation.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import os
from PIL import Image
import torch

from utils.logging_utils import get_logger
from utils.timer import Timer
from utils.metrics import compute_clipscore

# Core modules
from models.diffusion import BaseDiffusion
from models.retrieval import Retriever, Reranker
from models.constraints import ConstraintManager
from models.fusion import MultimodalFuser, AttentionBridge

logger = get_logger("multimodal_generator")


class MultimodalGenerator:
    """
    Unified multimodal generation controller.

    This module integrates retrieval, constraint management,
    multimodal fusion, and diffusion-based image generation.

    Typical usage:
        >>> mmg = MultimodalGenerator()
        >>> result = mmg.generate("A cyberpunk city at night", constraints={"style": "cyberpunk"})
    """

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        reranker: Optional[Reranker] = None,
        diffusion_model: Optional[BaseDiffusion] = None,
        constraint_manager: Optional[ConstraintManager] = None,
        fuser: Optional[MultimodalFuser] = None,
        bridge: Optional[AttentionBridge] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = retriever or Retriever()
        self.reranker = reranker or Reranker()
        self.diffusion = diffusion_model or BaseDiffusion()
        self.constraint_manager = constraint_manager or ConstraintManager()
        self.fuser = fuser or MultimodalFuser(mode="gated")
        self.bridge = bridge or AttentionBridge()

        logger.info("✅ MultimodalGenerator initialized successfully.")

    # ------------------------------------------------------------
    # 🧠 Main Generation Flow
    # ------------------------------------------------------------
    def generate(
        self,
        user_query: str,
        constraints: Optional[Dict[str, Any]] = None,
        num_images: int = 1,
        rag_k: int = 3,
        use_reranker: bool = True,
        fusion_mode: str = "gated",
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        style_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end multimodal generation pipeline.
        """
        logger.info(f"Starting generation for query: {user_query}")

        # --------------------------------------------------------
        # 1️⃣ Retrieve contextual knowledge (RAG)
        # --------------------------------------------------------
        with Timer("retrieval"):
            retrieved_docs = self.retriever.retrieve(user_query, k=rag_k)
            if use_reranker:
                reranked = self.reranker.rerank(user_query, retrieved_docs)
                retrieved_docs = [doc for doc, _ in reranked]
            logger.info(f"Retrieved {len(retrieved_docs)} supporting documents.")

        # --------------------------------------------------------
        # 2️⃣ Apply constraints (prompt shaping)
        # --------------------------------------------------------
        prompt = self.constraint_manager.apply(constraints or {}, user_query)

        # Optionally include retrieved evidence
        if retrieved_docs:
            evidence = " ".join(retrieved_docs[:rag_k])
            prompt += f"\n\nContextual Knowledge: {evidence}"

        # --------------------------------------------------------
        # 3️⃣ Generate image via Diffusion
        # --------------------------------------------------------
        with Timer("diffusion"):
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
        # 4️⃣ Fuse multimodal representations (optional)
        # --------------------------------------------------------
        fused_repr = None
        try:
            text_emb = torch.randn(len(images), 768)  # placeholder for future integration
            img_emb = torch.randn(len(images), 768)
            if fusion_mode == "attention":
                t_out, i_out = self.bridge(text_emb.unsqueeze(0), img_emb.unsqueeze(0))
                fused_repr = self.bridge.fuse_mean(t_out, i_out)
            else:
                fused_repr = self.fuser(text_emb, img_emb)
            logger.info(f"Fused representation shape: {fused_repr.shape}")
        except Exception as e:
            logger.warning(f"Fusion skipped: {e}")

        # --------------------------------------------------------
        # 5️⃣ Evaluate results
        # --------------------------------------------------------
        clip_score = None
        constraint_scores = None
        try:
            clip_score = compute_clipscore(images, captions)
        except Exception as e:
            logger.warning(f"CLIPScore evaluation failed: {e}")

        try:
            constraint_scores = self.constraint_manager.evaluate({
                "prompt": prompt,
                "images": images,
                "captions": captions,
                "constraints": constraints,
            })
        except Exception as e:
            logger.warning(f"Constraint evaluation failed: {e}")

        return {
            "query": user_query,
            "prompt": prompt,
            "retrieved": retrieved_docs,
            "images": images,
            "captions": captions,
            "clipscore": clip_score,
            "constraints_eval": constraint_scores,
            "fused_repr": fused_repr,
        }

    # ------------------------------------------------------------
    # 🧩 Save Outputs
    # ------------------------------------------------------------
    def save_results(self, results: Dict[str, Any], out_dir: str = "outputs"):
        os.makedirs(out_dir, exist_ok=True)
        for i, img in enumerate(results.get("images", [])):
            img_path = os.path.join(out_dir, f"gen_{i:03d}.png")
            if isinstance(img, Image.Image):
                img.save(img_path)
            logger.info(f"Saved image: {img_path}")

        meta_path = os.path.join(out_dir, "metadata.txt")
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {results.get('prompt')}\n")
            f.write(f"CLIPScore: {results.get('clipscore')}\n")
            f.write(f"Constraints: {results.get('constraints_eval')}\n")
        logger.info(f"Saved metadata to {meta_path}")


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    generator = MultimodalGenerator()
    output = generator.generate(
        "A futuristic robotic chef preparing food",
        constraints={"style": "realistic", "ethical": True, "diversity": True},
        num_images=1,
        rag_k=2,
    )

    print("\n=== Generation Summary ===")
    print("Prompt:", output["prompt"][:200])
    print("Retrieved Docs:", len(output["retrieved"]))
    print("CLIPScore:", output["clipscore"])
    print("Constraint Scores:", output["constraints_eval"])