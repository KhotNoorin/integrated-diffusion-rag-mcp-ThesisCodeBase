"""
models/diffusion/diffusion_pipeline.py

High-level pipeline that integrates:
  - Retrieval (optional) to fetch context for RAG
  - Prompt construction (utils.prompt_utils / PromptBuilder)
  - ConstraintManager to enforce/validate constraints
  - BaseDiffusion (and optional ControlNet) for image generation
  - Optional metrics scoring hooks (CLIPScore) for returned images

Design goals:
  - Modular: any component can be None (graceful fallback)
  - Reproducible: accepts seeds, returns latents when requested
  - Instrumented: uses Timer and logging for profiling
"""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
import os
from PIL import Image
import numpy as np

from utils.config_loader import get_config
from utils.logging_utils import get_logger
from utils.timer import Timer
from utils.prompt_utils import PromptBuilder, clean_prompt

# Optional metric functions (safe import)
from utils.metrics import compute_clipscore  # may raise if CLIP missing

# Model components (may not exist yet, handled at runtime)
try:
    from models.diffusion.base_diffusion import BaseDiffusion
except Exception:
    BaseDiffusion = None

logger = get_logger("diffusion_pipeline")


class DiffusionPipeline:
    """
    High-level pipeline for RAG + Diffusion + MCP generation.

    Args:
        diffusion_model: instance of BaseDiffusion (or None -> placeholder behavior)
        retriever: object with method `retrieve(query, k)` -> List[str] (optional)
        constraint_manager: object with method `apply(constraints, prompt)` and `evaluate(result)` (optional)
        prompt_builder: instance of PromptBuilder (optional)
    """

    def __init__(
        self,
        diffusion_model: Optional[BaseDiffusion] = None,
        retriever: Optional[Any] = None,
        constraint_manager: Optional[Any] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        device: Optional[str] = None,
    ):
        cfg = get_config()
        self.cfg = cfg
        self.device = device or ("cuda" if hasattr(cfg, "raw") and cfg.raw.get("device") is None and False else None)
        self.diffusion = diffusion_model or (BaseDiffusion() if BaseDiffusion is not None else None)
        self.retriever = retriever
        self.constraint_manager = constraint_manager
        self.prompt_builder = prompt_builder or PromptBuilder(mode="image")

        logger.info("DiffusionPipeline initialized.")
        if self.diffusion is None:
            logger.warning("No diffusion model available — pipeline will produce placeholder images.")

    # --------------------------
    # Core generation API
    # --------------------------
    def generate(
        self,
        user_query: str,
        constraints: Optional[Dict[str, Any]] = None,
        num_images: int = 1,
        seed: Optional[int] = None,
        style_hint: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        return_latents: bool = False,
        control_image: Optional[Image.Image] = None,
        controlnet_adapter: Optional[Any] = None,
        height: int = 512,
        width: int = 512,
        rag_k: int = 3,
        compute_scores: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        End-to-end generation step.

        Returns a dict containing:
          - images: List[PIL.Image]
          - captions: List[str]  (same as generated prompt or returned caption)
          - latents: optional numpy array (if requested)
          - clipscore: optional float (average) if computed
          - constraints_eval: optional dict from constraint_manager.evaluate(...)
        """
        # 1) Retrieve context (if retriever provided)
        retrieved = []
        if self.retriever is not None:
            try:
                with Timer("retriever"):
                    retrieved = self.retriever.retrieve(user_query, k=rag_k)
                logger.info(f"Retrieved {len(retrieved)} docs for prompt.")
            except Exception as e:
                logger.warning(f"Retriever failed: {e}")
                retrieved = []

        # 2) Build prompt with constraints + retrieved context
        prompt = self.prompt_builder.build(user_query, retrieved_contexts=retrieved, constraints=constraints)
        prompt = clean_prompt(prompt)
        logger.info(f"Constructed prompt (len={len(prompt)} chars).")

        # 3) Optionally let constraint manager transform or validate prompt
        if self.constraint_manager is not None:
            try:
                with Timer("constraint.apply"):
                    if hasattr(self.constraint_manager, "apply"):
                        prompt = self.constraint_manager.apply(constraints or {}, prompt)
                        logger.info("Constraints applied to prompt.")
            except Exception as e:
                logger.warning(f"Constraint application failed: {e}")

        # 4) If controlnet supplied, route to controlnet adapter
        images = []
        latents = None
        if control_image is not None and controlnet_adapter is not None:
            logger.info("Using ControlNet adapter for conditioned generation.")
            try:
                with Timer("controlnet_pipeline"):
                    images = controlnet_adapter.generate(
                        prompt=prompt,
                        control_image=control_image,
                        num_inference_steps=kwargs.get("num_inference_steps", 30),
                        guidance_scale=kwargs.get("guidance_scale", 7.5),
                        seed=seed,
                        height=height,
                        width=width,
                    )
            except Exception as e:
                logger.warning(f"ControlNet generation failed: {e}")
                images = []
        else:
            # 5) Diffusion generate
            if self.diffusion is None:
                logger.warning("No diffusion engine — returning placeholder(s).")
                placeholder = Image.new("RGB", (width, height), color=(200, 200, 200))
                images = [placeholder] * num_images
            else:
                try:
                    with Timer("diffusion_pipeline.generate"):
                        out = self.diffusion.generate(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_images=num_images,
                            guidance_scale=kwargs.get("guidance_scale", None),
                            num_inference_steps=kwargs.get("num_inference_steps", None),
                            seed=seed,
                            style_hint=style_hint,
                            return_latents=return_latents,
                            height=height,
                            width=width,
                            **kwargs,
                        )
                    if isinstance(out, dict) and "images" in out:
                        images = out["images"]
                        latents = out.get("latents", None)
                    elif isinstance(out, list):
                        images = out
                    else:
                        # unexpected return
                        images = out if isinstance(out, list) else []
                except Exception as e:
                    logger.exception(f"Diffusion generate failed: {e}")
                    images = [Image.new("RGB", (width, height), color=(220, 220, 220))] * num_images

        # Ensure images list length matches requested
        if len(images) < num_images:
            # pad with placeholders
            placeholder = Image.new("RGB", (width, height), color=(200, 200, 200))
            images += [placeholder] * (num_images - len(images))

        # 6) Optional evaluation hooks (CLIPScore)
        clipscore = None
        captions = [prompt] * len(images)
        if compute_scores:
            try:
                # compute average CLIPScore between each image and caption
                clipscore = compute_clipscore(images, captions)
                logger.info(f"CLIPScore (avg): {clipscore:.4f}")
            except Exception as e:
                logger.warning(f"CLIPScore computation failed (missing CLIP?): {e}")

        # 7) Constraint evaluation (post-hoc)
        constraints_eval = None
        if self.constraint_manager is not None and hasattr(self.constraint_manager, "evaluate"):
            try:
                constraints_eval = self.constraint_manager.evaluate({
                    "prompt": prompt,
                    "images": images,
                    "captions": captions,
                    "constraints": constraints,
                })
            except Exception as e:
                logger.warning(f"Constraint evaluation failed: {e}")

        result = {
            "images": images,
            "captions": captions,
            "latents": latents,
            "clipscore": clipscore,
            "constraints_eval": constraints_eval,
            "retrieved": retrieved,
            "prompt": prompt,
        }
        return result

    # --------------------------
    # Utilities
    # --------------------------
    def save_outputs(self, outputs: Dict[str, Any], out_dir: str, prefix: str = "run"):
        """
        Save generated images and metadata to out_dir with an index-based naming.
        """
        os.makedirs(out_dir, exist_ok=True)
        images = outputs.get("images", [])
        captions = outputs.get("captions", [""] * len(images))
        for i, img in enumerate(images):
            img_path = os.path.join(out_dir, f"{prefix}_img_{i:03d}.png")
            if isinstance(img, Image.Image):
                img.save(img_path)
            else:
                try:
                    Image.fromarray(np.asarray(img)).save(img_path)
                except Exception:
                    logger.warning(f"Unable to save image #{i}")
            # save caption
            cap_path = os.path.join(out_dir, f"{prefix}_cap_{i:03d}.txt")
            with open(cap_path, "w", encoding="utf-8") as f:
                f.write(captions[i])

        # save prompt + metadata
        meta_path = os.path.join(out_dir, f"{prefix}_meta.json")
        try:
            import json
            meta = {
                "prompt": outputs.get("prompt"),
                "clipscore": outputs.get("clipscore"),
                "constraints_eval": outputs.get("constraints_eval"),
                "retrieved": outputs.get("retrieved"),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            logger.warning("Failed to write metadata JSON.")

# ------------------------------------------------------------
# Quick test stub (guarded)
# ------------------------------------------------------------
if __name__ == "__main__":
    # This quick demo will run in degraded mode if full dependencies are missing.
    dp = DiffusionPipeline()
    out = dp.generate("A scenic painting of a mountain sunrise", constraints={"style": "oil painting", "factual": True}, num_images=1, seed=123)
    print("Generated keys:", list(out.keys()))
    print("Images count:", len(out.get("images", [])))