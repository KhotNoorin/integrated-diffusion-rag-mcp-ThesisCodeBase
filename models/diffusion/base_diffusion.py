"""
models/diffusion/base_diffusion.py

Core diffusion model wrapper for multimodal generation.
Implements a clean interface for:
  - Loading diffusion pipelines (Stable Diffusion, etc.)
  - Generating images from prompts
  - Optional ControlNet / constraint hooks
  - Performance tracking and safe fallback

Supports diffusers pipelines (Stable Diffusion, SDXL, etc.).
"""

from __future__ import annotations
import os
from typing import Optional, List, Dict, Any, Union

import torch
from PIL import Image
import numpy as np

from utils.config_loader import get_config
from utils.logging_utils import get_logger
from utils.timer import Timer

logger = get_logger("base_diffusion")

# Optional dependency: diffusers
try:
    from diffusers import (
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
    )
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False


class BaseDiffusion:
    """
    A simple, robust wrapper for diffusion pipelines.

    Example:
        >>> from models.diffusion.base_diffusion import BaseDiffusion
        >>> model = BaseDiffusion()
        >>> imgs = model.generate("A sunset over the mountains", num_images=2)
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        scheduler_type: str = "dpmsolver",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 40,
        use_auth_token: Optional[str] = None,
    ):
        cfg = get_config().raw if get_config() else {}

        self.model_id = model_id or cfg.get("diffusion", {}).get(
            "model_id", "runwayml/stable-diffusion-v1-5"
        )
        self.scheduler_type = scheduler_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.use_auth_token = use_auth_token or os.getenv("HF_TOKEN")

        self.pipeline = None
        self._load_pipeline()

    # ------------------------------------------------------------
    # 🧩 Load / configure pipeline
    # ------------------------------------------------------------
    def _load_pipeline(self):
        if not _HAS_DIFFUSERS:
            logger.warning("Diffusers not available — diffusion model disabled.")
            return

        try:
            logger.info(f"Loading diffusion model: {self.model_id}")
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                use_auth_token=self.use_auth_token,
            )

            if self.scheduler_type == "dpmsolver":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            elif self.scheduler_type == "euler":
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

            pipe = pipe.to(self.device)

            # Optional performance optimization
            if self.device.startswith("cuda"):
                try:
                    pipe.enable_attention_slicing()
                    pipe.enable_xformers_memory_efficient_attention()
                    pipe.enable_model_cpu_offload()
                except Exception as e:
                    logger.debug(f"Optimization skipped: {e}")

            self.pipeline = pipe
            logger.info("Diffusion pipeline loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load diffusion pipeline: {e}")
            self.pipeline = None

    # ------------------------------------------------------------
    # 🎨 Generate
    # ------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_images: int = 1,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
        style_hint: Optional[str] = None,
        return_latents: bool = False,
        safety_checker: Optional[Any] = None,
        **kwargs,
    ) -> Union[List[Image.Image], Dict[str, Any]]:
        """
        Generate one or more images given a text prompt.
        """

        # Fallback for missing pipeline
        if self.pipeline is None:
            logger.warning("Pipeline unavailable — returning placeholder images.")
            placeholder = Image.new("RGB", (width, height), color=(200, 200, 200))
            return [placeholder for _ in range(num_images)]

        # Merge optional style hint
        if style_hint:
            prompt = f"{prompt}, {style_hint}"

        guidance_scale = guidance_scale or self.guidance_scale
        num_inference_steps = num_inference_steps or self.num_inference_steps

        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )

        with Timer("diffusion.generate", verbose=True):
            result = self.pipeline(
                prompt=[prompt] * num_images,
                negative_prompt=[negative_prompt] * num_images if negative_prompt else None,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            )

        # Convert result to images
        images = result.images if hasattr(result, "images") else result
        if safety_checker:
            try:
                safe_images, flags = safety_checker(images)
                images = safe_images
                logger.info(f"Safety checker flagged {sum(flags)} outputs.")
            except Exception as e:
                logger.warning(f"Safety check failed: {e}")

        # Convert to PIL RGB
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img.astype("uint8"), "RGB"))
            else:
                try:
                    pil_images.append(Image.fromarray(np.asarray(img)).convert("RGB"))
                except Exception:
                    pil_images.append(Image.new("RGB", (width, height), color=(255, 255, 255)))

        if return_latents and hasattr(result, "latents"):
            latents = (
                result.latents.detach().cpu().numpy()
                if torch.is_tensor(result.latents)
                else np.asarray(result.latents)
            )
            return {"images": pil_images, "latents": latents}

        return pil_images

    # ------------------------------------------------------------
    # ⚙️ Utility
    # ------------------------------------------------------------
    def warmup(self, steps: int = 1):
        """Run a short warm-up to initialize GPU kernels and caches."""
        logger.info("Running diffusion warm-up...")
        try:
            _ = self.generate("A warm-up prompt", num_images=1, num_inference_steps=steps)
            logger.info("Warm-up complete.")
        except Exception as e:
            logger.warning(f"Warm-up skipped: {e}")

    def save_image(self, image: Image.Image, out_path: str):
        """Save a generated image."""
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        image.save(out_path)
        logger.info(f"Image saved: {out_path}")

    def is_available(self) -> bool:
        """Return True if a diffusion model is loaded."""
        return self.pipeline is not None

# ------------------------------------------------------------
# 🧩 Compatibility Wrapper
# ------------------------------------------------------------
class DiffusionModel(BaseDiffusion):
    """
    Backward-compatible alias for BaseDiffusion.
    Required because older modules/tests refer to DiffusionModel.
    """
    pass

# ------------------------------------------------------------
# ✅ Example standalone test
# ------------------------------------------------------------
if __name__ == "__main__":
    model = BaseDiffusion()
    result = model.generate(
        prompt="A futuristic city skyline at sunset",
        num_images=1,
        num_inference_steps=20,
        seed=123,
    )
    print(f"Generated {len(result)} image(s).")
    if isinstance(result, list):
        result[0].save("test_diffusion_output.png")
        print("Saved test_diffusion_output.png")