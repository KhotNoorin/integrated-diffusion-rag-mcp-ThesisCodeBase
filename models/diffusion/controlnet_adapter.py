"""
models/diffusion/controlnet_adapter.py

Optional ControlNet integration for the diffusion pipeline.
Allows conditioning image generation on structure maps such as:
  - Canny edges
  - Depth maps
  - Segmentation masks
  - Pose keypoints

Requires diffusers >= 0.14 and controlnet model weights.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any, List
import torch
from PIL import Image
import numpy as np

from utils.logging_utils import get_logger
from utils.timer import Timer

logger = get_logger("controlnet_adapter")

try:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        ControlNetModel,
        UniPCMultistepScheduler
    )
    _HAS_CONTROLNET = True
except Exception:
    _HAS_CONTROLNET = False


class ControlNetAdapter:
    """
    Wrapper class to load and apply ControlNet conditioning to diffusion.
    """

    def __init__(
        self,
        base_model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_id: Optional[str] = "lllyasviel/sd-controlnet-canny",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device.startswith("cuda") else torch.float32)

        if not _HAS_CONTROLNET:
            logger.warning("Diffusers ControlNet not available — skipping ControlNet setup.")
            self.pipe = None
            return

        logger.info(f"Loading ControlNet model: {controlnet_model_id}")
        try:
            controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id, torch_dtype=self.dtype
            )
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
            )

            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to(self.device)

            # memory optimizations
            if self.device.startswith("cuda"):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass

            self.pipe = pipe
            logger.info("ControlNet pipeline loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load ControlNet: {e}")
            self.pipe = None

    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        height: int = 512,
        width: int = 512,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate conditioned images using ControlNet.
        """
        if self.pipe is None:
            logger.warning("ControlNet pipeline unavailable, returning placeholder.")
            placeholder = Image.new("RGB", (width, height), color=(180, 180, 180))
            return [placeholder]

        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        control_image = control_image.convert("RGB").resize((width, height))
        with Timer("controlnet.generate", verbose=True):
            result = self.pipe(
                prompt=prompt,
                image=control_image,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
                **kwargs,
            )

        return result.images if hasattr(result, "images") else result

    @staticmethod
    def load_guidance_image(path: str, mode: str = "canny") -> Image.Image:
        """
        Load and preprocess guidance image (Canny edge or similar).
        """
        img = Image.open(path).convert("RGB")
        if mode == "canny":
            try:
                import cv2
                np_img = np.array(img)
                edges = cv2.Canny(np_img, 100, 200)
                edge_rgb = np.stack([edges]*3, axis=-1)
                return Image.fromarray(edge_rgb)
            except Exception:
                logger.warning("cv2 not installed or Canny failed — returning original image.")
                return img
        else:
            return img


# ------------------------------------------------------------
# ✅ Quick Test
# ------------------------------------------------------------
if __name__ == "__main__":
    adapter = ControlNetAdapter()
    if adapter.pipe:
        # using dummy gray image for test
        dummy_img = Image.new("RGB", (256, 256), color=(200, 200, 200))
        result = adapter.generate("a futuristic city skyline", dummy_img, num_inference_steps=10)
        print(f"Generated {len(result)} conditioned image(s).")
    else:
        print("ControlNet not available — skipped test.")