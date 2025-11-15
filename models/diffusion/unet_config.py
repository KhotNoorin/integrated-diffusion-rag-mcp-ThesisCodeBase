"""
models/diffusion/unet_config.py

Configuration and helper utilities for the UNet backbone
used inside diffusion models (e.g., Stable Diffusion).

Handles:
  - Loading and saving UNet configs
  - Inspecting architecture details
  - Modifying attention or channel parameters for fine-tuning
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os
import json
from utils.logging_utils import get_logger

logger = get_logger("unet_config")

try:
    from diffusers import UNet2DConditionModel
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False


class UNetConfig:
    """
    Stores configuration parameters for UNet2DConditionModel.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        attention_resolutions: str = "4,2,1",
        num_res_blocks: int = 2,
        channel_mult: str = "1,2,4,4",
        use_spatial_transformer: bool = True,
        transformer_depth: int = 1,
        context_dim: int = 768,
        use_checkpoint: bool = False,
        dtype: str = "float16",
    ):
        self.params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "model_channels": model_channels,
            "attention_resolutions": attention_resolutions,
            "num_res_blocks": num_res_blocks,
            "channel_mult": channel_mult,
            "use_spatial_transformer": use_spatial_transformer,
            "transformer_depth": transformer_depth,
            "context_dim": context_dim,
            "use_checkpoint": use_checkpoint,
            "dtype": dtype,
        }

    @classmethod
    def from_file(cls, path: str) -> "UNetConfig":
        if not os.path.exists(path):
            raise FileNotFoundError(f"UNet config not found: {path}")
        with open(path, "r") as f:
            params = json.load(f)
        return cls(**params)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.params, f, indent=2)
        logger.info(f"UNet configuration saved to {path}")

    def summary(self) -> str:
        summary = "\n".join([f"{k:25s}: {v}" for k, v in self.params.items()])
        return f"UNet Configuration:\n{summary}"

    def to_diffusers_model(self) -> Optional["UNet2DConditionModel"]:
        """
        Instantiate a UNet2DConditionModel from diffusers using stored params.
        """
        if not _HAS_DIFFUSERS:
            logger.warning("Diffusers not installed — cannot create UNet model.")
            return None

        logger.info("Creating UNet2DConditionModel from stored config...")
        # parse string params
        def parse_list(x):
            return [int(i) for i in str(x).split(",")]

        config_dict = {
            "sample_size": 64,
            "in_channels": self.params["in_channels"],
            "out_channels": self.params["out_channels"],
            "block_out_channels": [int(self.params["model_channels"]) * m for m in parse_list(self.params["channel_mult"])],
            "down_block_types": (
                ["DownBlock2D", "DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"]
            ),
            "up_block_types": (
                ["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D", "UpBlock2D"]
            ),
            "cross_attention_dim": self.params["context_dim"],
            "attention_head_dim": 8,
        }

        model = UNet2DConditionModel(**config_dict)
        logger.info("UNet model initialized successfully.")
        return model


# ------------------------------------------------------------
# 🧩 UNet Inspector
# ------------------------------------------------------------

def inspect_unet(model: Any):
    """
    Print model architecture summary (layer count, params, attention layers).
    """
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    attn_layers = [n for n, _ in model.named_modules() if "attn" in n.lower()]
    logger.info(f"UNet summary:\n  Total params: {total_params:.2f}M\n  Attention layers: {len(attn_layers)}")
    for n in attn_layers[:5]:
        logger.info(f"  - {n}")


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------

if __name__ == "__main__":
    cfg = UNetConfig(model_channels=320, context_dim=768)
    print(cfg.summary())

    save_path = "models/diffusion/unet_config_example.json"
    cfg.save(save_path)

    if _HAS_DIFFUSERS:
        unet = cfg.to_diffusers_model()
        inspect_unet(unet)