"""
models/diffusion/__init__.py

Public API for the diffusion subpackage.

Re-exports the main classes so callers can import them easily:

    from models.diffusion import BaseDiffusion, ControlNetAdapter, UNetConfig, DiffusionPipeline

Keep this file small and avoid heavy work on import.
"""

from .base_diffusion import BaseDiffusion  # type: ignore
from .controlnet_adapter import ControlNetAdapter  # type: ignore
from .unet_config import UNetConfig  # type: ignore
from .diffusion_pipeline import DiffusionPipeline  # type: ignore

__all__ = [
    "BaseDiffusion",
    "ControlNetAdapter",
    "UNetConfig",
    "DiffusionPipeline",
]
