"""
models/fusion/__init__.py

Public API for the fusion subpackage.

This package provides neural mechanisms for combining multimodal embeddings
(text + image) before or during generation.

Includes:
  - MultimodalFuser     → Lightweight fusion layer (concat, gated, or sum)
  - AttentionBridge     → Transformer-based cross-attention fusion module

Example:
    from models.fusion import MultimodalFuser, AttentionBridge

    text_emb = torch.randn(1, 768)
    img_emb = torch.randn(1, 768)
    fuser = MultimodalFuser(mode="gated")
    fused = fuser(text_emb, img_emb)

    bridge = AttentionBridge(embed_dim=768)
    text_out, img_out = bridge(text_batch, img_batch)
"""

from .multimodal_fuser import MultimodalFuser  # type: ignore
from .attention_bridge import AttentionBridge  # type: ignore

__all__ = [
    "MultimodalFuser",
    "AttentionBridge",
]
