"""
models/fusion/multimodal_fuser.py

Combines text and image embeddings into a unified multimodal representation.

Used in:
  - Multimodal generation pipeline
  - RAG + Diffusion integration
  - Evaluation of alignment between modalities

Supports:
  - Simple concatenation or gated fusion
  - Attention-based weighted combination (via AttentionBridge)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from utils.logging_utils import get_logger

logger = get_logger("multimodal_fuser")


class MultimodalFuser(nn.Module):
    """
    Combines text and image embeddings into a single joint representation.

    Fusion modes:
      - 'concat' : Concatenate and linear project
      - 'gated'  : Learn weighting gate between text and image
      - 'sum'    : Direct addition (for equal-dimension embeddings)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        fusion_dim: int = 512,
        mode: str = "gated",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim

        if mode not in ["concat", "gated", "sum"]:
            raise ValueError(f"Unsupported fusion mode: {mode}")

        if mode == "concat":
            self.proj = nn.Linear(embed_dim * 2, fusion_dim)
        elif mode == "gated":
            self.text_proj = nn.Linear(embed_dim, fusion_dim)
            self.image_proj = nn.Linear(embed_dim, fusion_dim)
            self.gate = nn.Linear(fusion_dim * 2, 1)
        elif mode == "sum":
            self.proj = nn.Linear(embed_dim, fusion_dim)

        self.dropout = nn.Dropout(dropout)
        logger.info(f"Initialized MultimodalFuser (mode={mode}, embed_dim={embed_dim}, fusion_dim={fusion_dim})")

    # ------------------------------------------------------------
    # 🔁 Forward Fusion
    # ------------------------------------------------------------
    def forward(
        self,
        text_emb: torch.Tensor,
        image_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb  : (B, D)
            image_emb : (B, D)
        Returns:
            fused_emb : (B, fusion_dim)
        """
        if text_emb.size(0) != image_emb.size(0):
            raise ValueError("Batch size mismatch between text and image embeddings.")

        if self.mode == "concat":
            fused = torch.cat([text_emb, image_emb], dim=-1)
            fused = F.relu(self.proj(fused))

        elif self.mode == "gated":
            t_proj = F.relu(self.text_proj(text_emb))
            i_proj = F.relu(self.image_proj(image_emb))
            gate_val = torch.sigmoid(self.gate(torch.cat([t_proj, i_proj], dim=-1)))
            fused = gate_val * t_proj + (1 - gate_val) * i_proj

        elif self.mode == "sum":
            fused = F.relu(self.proj(text_emb + image_emb))

        fused = self.dropout(fused)
        return fused

    # ------------------------------------------------------------
    # 🧠 Alignment Score
    # ------------------------------------------------------------
    def compute_alignment(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> float:
        """
        Computes cosine similarity between text and image embeddings
        as a measure of multimodal alignment.
        """
        with torch.no_grad():
            sim = F.cosine_similarity(text_emb, image_emb, dim=-1)
            return float(sim.mean().item())

    # ------------------------------------------------------------
    # 🧮 Example Fusion API
    # ------------------------------------------------------------
    def fuse_batch(
        self,
        batch: Dict[str, torch.Tensor],
        keys: Tuple[str, str] = ("text", "image"),
    ) -> torch.Tensor:
        """
        Convenience wrapper for dict-style batch fusion.
        """
        text_emb, image_emb = batch[keys[0]], batch[keys[1]]
        fused = self.forward(text_emb, image_emb)
        return fused


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    text_emb = torch.randn(4, 768)
    image_emb = torch.randn(4, 768)

    fuser = MultimodalFuser(embed_dim=768, fusion_dim=512, mode="gated")
    fused = fuser(text_emb, image_emb)
    print("Fused embedding shape:", fused.shape)
    print("Alignment Score:", fuser.compute_alignment(text_emb, image_emb))