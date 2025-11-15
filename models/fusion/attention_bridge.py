"""
models/fusion/attention_bridge.py

Implements transformer-style cross-attention for multimodal fusion.

Purpose:
  - Allows interaction between text and image features
  - Learns to align semantically related regions/phrases
  - Can serve as a multimodal encoder before diffusion or RAG steps

Core idea:
  text_embs ↔ image_embs
  => cross-attention → joint representation
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from utils.logging_utils import get_logger

logger = get_logger("attention_bridge")


class AttentionBridge(nn.Module):
    """
    Transformer-based attention bridge for multimodal fusion.

    Supports:
      - Bidirectional cross-attention
      - Optional self-attention refinement
      - Outputs unified multimodal embeddings
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.text_to_img_attn = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout
            )
            for _ in range(num_layers)
        ])
        self.img_to_text_attn = (
            nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout
                )
                for _ in range(num_layers)
            ])
            if bidirectional else None
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        logger.info(f"AttentionBridge initialized (bidirectional={bidirectional}, layers={num_layers}, heads={num_heads})")

    # ------------------------------------------------------------
    # 🔁 Forward Pass
    # ------------------------------------------------------------
    def forward(
        self,
        text_embs: torch.Tensor,
        img_embs: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        img_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            text_embs : (B, T, D)
            img_embs  : (B, V, D)
            text_mask : (B, T)
            img_mask  : (B, V)
        Returns:
            text_out, img_out : refined multimodal embeddings
        """

        B, T, D = text_embs.shape
        _, V, _ = img_embs.shape

        text_out = text_embs
        img_out = img_embs

        for layer_idx in range(self.num_layers):
            # Text attending to Image
            attn_layer = self.text_to_img_attn[layer_idx]
            # combine both: [text | image]
            joint = torch.cat([text_out, img_out], dim=1)
            text_out = attn_layer(joint, src_key_padding_mask=None)[:, :T, :]

            # Image attending to Text (if bidirectional)
            if self.bidirectional and self.img_to_text_attn:
                img_layer = self.img_to_text_attn[layer_idx]
                joint = torch.cat([img_out, text_out], dim=1)
                img_out = img_layer(joint, src_key_padding_mask=None)[:, :V, :]

        text_out = self.norm(text_out)
        img_out = self.norm(img_out)

        return self.dropout(text_out), self.dropout(img_out)

    # ------------------------------------------------------------
    # 🧠 Fusion Helper
    # ------------------------------------------------------------
    def fuse_mean(self, text_embs: torch.Tensor, img_embs: torch.Tensor) -> torch.Tensor:
        """
        Fuses attended text/image embeddings by mean pooling.
        """
        text_mean = text_embs.mean(dim=1)
        img_mean = img_embs.mean(dim=1)
        fused = (text_mean + img_mean) / 2
        return self.norm(fused)

    # ------------------------------------------------------------
    # 💡 Alignment Scoring
    # ------------------------------------------------------------
    def compute_cross_similarity(self, text_embs: torch.Tensor, img_embs: torch.Tensor) -> float:
        """
        Computes mean cosine similarity between text and image representations.
        """
        with torch.no_grad():
            text_mean = text_embs.mean(dim=1)
            img_mean = img_embs.mean(dim=1)
            sim = F.cosine_similarity(text_mean, img_mean, dim=-1)
            return float(sim.mean().item())


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, V, D = 2, 5, 8, 768

    text_embs = torch.randn(B, T, D)
    img_embs = torch.randn(B, V, D)

    bridge = AttentionBridge(embed_dim=D, num_heads=8, num_layers=2, bidirectional=True)
    text_out, img_out = bridge(text_embs, img_embs)
    fused = bridge.fuse_mean(text_out, img_out)

    print("Output shapes:", text_out.shape, img_out.shape)
    print("Fused vector shape:", fused.shape)
    print("Cross-modal similarity:", bridge.compute_cross_similarity(text_out, img_out))