"""
training/loss_functions.py

Defines all loss functions used for training and fine-tuning
multimodal models (Diffusion, CLIP, ControlNet, etc.) with
Retrieval-Augmented Generation (RAG) and Multi-Constraint Prompting (MCP).

Includes:
  - Standard reconstruction & contrastive losses
  - Cross-modal alignment (CLIP-style)
  - Constraint-specific auxiliary losses
  - Composite weighted loss function
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from utils.logging_utils import get_logger

logger = get_logger("loss_functions")


# ------------------------------------------------------------
# 🎯 1. Reconstruction Losses
# ------------------------------------------------------------
class L1Loss(nn.Module):
    """Basic L1 reconstruction loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pred, target)


class L2Loss(nn.Module):
    """Basic L2 (MSE) reconstruction loss."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


# ------------------------------------------------------------
# 🔁 2. Cross-Modal Alignment (CLIP-style)
# ------------------------------------------------------------
class CLIPAlignmentLoss(nn.Module):
    """
    Cross-modal alignment loss for text-image embedding matching.
    Uses cosine similarity and contrastive InfoNCE.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text_emb: torch.Tensor, image_emb: torch.Tensor) -> torch.Tensor:
        text_emb = F.normalize(text_emb, dim=-1)
        image_emb = F.normalize(image_emb, dim=-1)
        logits = (text_emb @ image_emb.T) / self.temperature
        labels = torch.arange(len(text_emb), device=text_emb.device)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2.0


# ------------------------------------------------------------
# 🧠 3. Constraint-Specific Losses (MCP)
# ------------------------------------------------------------
class FactualityLoss(nn.Module):
    """
    Penalizes generation that deviates from retrieved factual content.
    Ground truth factual embeddings or vectorized references can be used.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, gen_emb: torch.Tensor, fact_emb: torch.Tensor) -> torch.Tensor:
        loss = 1 - F.cosine_similarity(gen_emb, fact_emb, dim=-1).mean()
        return self.weight * loss


class StyleConsistencyLoss(nn.Module):
    """
    Ensures stylistic coherence between generated image/text and target style embedding.
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, gen_style_emb: torch.Tensor, target_style_emb: torch.Tensor) -> torch.Tensor:
        return self.weight * F.mse_loss(gen_style_emb, target_style_emb)


class EthicalLoss(nn.Module):
    """
    Penalizes unsafe, biased, or NSFW representations.
    Ideally connected to a pretrained safety classifier.
    """
    def __init__(self, penalty_weight: float = 1.0):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, safety_score: torch.Tensor) -> torch.Tensor:
        """
        Expects `safety_score` between [0,1] (1 = safe, 0 = unsafe).
        """
        unsafe_penalty = (1 - safety_score).mean()
        return self.penalty_weight * unsafe_penalty


class DiversityLoss(nn.Module):
    """
    Promotes diversity among generated samples (reduce redundancy).
    """
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (N, D)
        Encourages lower pairwise cosine similarity.
        """
        emb_norm = F.normalize(embeddings, dim=-1)
        sim_matrix = emb_norm @ emb_norm.T
        mean_sim = (sim_matrix.sum() - torch.trace(sim_matrix)) / (len(embeddings)**2 - len(embeddings))
        diversity_loss = mean_sim
        return self.weight * diversity_loss


# ------------------------------------------------------------
# ⚖️ 4. Composite Multi-Constraint Loss
# ------------------------------------------------------------
class MultiConstraintLoss(nn.Module):
    """
    Combines multiple constraint-driven auxiliary losses into a single objective.
    Used for fine-tuning with Multi-Constraint Prompting (MCP).
    """
    def __init__(
        self,
        factual_w: float = 1.0,
        style_w: float = 1.0,
        ethical_w: float = 1.0,
        diversity_w: float = 1.0,
    ):
        super().__init__()
        self.factual_loss = FactualityLoss(weight=factual_w)
        self.style_loss = StyleConsistencyLoss(weight=style_w)
        self.ethical_loss = EthicalLoss(penalty_weight=ethical_w)
        self.diversity_loss = DiversityLoss(weight=diversity_w)

    def forward(
        self,
        gen_emb: torch.Tensor,
        fact_emb: Optional[torch.Tensor] = None,
        gen_style_emb: Optional[torch.Tensor] = None,
        target_style_emb: Optional[torch.Tensor] = None,
        safety_score: Optional[torch.Tensor] = None,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss_total = 0.0
        if fact_emb is not None:
            loss_total += self.factual_loss(gen_emb, fact_emb)
        if gen_style_emb is not None and target_style_emb is not None:
            loss_total += self.style_loss(gen_style_emb, target_style_emb)
        if safety_score is not None:
            loss_total += self.ethical_loss(safety_score)
        if embeddings is not None:
            loss_total += self.diversity_loss(embeddings)
        return loss_total


# ------------------------------------------------------------
# 🧩 5. Combined Training Loss Builder
# ------------------------------------------------------------
def build_training_loss(config: Dict[str, float]) -> nn.Module:
    """
    Creates a weighted sum loss based on configuration dictionary.
    Example config:
        {
            "recon_l1": 1.0,
            "recon_l2": 0.0,
            "clip_align": 1.0,
            "mcp": 0.5
        }
    """
    losses = {}
    if config.get("recon_l1", 0) > 0:
        losses["l1"] = (L1Loss(), config["recon_l1"])
    if config.get("recon_l2", 0) > 0:
        losses["l2"] = (L2Loss(), config["recon_l2"])
    if config.get("clip_align", 0) > 0:
        losses["clip"] = (CLIPAlignmentLoss(), config["clip_align"])
    if config.get("mcp", 0) > 0:
        losses["mcp"] = (MultiConstraintLoss(), config["mcp"])

    logger.info(f"Initialized composite loss with components: {list(losses.keys())}")

    class CompositeLoss(nn.Module):
        def __init__(self, losses_dict):
            super().__init__()
            self.losses_dict = nn.ModuleDict({k: v[0] for k, v in losses_dict.items()})
            self.weights = {k: v[1] for k, v in losses_dict.items()}

        def forward(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            total_loss = 0.0
            for name, loss_fn in self.losses_dict.items():
                w = self.weights[name]
                if name == "l1":
                    total_loss += w * loss_fn(outputs["pred"], outputs["target"])
                elif name == "l2":
                    total_loss += w * loss_fn(outputs["pred"], outputs["target"])
                elif name == "clip":
                    total_loss += w * loss_fn(outputs["text_emb"], outputs["image_emb"])
                elif name == "mcp":
                    total_loss += w * loss_fn(
                        outputs["gen_emb"],
                        outputs.get("fact_emb"),
                        outputs.get("gen_style_emb"),
                        outputs.get("target_style_emb"),
                        outputs.get("safety_score"),
                        outputs.get("embeddings"),
                    )
            return total_loss

    return CompositeLoss(losses)


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    config = {"recon_l1": 1.0, "clip_align": 1.0, "mcp": 0.5}
    criterion = build_training_loss(config)

    outputs = {
        "pred": torch.randn(4, 3, 64, 64),
        "target": torch.randn(4, 3, 64, 64),
        "text_emb": torch.randn(4, 512),
        "image_emb": torch.randn(4, 512),
        "gen_emb": torch.randn(4, 512),
        "fact_emb": torch.randn(4, 512),
        "safety_score": torch.rand(4),
        "embeddings": torch.randn(4, 512),
    }

    loss = criterion(outputs)
    print(f"Composite loss: {loss.item():.4f}")