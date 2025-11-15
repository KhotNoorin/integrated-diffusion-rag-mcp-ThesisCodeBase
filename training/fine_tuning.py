"""
training/fine_tuning.py

Utilities and training loops to fine-tune multimodal components:
  - Fine-tune CLIP's image/text heads (contrastive / similarity loss)
  - Fine-tune an adapter for ControlNet (or a lightweight UNet adapter)
  - Support constraint-aware loss weighting (e.g., factuality/style/ethical/diversity penalties)

Design goals:
  - Modular: support different backbones via optional imports
  - Safe: graceful fallback if a dependency isn't installed
  - Practical: checkpointing, mixed precision (if available), metric logging
"""

from __future__ import annotations
import os
import time
from typing import Optional, Dict, Any, Iterable, Tuple
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.logging_utils import get_logger
from utils.timer import Timer

logger = get_logger("fine_tuning")

# Optional CLIP (openai/clip)
try:
    import clip  # type: ignore
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

# Optional diffusers / controlnet (for adapter finetuning)
try:
    from diffusers import ControlNetModel  # type: ignore
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False

# Optional apex/amp (use native torch.cuda.amp if available)
_HAS_AMP = hasattr(torch.cuda, "amp")


# -------------------------------------------------------------------------
# Helper losses and utilities
# -------------------------------------------------------------------------
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (InfoNCE) for contrastive training.
    Assumes input: image_embeds (B, D), text_embeds (B, D)
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        # normalize
        image_emb = image_emb / (image_emb.norm(dim=-1, keepdim=True) + 1e-10)
        text_emb = text_emb / (text_emb.norm(dim=-1, keepdim=True) + 1e-10)

        logits = image_emb @ text_emb.t()  # (B, B)
        logits = logits / self.temperature

        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2.0


def save_checkpoint(state: dict, out_dir: str, name: str = "checkpoint.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    logger.info(f"Saved checkpoint: {path}")


# -------------------------------------------------------------------------
# Lightweight adapter example (for CLIP or UNet adapters)
# -------------------------------------------------------------------------
class SmallAdapter(nn.Module):
    """
    Small bottleneck adapter: D -> bottleneck -> D (residual).
    Useful to fine-tune small number of params.
    """
    def __init__(self, dim: int = 768, bottleneck: int = 128):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


# -------------------------------------------------------------------------
# Fine-tuning CLIP (contrastive) loop
# -------------------------------------------------------------------------
def finetune_clip(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Optional[str] = None,
    epochs: int = 3,
    out_dir: Optional[str] = None,
    scheduler: Optional[Any] = None,
    adapter: Optional[nn.Module] = None,
    constraint_weighting: Optional[Dict[str, float]] = None,
    clip_grad_norm: Optional[float] = 1.0,
):
    """
    Fine-tune a CLIP model using contrastive InfoNCE loss.
    - model: clip model (loaded via clip.load)
    - dataloader: yields dicts with 'image' (tensor) and 'caption' (str)
    - adapter: optional small adapter module applied to embeddings
    - constraint_weighting: optional dict to apply additional penalties (factual/style/etc) per-sample
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if adapter is not None:
        adapter = adapter.to(device)

    loss_fn = NTXentLoss(temperature=0.07)
    scaler = torch.cuda.amp.GradScaler() if _HAS_AMP and device.startswith("cuda") else None

    global_step = 0
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(dataloader):
            images = batch.get("image")
            captions = batch.get("caption")
            if images is None or captions is None:
                continue

            images = images.to(device)
            # Tokenize captions using CLIP tokenizer
            if _HAS_CLIP:
                tokens = clip.tokenize(captions, truncate=True).to(device)
            else:
                # fallback: dummy tensors
                tokens = torch.zeros((len(captions), 77), dtype=torch.long, device=device)

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    image_emb = model.encode_image(images)
                    text_emb = model.encode_text(tokens)
                    if adapter is not None:
                        image_emb = adapter(image_emb)
                        text_emb = adapter(text_emb)
                    loss = loss_fn(image_emb, text_emb)

                    # Optional constraint-weighted penalties (per-batch)
                    if constraint_weighting:
                        # Example: penalize if constraints indicate low factuality (user supplies per-sample weights)
                        # Here we expect `constraint_weighting` to be a dict of scalars or callable producing scalar.
                        # For simplicity, apply uniform weighting factor if provided.
                        w = float(constraint_weighting.get("scale", 1.0))
                        loss = loss * w

                scaler.scale(loss).backward()
                if clip_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                image_emb = model.encode_image(images)
                text_emb = model.encode_text(tokens)
                if adapter is not None:
                    image_emb = adapter(image_emb)
                    text_emb = adapter(text_emb)
                loss = loss_fn(image_emb, text_emb)
                loss.backward()
                if clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass

            step_loss = float(loss.detach().cpu().item())
            epoch_loss += step_loss
            global_step += 1

            if batch_idx % 10 == 0:
                logger.info(f"Epoch[{epoch+1}] Step[{batch_idx}] Loss: {step_loss:.4f}")

        avg_loss = epoch_loss / (batch_idx + 1)
        epoch_time = time.time() - t0
        logger.info(f"Epoch {epoch+1} finished — avg loss: {avg_loss:.4f} ({epoch_time:.1f}s)")

        # Save checkpoint per epoch
        if out_dir:
            ckpt = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "adapter_state": adapter.state_dict() if adapter is not None else None,
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(ckpt, out_dir, name=f"clip_finetune_epoch{epoch+1}.pt")

    return model, adapter


# -------------------------------------------------------------------------
# Fine-tune ControlNet adapter (simplified)
# -------------------------------------------------------------------------
def finetune_controlnet_adapter(
    controlnet_model: Any,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: Optional[str] = None,
    epochs: int = 3,
    out_dir: Optional[str] = None,
    scheduler: Optional[Any] = None,
    loss_fn: Optional[nn.Module] = None,
):
    """
    Fine-tune ControlNet adapter weights. This is intentionally generic:
    - controlnet_model: a ControlNetModel instance from diffusers
    - dataloader: yields dicts with 'image', 'condition' (guidance map), 'target_image'
    - loss_fn: pixel or perceptual loss (MSE, VGG perceptual)
    """
    if not _HAS_DIFFUSERS:
        raise RuntimeError("diffusers.ControlNet training requires diffusers installed.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    controlnet_model = controlnet_model.to(device)
    loss_fn = loss_fn or nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler() if _HAS_AMP and device.startswith("cuda") else None

    for epoch in range(epochs):
        controlnet_model.train()
        t0 = time.time()
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            cond_img = batch.get("condition").to(device)  # e.g., edge map tensor
            target_img = batch.get("target").to(device)
            raw_img = batch.get("image").to(device)

            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    # Model-specific forward: returning latent / predicted residuals depends on ControlNet internals
                    outputs = controlnet_model(cond_img, return_dict=True)
                    # Simplified surrogate: compute loss between cond_img and target
                    loss = loss_fn(outputs[0], target_img)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = controlnet_model(cond_img)
                loss = loss_fn(outputs[0], target_img)
                loss.backward()
                optimizer.step()

            epoch_loss += float(loss.detach().cpu().item())
            if batch_idx % 10 == 0:
                logger.info(f"ControlNet Epoch[{epoch+1}] Step[{batch_idx}] Loss: {loss:.4f}")

        avg_loss = epoch_loss / (batch_idx + 1)
        logger.info(f"ControlNet Epoch {epoch+1} avg loss: {avg_loss:.4f} time: {time.time() - t0:.1f}s")

        if out_dir:
            ckpt = {
                "epoch": epoch + 1,
                "model_state": controlnet_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            save_checkpoint(ckpt, out_dir, name=f"controlnet_epoch{epoch+1}.pt")

    return controlnet_model


# -------------------------------------------------------------------------
# High-level trainer wrapper
# -------------------------------------------------------------------------
def train(
    *,
    task: str = "clip",
    dataset_path: Optional[str] = None,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 1e-4,
    out_dir: str = "training_checkpoints",
    adapter_bottleneck: int = 128,
    pretrained_clip_name: str = "ViT-B/32",
):
    """
    Entry point to run a training job.
    - task: "clip" | "controlnet"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    # Lazy import local dataset loader to avoid circular imports
    from training.dataset_loader import get_dataloader

    loader = get_dataloader(dataset_path, batch_size=batch_size, shuffle=True)  # type: ignore

    if task == "clip":
        if not _HAS_CLIP:
            raise RuntimeError("CLIP is required for CLIP fine-tuning. Install the 'clip' package.")
        # Load CLIP model
        model, _ = clip.load(pretrained_clip_name, device=device)
        # Freeze majority of weights and add adapter
        for p in model.parameters():
            p.requires_grad = False
        # attach adapters to image & text encoders
        adapter = SmallAdapter(dim=model.visual.output_dim, bottleneck=adapter_bottleneck)
        # Only train adapter params
        optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)
        logger.info("Starting CLIP fine-tuning (adapter-only).")
        model, adapter = finetune_clip(
            model=model,
            dataloader=loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            out_dir=out_dir,
            adapter=adapter,
        )
        # Save final adapter
        save_checkpoint({"adapter_state": adapter.state_dict()}, out_dir, "clip_adapter_final.pt")
        return model, adapter

    elif task == "controlnet":
        if not _HAS_DIFFUSERS:
            raise RuntimeError("diffusers is required for ControlNet fine-tuning.")
        # Example: load ControlNetModel (user should swap model id)
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
        # Optionally freeze base and train small adapter layers
        optimizer = torch.optim.Adam(controlnet.parameters(), lr=lr)
        logger.info("Starting ControlNet fine-tuning.")
        controlnet = finetune_controlnet_adapter(
            controlnet_model=controlnet,
            dataloader=loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            out_dir=out_dir,
        )
        return controlnet

    else:
        raise ValueError(f"Unknown training task: {task}")


# -------------------------------------------------------------------------
# Example CLI usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune multimodal components")
    parser.add_argument("--task", type=str, default="clip", choices=["clip", "controlnet"])
    parser.add_argument("--dataset", type=str, required=True, help="Path to processed dataset folder")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="training_checkpoints")
    parser.add_argument("--adapter_bottleneck", type=int, default=128)
    args = parser.parse_args()

    train(
        task=args.task,
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        out_dir=args.out_dir,
        adapter_bottleneck=args.adapter_bottleneck,
    )