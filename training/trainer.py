"""
training/trainer.py

Universal trainer for multimodal fine-tuning and constraint-aware learning.

Supports:
  - Fine-tuning CLIP / Diffusion adapters
  - Multi-Constraint Prompting (MCP)-weighted losses
  - Evaluation integration
  - Checkpointing and metric tracking

This serves as the main training engine for:
  - RAG + Diffusion + MCP integration experiments
  - CLIP / ControlNet / Fusion module fine-tuning
"""

from __future__ import annotations
import os
import time
import json
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging_utils import get_logger
from utils.timer import Timer
from training.loss_functions import build_training_loss
from training.dataset_loader import get_dataloader

logger = get_logger("trainer")


# ------------------------------------------------------------
# 🧱 Trainer Class
# ------------------------------------------------------------
class Trainer:
    """
    Generic multimodal trainer supporting constraint-weighted learning.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset_path: str,
        loss_config: Dict[str, float],
        batch_size: int = 8,
        lr: float = 1e-4,
        epochs: int = 5,
        out_dir: str = "training_outputs",
        optimizer_name: str = "adam",
        device: Optional[str] = None,
        val_split: float = 0.1,
        save_every: int = 1,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Loss
        self.criterion = build_training_loss(loss_config)

        # Data
        self.dataloader = get_dataloader(dataset_path, batch_size=batch_size, shuffle=True)
        self.val_loader = None  # optional split

        # Optimizer
        if optimizer_name.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Scheduler (optional)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Config
        self.epochs = epochs
        self.save_every = save_every
        self.global_step = 0

        logger.info(f"Trainer initialized: epochs={epochs}, batch_size={batch_size}, lr={lr}")

    # ------------------------------------------------------------
    # 🧠 Training loop
    # ------------------------------------------------------------
    def train(self):
        logger.info("🚀 Starting training...")
        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Epoch {epoch+1}/{self.epochs}")

            for i, batch in pbar:
                self.global_step += 1
                images = batch.get("image").to(self.device)
                captions = batch.get("caption")

                # Forward pass — placeholder: your model should produce outputs dict
                outputs = self.forward_pass(images, captions)

                # Compute loss
                loss = self.criterion(outputs)
                running_loss += loss.item()

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = running_loss / len(self.dataloader)
            logger.info(f"Epoch {epoch+1}/{self.epochs} completed | Avg Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch + 1, avg_loss)

        total_time = (time.time() - start_time) / 60
        logger.info(f"✅ Training complete in {total_time:.2f} min")

    # ------------------------------------------------------------
    # 🔁 Forward pass adapter
    # ------------------------------------------------------------
    def forward_pass(self, images, captions) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal models.
        Replace this logic with actual model-specific flow.
        """
        # Example placeholder: assume model returns embeddings
        image_emb = self.model.encode_image(images)
        text_emb = self.model.encode_text(captions)
        outputs = {
            "pred": images,
            "target": images,
            "text_emb": text_emb,
            "image_emb": image_emb,
            "gen_emb": image_emb,
            "fact_emb": text_emb,
            "safety_score": torch.rand(images.shape[0], device=self.device),
            "embeddings": image_emb,
        }
        return outputs

    # ------------------------------------------------------------
    # 💾 Checkpoint management
    # ------------------------------------------------------------
    def save_checkpoint(self, epoch: int, loss: float):
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss,
            "global_step": self.global_step,
        }
        path = os.path.join(self.out_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved at {path}")

        # Save summary log
        summary_path = os.path.join(self.out_dir, "training_log.json")
        record = {"epoch": epoch, "avg_loss": loss, "timestamp": time.time()}
        if os.path.exists(summary_path):
            with open(summary_path, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.append(record)
                f.seek(0)
                json.dump(data, f, indent=2)
        else:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump([record], f, indent=2)

    # ------------------------------------------------------------
    # 🧩 Evaluation hook
    # ------------------------------------------------------------
    def evaluate(self):
        if not self.val_loader:
            logger.warning("No validation loader provided.")
            return None
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch.get("image").to(self.device)
                captions = batch.get("caption")
                outputs = self.forward_pass(images, captions)
                loss = self.criterion(outputs)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss


# ------------------------------------------------------------
# ✅ Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    import clip  # type: ignore
    from training.dataset_loader import get_dataloader

    dataset_path = "data/processed/sample_dataset"
    model, _ = clip.load("ViT-B/32", device="cpu")

    loss_config = {
        "recon_l1": 0.0,
        "clip_align": 1.0,
        "mcp": 0.5,
    }

    trainer = Trainer(
        model=model,
        dataset_path=dataset_path,
        loss_config=loss_config,
        batch_size=4,
        lr=1e-4,
        epochs=2,
        out_dir="training_outputs_demo",
    )
    trainer.train()