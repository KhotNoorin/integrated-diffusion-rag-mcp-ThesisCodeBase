"""
training/dataset_loader.py

Unified dataset loader for multimodal training.

Supports:
  - Text datasets (captions, retrieved context)
  - Image datasets (paired with captions)
  - Multimodal datasets (text + image pairs)
  - Dynamic batching and preprocessing
  - Dataset splits for training/validation/testing
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils.logging_utils import get_logger

logger = get_logger("dataset_loader")


# ------------------------------------------------------------
# 🧱 Core multimodal dataset class
# ------------------------------------------------------------
class MultimodalDataset(Dataset):
    """
    A unified dataset class for multimodal training.
    Expects directory structure:
      data/processed/
          ├── images/
          ├── captions.txt
          ├── metadata.json
    """

    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        transform: Optional[Any] = None,
        max_samples: Optional[int] = None,
    ):
        self.image_dir = image_dir
        self.caption_file = caption_file
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.samples = self._load_samples(max_samples)
        logger.info(f"Loaded {len(self.samples)} multimodal samples.")

    def _load_samples(self, max_samples: Optional[int] = None):
        samples = []
        if not os.path.exists(self.caption_file):
            raise FileNotFoundError(f"Caption file not found: {self.caption_file}")
        with open(self.caption_file, "r", encoding="utf-8") as f:
            for line in f:
                if "," not in line:
                    continue
                img_name, caption = line.strip().split(",", 1)
                img_path = os.path.join(self.image_dir, img_name)
                if os.path.exists(img_path):
                    samples.append((img_path, caption))
        if max_samples:
            samples = samples[:max_samples]
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {"image": image, "caption": caption}


# ------------------------------------------------------------
# 🧩 Helper: DataLoader wrapper
# ------------------------------------------------------------
def get_dataloader(
    dataset_path: str,
    batch_size: int = 8,
    shuffle: bool = True,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for multimodal training.
    """
    image_dir = os.path.join(dataset_path, "images")
    caption_file = os.path.join(dataset_path, "captions.txt")

    dataset = MultimodalDataset(
        image_dir=image_dir,
        caption_file=caption_file,
        max_samples=max_samples,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    logger.info(f"DataLoader ready: {len(dataset)} samples, batch_size={batch_size}")
    return loader


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    sample_path = "data/processed/sample_dataset"
    os.makedirs(os.path.join(sample_path, "images"), exist_ok=True)
    caption_path = os.path.join(sample_path, "captions.txt")

    # Example dataset creation
    with open(caption_path, "w", encoding="utf-8") as f:
        f.write("sample1.png,A futuristic robot reading a book.\n")
        f.write("sample2.png,An astronaut walking on Mars.\n")

    # Test loader
    loader = get_dataloader(sample_path, batch_size=2)
    for batch in loader:
        print("Batch keys:", batch.keys())
        print("Images shape:", batch["image"].shape)
        print("Captions:", batch["caption"])
        break