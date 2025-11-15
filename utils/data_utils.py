"""
utils/data_utils.py

Dataset loading, cleaning, batching, and embedding utilities for:
  - RAG (text retrieval)
  - Diffusion (image data)
  - Constraint evaluation (metadata)
"""

from __future__ import annotations
import os
import json
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image
import re

# ------------------------------------------------------------
# 🧹 Basic Text Utilities
# ------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean and normalize text (used for retrieval + generation).
    """
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_text(text: str) -> List[str]:
    """
    Very basic whitespace tokenization (replace with spaCy or nltk if needed).
    """
    return text.split()


# ------------------------------------------------------------
# 🧾 Data Loading
# ------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL (JSON per line) file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, data: List[Dict[str, Any]]):
    """
    Save a list of dicts to JSONL file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# 🖼️ Image Utilities
# ------------------------------------------------------------

def load_image(path: str, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Load and optionally resize an image.
    """
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.Resampling.LANCZOS)
    return img


def save_image(img: Image.Image, path: str):
    """
    Save PIL image to disk.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


# ------------------------------------------------------------
# 🧩 Multimodal Dataset Handling
# ------------------------------------------------------------

def load_multimodal_dataset(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Load a dataset containing text-image pairs.
    Expects JSONL with fields: {"text": ..., "image_path": ..., "metadata": ...}
    """
    data_path = os.path.join(dataset_dir, "metadata.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No metadata.jsonl found in {dataset_dir}")

    dataset = load_jsonl(data_path)
    samples = []
    for item in dataset:
        text = clean_text(item.get("text", ""))
        image_path = os.path.join(dataset_dir, item.get("image_path", ""))
        metadata = item.get("metadata", {})
        samples.append({"text": text, "image_path": image_path, "metadata": metadata})
    return samples


def sample_dataset(
    dataset: List[Dict[str, Any]],
    n: int = 10,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Randomly sample n items from a dataset for visualization or evaluation.
    """
    random.seed(seed)
    return random.sample(dataset, min(n, len(dataset)))


# ------------------------------------------------------------
# 🔎 Embedding Utilities (for Retrieval)
# ------------------------------------------------------------

def save_embeddings(embeddings: np.ndarray, paths: List[str], out_path: str):
    """
    Save embeddings and their corresponding document/image paths.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, embeddings=embeddings, paths=paths)


def load_embeddings(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings (.npz) and return (array, paths).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["paths"].tolist()


# ------------------------------------------------------------
# 🧮 Batching Utilities
# ------------------------------------------------------------

def batch_data(
    data: List[Any],
    batch_size: int
) -> List[List[Any]]:
    """
    Split a list into batches.
    """
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def collate_multimodal_batch(
    batch: List[Dict[str, Any]],
    image_size: Optional[Tuple[int, int]] = (224, 224)
) -> Tuple[List[str], List[Image.Image]]:
    """
    Collate a batch into (texts, images).
    """
    texts = [item["text"] for item in batch]
    images = [load_image(item["image_path"], size=image_size) for item in batch]
    return texts, images


# ------------------------------------------------------------
# 🧠 Dataset Statistics
# ------------------------------------------------------------

def dataset_summary(dataset: List[Dict[str, Any]]):
    """
    Print dataset summary for quick debugging.
    """
    print(f"Total samples: {len(dataset)}")
    sample_texts = [len(item["text"].split()) for item in dataset if item["text"]]
    avg_len = np.mean(sample_texts) if sample_texts else 0
    print(f"Average text length: {avg_len:.1f} words")

    example = dataset[0] if dataset else {}
    print(f"Example item:\n  Text: {example.get('text', '')[:80]}...\n  Image: {example.get('image_path', '')}")


# ------------------------------------------------------------
# ✅ Example Usage
# ------------------------------------------------------------

if __name__ == "__main__":
    dataset_dir = "data/processed/example_dataset"
    if os.path.exists(dataset_dir):
        dataset = load_multimodal_dataset(dataset_dir)
        dataset_summary(dataset)
        sample = sample_dataset(dataset, 3)
        texts, images = collate_multimodal_batch(sample)
        print(f"\nLoaded {len(images)} images and {len(texts)} texts for demo.")
    else:
        print(f"Demo dataset not found: {dataset_dir}")