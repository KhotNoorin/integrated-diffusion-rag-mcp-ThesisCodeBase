"""
models/retrieval/embedder.py

Embedding utilities for text and images.

Features:
  - Supports CLIP (multimodal image+text embeddings) if available.
  - Supports SentenceTransformers (text-only) as fallback.
  - Batched encoding with device handling.
  - Save / load embeddings helper.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Union
import os
import numpy as np
import logging

# Optional heavy deps
try:
    import torch
except Exception:
    torch = None

# CLIP (OpenAI) via clip package
try:
    import clip
    from PIL import Image
    _HAS_CLIP = True
except Exception:
    _HAS_CLIP = False

# SentenceTransformers for text embeddings fallback
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

from utils.logging_utils import get_logger
logger = get_logger("embedder")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def l2_norm(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norms + eps)


# ------------------------------------------------------------
# Embedder class
# ------------------------------------------------------------

class Embedder:
    """
    Unified embedder supporting CLIP (image + text) and SentenceTransformers (text).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        use_clip: bool = True,
    ):
        self.device = device or ("cuda" if torch is not None and torch.cuda.is_available() else "cpu")
        self.use_clip = use_clip and _HAS_CLIP
        self.clip_model = None
        self.clip_preprocess = None
        self.st_model = None

        if self.use_clip:
            try:
                model_name = model_name or "ViT-B/32"
                logger.info(f"Loading CLIP model: {model_name} on {self.device}")
                self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
                self.clip_model.eval()
            except Exception as e:
                logger.warning(f"Failed to load CLIP: {e}")
                self.use_clip = False

        if (not self.use_clip) and _HAS_ST:
            try:
                st_name = model_name or "all-MiniLM-L6-v2"
                logger.info(f"Loading SentenceTransformer model: {st_name} on cpu")
                self.st_model = SentenceTransformer(st_name)
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
                self.st_model = None

        if not self.use_clip and self.st_model is None:
            logger.warning("No embedding model available. Install 'clip' or 'sentence-transformers' for embeddings.")

    # -------------------------
    # Text encoding
    # -------------------------
    def encode_texts(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        """
        Encode a list of texts to embeddings.
        Returns: np.ndarray of shape (N, D)
        """
        if self.use_clip:
            return self._encode_texts_clip(texts, batch_size, normalize)
        if self.st_model is not None:
            return self._encode_texts_st(texts, batch_size, normalize)
        raise RuntimeError("No text embedding model available.")

    def _encode_texts_clip(self, texts: List[str], batch_size: int, normalize: bool) -> np.ndarray:
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                tokens = clip.tokenize(batch).to(self.device)
                emb = self.clip_model.encode_text(tokens)  # (B, D)
                emb = emb.cpu().numpy()
                all_embeds.append(emb)
        emb = np.vstack(all_embeds)
        return l2_norm(emb) if normalize else emb

    def _encode_texts_st(self, texts: List[str], batch_size: int, normalize: bool) -> np.ndarray:
        emb = self.st_model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        emb = np.asarray(emb)
        return l2_norm(emb) if normalize else emb

    # -------------------------
    # Image encoding (CLIP only)
    # -------------------------
    def encode_images(self, images: List[Union[str, Image.Image]], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        """
        Encode a list of PIL Images or image paths using CLIP.
        """
        if not self.use_clip:
            raise RuntimeError("Image encoding requires CLIP. Install 'clip' package and model weights.")

        from PIL import Image as PILImage

        def _load(i):
            if isinstance(i, PILImage.Image):
                return i
            return PILImage.open(i).convert("RGB")

        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                imgs = [self.clip_preprocess(_load(x)).unsqueeze(0) for x in batch]
                imgs = torch.cat(imgs, dim=0).to(self.device)
                emb = self.clip_model.encode_image(imgs)
                emb = emb.cpu().numpy()
                all_embeds.append(emb)
        emb = np.vstack(all_embeds)
        return l2_norm(emb) if normalize else emb

    # -------------------------
    # Save / Load embeddings
    # -------------------------
    def save_embeddings(self, embeddings: np.ndarray, paths: List[str], out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, embeddings=embeddings, paths=np.array(paths, dtype=object))
        logger.info(f"Saved embeddings to {out_path}")

    def load_embeddings(self, path: str) -> Tuple[np.ndarray, List[str]]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        data = np.load(path, allow_pickle=True)
        embeds = data["embeddings"]
        paths = data["paths"].tolist() if hasattr(data["paths"], "tolist") else list(data["paths"])
        return embeds, paths


# ------------------------------------------------------------
# Quick test (guarded)
# ------------------------------------------------------------
if __name__ == "__main__":
    e = Embedder()
    texts = ["a cat on a skateboard", "a scenic mountain view"]
    try:
        te = e.encode_texts(texts)
        print("Text embeds shape:", te.shape)
    except Exception as ex:
        logger.error(f"Text encode failed: {ex}")

    if _HAS_CLIP:
        from PIL import Image
        dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
        ie = e.encode_images([dummy, dummy])
        print("Image embeds shape:", ie.shape)