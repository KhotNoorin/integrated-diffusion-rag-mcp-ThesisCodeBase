"""
models/retrieval/index_builder.py

Handles vector index creation, saving, and loading for retrieval.

Supports:
  - FAISS (high-performance dense index)
  - Chroma (persistent vector store, for RAG)
  - Embedding from `Embedder`
"""

from __future__ import annotations
import os
import numpy as np
from typing import List, Tuple, Optional, Union

from utils.logging_utils import get_logger
from utils.config_loader import get_config
from models.retrieval.embedder import Embedder

logger = get_logger("index_builder")

# Optional dependencies
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMA = True
except Exception:
    _HAS_CHROMA = False


class IndexBuilder:
    """
    Unified index builder supporting FAISS or Chroma.

    Example:
        >>> embedder = Embedder()
        >>> idx = IndexBuilder("faiss", dim=512)
        >>> vectors = embedder.encode_texts(["text 1", "text 2"])
        >>> idx.build(vectors, ["doc1", "doc2"])
        >>> idx.save("data/embeddings/index.faiss")
    """

    def __init__(
        self,
        index_type: str = "faiss",
        dim: Optional[int] = None,
        persist_dir: str = "data/embeddings/",
        metric: str = "cosine",
    ):
        cfg = get_config().raw if get_config() else {}
        self.index_type = index_type.lower()
        self.persist_dir = persist_dir or cfg.get("paths", {}).get("embeddings_dir", "data/embeddings/")
        self.metric = metric.lower()
        self.dim = dim or 512
        self.index = None
        self.meta = []
        self._init_index()

    # ------------------------------------------------------------
    # 🧱 Index Initialization
    # ------------------------------------------------------------
    def _init_index(self):
        if self.index_type == "faiss" and _HAS_FAISS:
            metric_type = faiss.METRIC_INNER_PRODUCT if self.metric == "cosine" else faiss.METRIC_L2
            self.index = faiss.IndexFlatIP(self.dim) if metric_type == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(self.dim)
            logger.info(f"Initialized FAISS index (dim={self.dim}, metric={self.metric})")
        elif self.index_type == "chroma" and _HAS_CHROMA:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = chromadb.Client(Settings(persist_directory=self.persist_dir))
            self.collection = self.client.get_or_create_collection("rag_collection")
            logger.info(f"Initialized ChromaDB collection at {self.persist_dir}")
        else:
            logger.warning("No valid index backend found — fallback to in-memory numpy search.")
            self.index = None

    # ------------------------------------------------------------
    # 🧩 Build Index
    # ------------------------------------------------------------
    def build(self, embeddings: np.ndarray, metadata: List[str]):
        """
        Build an index from embeddings and associated metadata (text, paths, etc.).
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be 2D array (N, D)")

        if self.index_type == "faiss" and self.index is not None:
            if self.metric == "cosine":
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.index.add(embeddings.astype(np.float32))
            self.meta.extend(metadata)
            logger.info(f"Added {len(metadata)} vectors to FAISS index.")

        elif self.index_type == "chroma" and _HAS_CHROMA:
            ids = [str(i) for i in range(len(metadata))]
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=metadata,
                ids=ids,
            )
            logger.info(f"Added {len(metadata)} vectors to ChromaDB collection.")

        else:
            self.index = embeddings
            self.meta = metadata
            logger.warning("Built a temporary in-memory index (no FAISS/Chroma).")

    # ------------------------------------------------------------
    # 💾 Save / Load
    # ------------------------------------------------------------
    def save(self, path: Optional[str] = None):
        """
        Save FAISS index or metadata to disk.
        """
        path = path or os.path.join(self.persist_dir, "index.faiss")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if self.index_type == "faiss" and _HAS_FAISS and self.index is not None:
            faiss.write_index(self.index, path)
            meta_path = path.replace(".faiss", "_meta.npy")
            np.save(meta_path, np.array(self.meta))
            logger.info(f"FAISS index saved: {path}")
        elif self.index_type == "chroma" and _HAS_CHROMA:
            self.client.persist()
            logger.info(f"Chroma collection persisted at {self.persist_dir}")
        else:
            np.savez_compressed(path.replace(".faiss", "_simple.npz"), embeddings=self.index, meta=self.meta)
            logger.info(f"Saved simple numpy index to {path}")

    def load(self, path: Optional[str] = None):
        """
        Load an existing FAISS or numpy index.
        """
        path = path or os.path.join(self.persist_dir, "index.faiss")

        if self.index_type == "faiss" and _HAS_FAISS and os.path.exists(path):
            self.index = faiss.read_index(path)
            meta_path = path.replace(".faiss", "_meta.npy")
            if os.path.exists(meta_path):
                self.meta = np.load(meta_path, allow_pickle=True).tolist()
            logger.info(f"Loaded FAISS index with {len(self.meta)} items.")
        elif os.path.exists(path.replace(".faiss", "_simple.npz")):
            data = np.load(path.replace(".faiss", "_simple.npz"), allow_pickle=True)
            self.index = data["embeddings"]
            self.meta = data["meta"].tolist()
            logger.info("Loaded simple numpy index.")
        else:
            logger.warning(f"No index found at {path}")

    # ------------------------------------------------------------
    # 🔍 Search
    # ------------------------------------------------------------
    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search nearest neighbors and return [(metadata, score)].
        """
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]

        if self.index_type == "faiss" and self.index is not None:
            if self.metric == "cosine":
                query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
            distances, indices = self.index.search(query_emb.astype(np.float32), top_k)
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.meta):
                    results.append((self.meta[idx], float(score)))
            return results

        elif self.index_type == "chroma" and _HAS_CHROMA:
            q = query_emb.tolist()
            result = self.collection.query(query_embeddings=q, n_results=top_k)
            docs = result["documents"][0]
            scores = result["distances"][0]
            return list(zip(docs, scores))

        elif isinstance(self.index, np.ndarray):
            sims = (self.index @ query_emb.T).ravel()
            top_idx = np.argsort(-sims)[:top_k]
            return [(self.meta[i], float(sims[i])) for i in top_idx]

        else:
            logger.warning("No valid index found; returning empty results.")
            return []


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    # Create fake embeddings
    data = ["apple is a fruit", "banana is yellow", "cats are animals"]
    embs = np.random.randn(3, 512).astype(np.float32)
    idx = IndexBuilder(index_type="faiss", dim=512)
    idx.build(embs, data)
    idx.save("data/embeddings/test_index.faiss")

    q = np.random.randn(1, 512).astype(np.float32)
    print("Search Results:", idx.search(q))