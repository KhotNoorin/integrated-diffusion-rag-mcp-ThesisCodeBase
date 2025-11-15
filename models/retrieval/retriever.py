"""
models/retrieval/retriever.py

Retriever module for multimodal RAG.

Integrates:
  - Embedder (text/image embedding)
  - IndexBuilder (FAISS / Chroma)
  - Optional hybrid (text + image) retrieval
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from utils.logging_utils import get_logger
from models.retrieval.embedder import Embedder
from models.retrieval.index_builder import IndexBuilder

logger = get_logger("retriever")


class Retriever:
    """
    High-level retriever for multimodal RAG.
    Supports:
      - Dense text retrieval (SentenceTransformer or CLIP text embeddings)
      - Image-conditioned retrieval (if CLIP image embeddings available)
      - Hybrid scoring (text + image)
    """

    def __init__(
        self,
        index_builder: Optional[IndexBuilder] = None,
        embedder: Optional[Embedder] = None,
        top_k: int = 5,
        hybrid_weight: float = 0.5,
    ):
        self.embedder = embedder or Embedder()
        self.index = index_builder or IndexBuilder(index_type="faiss", dim=512)
        self.top_k = top_k
        self.hybrid_weight = hybrid_weight

        logger.info(f"Retriever initialized (top_k={top_k}, hybrid_weight={hybrid_weight}).")

    # ------------------------------------------------------------
    # 🔍 Core Retrieval
    # ------------------------------------------------------------
    def retrieve(
        self,
        query: Union[str, Dict[str, Union[str, np.ndarray]]],
        k: Optional[int] = None,
        use_image: bool = False,
        image: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Retrieve top-k relevant documents for a query.

        Args:
            query: user query text (or dict with "text" and "image")
            k: number of results to return
            use_image: whether to include image similarity
            image: optional image (PIL or np.ndarray) for CLIP hybrid
        Returns:
            list of top-ranked metadata entries (texts, paths, etc.)
        """
        k = k or self.top_k

        # 1️⃣ Encode text query
        if isinstance(query, dict):
            text_query = query.get("text", "")
        else:
            text_query = query

        with np.errstate(all="ignore"):
            text_emb = self.embedder.encode_texts([text_query])
            text_emb = np.asarray(text_emb, dtype=np.float32)

        # 2️⃣ If hybrid, also encode image
        if use_image and image is not None:
            try:
                image_emb = self.embedder.encode_images([image])
                image_emb = np.asarray(image_emb, dtype=np.float32)
                # combine with weighting
                query_emb = (
                    self.hybrid_weight * text_emb + (1 - self.hybrid_weight) * image_emb
                )
            except Exception as e:
                logger.warning(f"Image embedding failed ({e}), using text only.")
                query_emb = text_emb
        else:
            query_emb = text_emb

        # 3️⃣ Search index
        try:
            results = self.index.search(query_emb, top_k=k)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            results = []

        # 4️⃣ Format results
        retrieved_texts = [r[0] for r in results]
        scores = [r[1] for r in results]
        if not retrieved_texts:
            logger.warning("No documents retrieved.")
        else:
            logger.info(f"Retrieved {len(retrieved_texts)} documents (avg score={np.mean(scores):.3f}).")

        return retrieved_texts

    # ------------------------------------------------------------
    # 🧠 Add documents dynamically
    # ------------------------------------------------------------
    def add_documents(self, texts: List[str]):
        """
        Add new text entries to index (encodes + updates FAISS/Chroma).
        """
        logger.info(f"Adding {len(texts)} documents to index...")
        try:
            embeddings = self.embedder.encode_texts(texts)
            self.index.build(embeddings, texts)
            logger.info("Documents added successfully.")
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")

    # ------------------------------------------------------------
    # 💾 Save / Load Index
    # ------------------------------------------------------------
    def save_index(self, path: str):
        self.index.save(path)

    def load_index(self, path: str):
        self.index.load(path)


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    # Fake dataset
    docs = [
        "Renewable energy reduces carbon emissions.",
        "Artificial intelligence enables multimodal generation.",
        "Stable Diffusion is a latent text-to-image model.",
        "Reinforcement learning is used for policy optimization.",
        "Solar panels convert sunlight into electricity.",
    ]

    # Build index
    embedder = Embedder()
    idx = IndexBuilder(index_type="faiss", dim=512)
    retriever = Retriever(index_builder=idx, embedder=embedder, top_k=2)
    retriever.add_documents(docs)

    # Query
    query = "How does solar energy work?"
    results = retriever.retrieve(query)
    print("\nTop Retrieved Docs:")
    for i, r in enumerate(results):
        print(f"{i+1}. {r}")