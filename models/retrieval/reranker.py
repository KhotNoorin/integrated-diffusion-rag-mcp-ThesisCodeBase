"""
models/retrieval/reranker.py

Optional cross-encoder reranker for refining retrieved documents.

Used after dense retrieval to rescore top-k results based on
cross-attention similarity (query, document).

Supports:
  - SentenceTransformers cross-encoders (e.g., 'cross-encoder/ms-marco-MiniLM-L-6-v2')
  - Hugging Face models if SentenceTransformers unavailable
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np
from utils.logging_utils import get_logger

logger = get_logger("reranker")

# Optional dependencies
try:
    from sentence_transformers import CrossEncoder
    _HAS_ST = True
except Exception:
    _HAS_ST = False

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _HAS_HF = True
except Exception:
    _HAS_HF = False


class Reranker:
    """
    Reranks retrieved documents based on semantic relevance.

    Example:
        >>> reranker = Reranker()
        >>> reranked = reranker.rerank("what is diffusion model?", retrieved_docs)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if _HAS_HF and torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.backend = None

        # Prefer SentenceTransformers CrossEncoder
        if _HAS_ST:
            try:
                logger.info(f"Loading CrossEncoder model: {model_name}")
                self.model = CrossEncoder(model_name, device=self.device)
                self.backend = "sentence-transformers"
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformers CrossEncoder: {e}")

        # Fallback: Hugging Face SequenceClassification
        if self.model is None and _HAS_HF:
            try:
                logger.info(f"Loading Hugging Face cross-encoder: {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
                self.backend = "transformers"
            except Exception as e:
                logger.warning(f"Failed to load HF model: {e}")

        if self.model is None:
            logger.warning("No reranker model available. Reranking will be skipped.")

    # ------------------------------------------------------------
    # 🔁 Reranking
    # ------------------------------------------------------------
    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using semantic similarity between query and doc.
        Returns a sorted list of (doc, score).
        """
        if not docs or self.model is None:
            logger.warning("Reranker inactive or no docs to rerank.")
            return [(d, 0.0) for d in docs]

        logger.info(f"Reranking {len(docs)} candidates for query: '{query[:60]}...'")

        if self.backend == "sentence-transformers":
            pairs = [(query, d) for d in docs]
            scores = self.model.predict(pairs, show_progress_bar=False)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k] if top_k else ranked

        elif self.backend == "transformers":
            scores = []
            self.model.eval()
            with torch.no_grad():
                for d in docs:
                    inputs = self.tokenizer(
                        query,
                        d,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                        max_length=512,
                    ).to(self.device)
                    output = self.model(**inputs)
                    score = output.logits.squeeze().item()
                    scores.append(score)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return ranked[:top_k] if top_k else ranked

        else:
            logger.warning("No valid reranking backend available.")
            return [(d, 0.0) for d in docs]

    # ------------------------------------------------------------
    # ✅ Helper for integrated pipeline
    # ------------------------------------------------------------
    def rerank_results(
        self,
        query: str,
        retrieved: List[Tuple[str, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Takes output from Retriever.search() and reorders them.
        """
        docs = [r[0] for r in retrieved]
        reranked = self.rerank(query, docs, top_k=top_k)
        return reranked


# ------------------------------------------------------------
# 🔬 Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    docs = [
        "Diffusion models are a type of generative model.",
        "Transformers are used in NLP and vision.",
        "GANs generate data through adversarial learning.",
        "Diffusion models progressively denoise images.",
    ]
    query = "How do diffusion models work?"

    reranker = Reranker()
    results = reranker.rerank(query, docs, top_k=2)
    print("\nTop Reranked Results:")
    for d, s in results:
        print(f"{s:.3f} :: {d}")