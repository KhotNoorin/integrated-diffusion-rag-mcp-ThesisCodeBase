"""
models/retrieval/__init__.py

Public API for the retrieval subpackage.

Provides unified imports for:
  - Embedder      : For text/image embedding generation
  - IndexBuilder  : For vector index creation and persistence
  - Retriever     : For dense or hybrid retrieval
  - Reranker      : For semantic reranking of retrieved results

Example:
    from models.retrieval import Retriever, Embedder, Reranker

    embedder = Embedder()
    retriever = Retriever(embedder=embedder)
    results = retriever.retrieve("How does diffusion work?")
"""

from .embedder import Embedder  # type: ignore
from .index_builder import IndexBuilder  # type: ignore
from .retriever import Retriever  # type: ignore
from .reranker import Reranker  # type: ignore

__all__ = [
    "Embedder",
    "IndexBuilder",
    "Retriever",
    "Reranker",
]
