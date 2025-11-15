"""
models/__init__.py

Unified entry point for all model components in the
"Integrating Diffusion Models with RAG and Multi-Constraint Prompting"
project.

Submodules:
  - diffusion     : Diffusion models (Stable Diffusion, ControlNet)
  - retrieval     : RAG embedding + retrieval modules
  - constraints   : Multi-Constraint Prompting (factuality, style, ethics, diversity)
  - fusion        : Cross-modal fusion mechanisms
  - multimodal_generator : Core integration pipeline
  - prompt_constructor   : Dynamic prompt builder
  - evaluator            : Unified multimodal evaluation
"""

from .diffusion import (
    BaseDiffusion,
    ControlNetAdapter,
    UNetConfig,
    DiffusionPipeline,
)

from .retrieval import (
    Embedder,
    Retriever,
    IndexBuilder,
    Reranker,
)

from .constraints import (
    ConstraintManager,
    FactualityChecker,
    StyleController,
    EthicalFilter,
    DiversityController,
)

from .fusion import (
    MultimodalFuser,
    AttentionBridge,
)

from .multimodal_generator import MultimodalGenerator
from .prompt_constructor import PromptConstructor
from .evaluator import Evaluator


__all__ = [
    # Diffusion
    "BaseDiffusion",
    "ControlNetAdapter",
    "UNetConfig",
    "DiffusionPipeline",

    # Retrieval
    "Embedder",
    "Retriever",
    "IndexBuilder",
    "Reranker",

    # Constraints
    "ConstraintManager",
    "FactualityChecker",
    "StyleController",
    "EthicalFilter",
    "DiversityController",

    # Fusion
    "MultimodalFuser",
    "AttentionBridge",

    # Core Components
    "MultimodalGenerator",
    "PromptConstructor",
    "Evaluator",
]
