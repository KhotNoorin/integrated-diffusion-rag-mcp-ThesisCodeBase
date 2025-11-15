"""
tests/test_multimodal_pipeline.py
Tests the full multimodal generator integration (RAG + Diffusion + MCP).
"""

import pytest
from models.multimodal_generator import MultimodalGenerator
from models.retrieval.retriever import Retriever
from models.diffusion.base_diffusion import DiffusionModel
from models.constraints.constraint_manager import ConstraintManager

@pytest.fixture
def multimodal_generator(tmp_path):
    retriever = Retriever(index_path=tmp_path)
    diffusion = DiffusionModel(pretrained_name="mock-model")
    constraints = ConstraintManager()
    return MultimodalGenerator(retriever, diffusion, constraints)

def test_generator_initialization(multimodal_generator):
    assert hasattr(multimodal_generator, "generate")

def test_text_to_image_integration(monkeypatch, multimodal_generator):
    monkeypatch.setattr(multimodal_generator.diffusion, "generate", lambda p, **_: {"image": "mock.png"})
    output = multimodal_generator.generate("A futuristic robot", mode="image")
    assert "image" in output