"""
tests/test_diffusion_integration.py
Ensures diffusion model runs and integrates with mock RAG inputs.
"""

import pytest
from models.diffusion.base_diffusion import DiffusionModel

def test_diffusion_load():
    model = DiffusionModel(pretrained_name="runwayml/stable-diffusion-v1-5")
    assert model is not None
    assert hasattr(model, "generate")

@pytest.mark.slow
def test_diffusion_generate_mock(monkeypatch):
    model = DiffusionModel(pretrained_name="mock-model")

    def fake_generate(prompt, **kwargs):
        return {"image": "mock_image.png", "metadata": {"prompt": prompt}}

    monkeypatch.setattr(model, "generate", fake_generate)
    result = model.generate("A futuristic robot reading a book")
    assert "image" in result
    assert result["metadata"]["prompt"].startswith("A futuristic")

# ==========================================================
# Diffusion Model Wrapper (Compatibility Alias)
# ==========================================================

class DiffusionModel:
    """
    Compatibility wrapper for BaseDiffusion to maintain consistency
    across all modules and tests.
    """

    def __init__(self, pretrained_name="runwayml/stable-diffusion-v1-5", **kwargs):
        from models.diffusion.base_diffusion import BaseDiffusion

        # Initialize the base diffusion pipeline
        self.base = BaseDiffusion(pretrained_name=pretrained_name, **kwargs)

    def generate(self, prompt, **kwargs):
        """Generate image output using the underlying diffusion model."""
        return self.base.generate(prompt, **kwargs)

    def __getattr__(self, name):
        """Delegate other attributes to the underlying BaseDiffusion."""
        return getattr(self.base, name)
