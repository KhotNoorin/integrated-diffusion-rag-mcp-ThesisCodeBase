"""
tests/test_end_to_end.py
End-to-end pipeline integration test — verifies entire flow works.
"""

import pytest
from pipelines.multimodal_generation import MultimodalGenerationPipeline

@pytest.mark.integration
def test_full_pipeline_run(monkeypatch):
    pipe = MultimodalGenerationPipeline()

    # Mock image generation
    def fake_run(prompt):
        return {"images": ["mock.png"], "caption": f"Generated for: {prompt}"}

    monkeypatch.setattr(pipe, "run", fake_run)
    result = pipe.run("A scientist exploring the future of AI")

    assert isinstance(result, dict)
    assert "images" in result
    assert result["caption"].startswith("Generated for")
