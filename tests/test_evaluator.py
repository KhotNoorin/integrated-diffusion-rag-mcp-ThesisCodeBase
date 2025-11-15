"""
tests/test_evaluator.py
Validates evaluator metrics: BLEU, FID, CLIPScore, CSR.
"""

import pytest
from models.evaluator import Evaluator

@pytest.fixture
def evaluator():
    return Evaluator()

def test_bleu_score(evaluator):
    score = evaluator.compute_bleu("A cat sits on mat", "A cat is on the mat")
    assert 0 <= score <= 1

def test_clip_score(evaluator):
    score = evaluator.compute_clip_score("A dog in the park", "An image of a dog in grass")
    assert isinstance(score, float)

def test_fid_mock(evaluator):
    score = evaluator.compute_fid(["img1.png"], ["img2.png"])
    assert score >= 0
