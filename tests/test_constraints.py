"""
tests/test_constraints.py
Tests constraint logic (factuality, style, ethical, diversity).
"""

import pytest
from models.constraints.constraint_manager import ConstraintManager

def test_constraint_manager_init():
    cm = ConstraintManager()
    assert hasattr(cm, "apply_constraints")

def test_factuality_checker():
    cm = ConstraintManager()
    result = cm.apply_constraints("The sun rises in the west.", constraints={"factual": True})
    assert isinstance(result, dict)
    assert "text" in result

def test_style_controller():
    cm = ConstraintManager()
    out = cm.apply_constraints("A city skyline at night", constraints={"style": "cinematic"})
    assert isinstance(out, dict)