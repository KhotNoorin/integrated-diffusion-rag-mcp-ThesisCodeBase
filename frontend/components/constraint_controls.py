"""
frontend/components/constraint_controls.py

Renders constraint controls for MCP (factual, style, ethical, diversity).
Returns a dict of selected constraints.
"""

from __future__ import annotations
import streamlit as st
from typing import Dict, Any

def render_constraint_controls() -> Dict[str, Any]:
    st.subheader("Constraints (Multi-Constraint Prompting)")
    col1, col2 = st.columns(2)

    with col1:
        factual = st.checkbox("Factuality", value=True, help="Enforce factual grounding from retrieved evidence")
        ethical = st.checkbox("Ethical Filter", value=True, help="Avoid NSFW or biased content")
        diversity = st.checkbox("Promote Diversity", value=True, help="Encourage varied outputs")

    with col2:
        style = st.selectbox("Style", ["realistic", "cinematic", "photorealistic", "cyberpunk", "scientific", "abstract", None], index=0)
        style_strength = st.slider("Style strength", 0.0, 2.0, 1.0, step=0.1)
        factual_weight = st.slider("Factual weight", 0.0, 2.0, 1.0, step=0.1)

    constraints: Dict[str, Any] = {
        "factual": factual,
        "ethical": ethical,
        "diversity": diversity,
        "style": style,
        "style_strength": style_strength,
        "factual_weight": factual_weight,
    }
    return constraints