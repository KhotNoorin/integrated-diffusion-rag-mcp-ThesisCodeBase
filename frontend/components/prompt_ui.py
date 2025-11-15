# frontend/components/prompt_ui.py
from __future__ import annotations
import streamlit as st

def render_prompt_ui(default_prompt: str = "A cinematic portrait of a scientist in a futuristic lab...") -> str:
    """Render text input for prompt (no form wrapping)."""
    st.markdown("### Prompt")
    st.caption("Enter prompt (what do you want generated?)")

    prompt = st.text_area(
        "Prompt Input",
        value=default_prompt,
        height=120,
        key="prompt_text",
    )

    # Examples below input
    st.markdown("**💡 Example prompts:**")
    st.markdown("- Astronaut in space")
    st.markdown("- Ancient city on Mars")
    st.markdown("- Abstract art of data flow")

    return prompt
