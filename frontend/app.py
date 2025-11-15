"""
frontend/app.py

Streamlit front-end for interactive multimodal generation demo.

Features
--------
 - Prompt input + constraint controls
 - Calls pipelines (MultimodalGenerationPipeline / ImageGenerationPipeline)
 - Shows generated images and captions
 - Simple evaluation viewer hooks
 - Works with or without backend pipelines
"""

from __future__ import annotations

# --- Path setup ---
import os, sys
from pathlib import Path
import time
from typing import Optional, Dict, Any
import streamlit as st

# Ensure imports work when run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Local components ---
from frontend.components.prompt_ui import render_prompt_ui
from frontend.components.constraint_controls import render_constraint_controls
from frontend.components.image_display import show_image_grid
from frontend.components.evaluation_viewer import render_evaluation_viewer

# Try to import pipelines (graceful fallback)
try:
    from pipelines.realworld_demo_pipeline import RealWorldDemoPipeline
    _HAS_PIPELINE = True
except Exception:
    _HAS_PIPELINE = False

APP_DIR = Path(__file__).resolve().parent
CSS_PATH = APP_DIR / "static" / "css" / "style.css"

# --- Page setup ---
st.set_page_config(page_title="Multimodal Demo", layout="wide", page_icon="🤖")

if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Multimodal Generator")
st.markdown(
    "Interactive demo for **RAG + Diffusion + Multi-Constraint Prompting (MCP)**. "
    "Use the controls to craft prompts, set constraints, and generate outputs."
)

# --- Sidebar controls ---
st.sidebar.header("Quick Controls")
preset = st.sidebar.selectbox("Preset scenario", ["Custom", "Education", "Cinematic", "Art", "News"])
num_images = st.sidebar.slider("Number of images", 1, 4, 1)
height = st.sidebar.selectbox("Height", [256, 384, 512], index=2)
width = st.sidebar.selectbox("Width", [256, 384, 512], index=2)

# ================================================================
# Main prompt / constraints form (only one form in the whole app)
# ================================================================
with st.form(key="generation_form", clear_on_submit=False):
    user_prompt = render_prompt_ui()
    constraints = render_constraint_controls()
    submitted = st.form_submit_button("✨ Generate")

# ================================================================
# Generation logic
# ================================================================
if submitted:
    # Apply preset defaults
    if preset == "Education":
        user_prompt = user_prompt or "An infographic explaining climate change for high school students."
        constraints.setdefault("style", "scientific")
    elif preset == "Cinematic":
        constraints.setdefault("style", "cinematic")
    elif preset == "Art":
        constraints.setdefault("style", "abstract")

    st.info("Running generation... (this will call local pipelines if configured)")
    start = time.time()

    if _HAS_PIPELINE:
        pipeline = RealWorldDemoPipeline()
        try:
            result = pipeline.run_demo(
                scenario=user_prompt,
                constraints=constraints,
                style_hint=constraints.get("style"),
                num_images=num_images,
                height=height,
                width=width,
            )
            images_dir = result.get("output_dir")
            from PIL import Image
            images = []
            for fname in sorted(Path(images_dir).glob("demo_image_*.png")):
                try:
                    images.append(Image.open(fname))
                except Exception:
                    pass
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            images = []
    else:
        # Mock generation if backend not present
        from PIL import Image, ImageDraw
        images = []
        for i in range(num_images):
            img = Image.new("RGB", (width, height), color=(240, 240, 240))
            d = ImageDraw.Draw(img)
            d.text((10, 10), f"Mock image {i+1}\nPrompt:\n{user_prompt[:80]}...", fill=(0, 0, 0))
            images.append(img)

    elapsed = time.time() - start
    st.success(f"✅ Done — generated {len(images)} image(s) in {elapsed:.1f}s")

    show_image_grid(images, captions=[user_prompt] * len(images))
    render_evaluation_viewer(short=True)

st.markdown("---")
st.caption(
    "Frontend demo: replace mock outputs with real pipeline by installing project dependencies "
    "and running pipelines in the backend."
)