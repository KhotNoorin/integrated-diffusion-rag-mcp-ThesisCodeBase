"""
utils/visualization.py

Visualization utilities for:
  - Image and text comparison grids
  - Metric trend plots
  - Streamlit integration for multimodal demo

Dependencies:
  - matplotlib
  - Pillow
  - numpy
  - streamlit (optional)
"""

import os
import io
import math
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
import matplotlib.pyplot as plt

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


# ------------------------------------------------------------
# 🖼️  Image and Text Comparison Utilities
# ------------------------------------------------------------

def show_comparison(
    prompts: List[str],
    models: List[str],
    results: Dict[str, List[Dict]],
    save_path: Optional[str] = None,
):
    """
    Display side-by-side image/text comparisons for multiple models.

    Args:
        prompts: list of input prompts
        models: list of model names (order defines columns)
        results: dict mapping model_name -> list of outputs
                 each output: {"image": np.ndarray or PIL.Image, "caption": str, "score": float}
        save_path: optional path to save matplotlib figure
    """
    num_prompts = len(prompts)
    num_models = len(models)

    fig, axes = plt.subplots(num_prompts, num_models, figsize=(4 * num_models, 4 * num_prompts))

    if num_prompts == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_models == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, prompt in enumerate(prompts):
        for j, model in enumerate(models):
            ax = axes[i, j]
            output = results.get(model, [{}])[i] if model in results else {}

            img = output.get("image")
            caption = output.get("caption", "N/A")
            score = output.get("score", None)

            if img is None:
                ax.text(0.5, 0.5, "No Image", ha="center", va="center")
            else:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                ax.imshow(img)
            
            ax.axis("off")
            title = f"{model}"
            if score is not None:
                title += f"\nScore: {score:.2f}"
            ax.set_title(title, fontsize=10)

        # left margin prompt
        fig.text(
            0.02,
            (num_prompts - i - 0.5) / num_prompts,
            f"Prompt:\n{prompt}",
            va="center",
            ha="left",
            fontsize=9,
            wrap=True,
        )

    plt.tight_layout(rect=[0.05, 0, 1, 1])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


# ------------------------------------------------------------
# 📊 Metric Trend Plotting
# ------------------------------------------------------------

def plot_metric_trends(metrics_dict: Dict[str, List[float]], title: str = "Metric Trends", save_path: Optional[str] = None):
    """
    Plot multiple metrics over epochs or experiments.
    Args:
        metrics_dict: {"metric_name": [values]}
    """
    plt.figure(figsize=(7, 5))
    for metric, values in metrics_dict.items():
        plt.plot(range(1, len(values) + 1), values, marker="o", label=metric)

    plt.title(title)
    plt.xlabel("Step / Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


# ------------------------------------------------------------
# 🧱 Streamlit Components (Optional)
# ------------------------------------------------------------

def streamlit_comparison(prompts: List[str], models: List[str], results: Dict[str, List[Dict]]):
    """
    Display multimodal comparison interactively in Streamlit.
    """
    if not _HAS_STREAMLIT:
        raise ImportError("Streamlit is not installed. Run `pip install streamlit` to use this feature.")

    st.header("🧩 Multimodal Output Comparison")
    for i, prompt in enumerate(prompts):
        st.subheader(f"Prompt {i+1}: {prompt}")

        cols = st.columns(len(models))
        for j, model in enumerate(models):
            with cols[j]:
                output = results.get(model, [{}])[i] if model in results else {}
                img = output.get("image")
                caption = output.get("caption", "")
                score = output.get("score", None)

                st.markdown(f"**{model}**")
                if img is not None:
                    if isinstance(img, np.ndarray):
                        img = Image.fromarray(img)
                    st.image(img, caption=caption, use_column_width=True)
                else:
                    st.warning("No image available.")

                if score is not None:
                    st.caption(f"Score: {score:.2f}")


# ------------------------------------------------------------
# 🧩 Utility Helpers
# ------------------------------------------------------------

def fig_to_image(fig) -> Image.Image:
    """
    Convert matplotlib figure to PIL Image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)


def grid_of_images(images: List[Image.Image], ncols: int = 4, save_path: Optional[str] = None) -> Image.Image:
    """
    Combine a list of PIL images into a grid.
    """
    if not images:
        raise ValueError("No images provided for grid.")

    nrows = math.ceil(len(images) / ncols)
    w, h = images[0].size
    grid = Image.new("RGB", (ncols * w, nrows * h), color=(255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // ncols
        col = idx % ncols
        grid.paste(img, (col * w, row * h))

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        grid.save(save_path)

    return grid


# ------------------------------------------------------------
# 🧩 Compatibility Helper — save_image_grid()
# ------------------------------------------------------------

def save_image_grid(
    images: List[Image.Image],
    captions: Optional[List[str]] = None,
    ncols: int = 2,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 8),
):
    """
    Backward-compatible utility for saving a simple image grid.
    Used by evaluation/qualitative_examples.py
    """
    import matplotlib.pyplot as plt

    if not images:
        raise ValueError("No images provided to save_image_grid")

    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            ax.imshow(img)
            ax.axis("off")
            if captions and i < len(captions):
                ax.set_title(captions[i], fontsize=10)
        else:
            ax.axis("off")

    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200)
        plt.close(fig)
        print(f"✅ Saved image grid to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # quick manual test
    dummy_img = Image.new("RGB", (256, 256), color=(200, 180, 255))
    show_comparison(
        prompts=["A cat riding a bicycle", "A city skyline at night"],
        models=["Baseline", "Proposed"],
        results={
            "Baseline": [{"image": dummy_img, "caption": "Generic cat", "score": 0.65}] * 2,
            "Proposed": [{"image": dummy_img, "caption": "Realistic cat on bike", "score": 0.89}] * 2,
        },
    )