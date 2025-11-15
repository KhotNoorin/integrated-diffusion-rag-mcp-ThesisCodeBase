"""
frontend/components/image_display.py

Utilities to render generated images in grids with captions and download links.
"""

from __future__ import annotations
from typing import List, Optional
import streamlit as st
from PIL import Image
import io

def show_image_grid(images: List[Image.Image], captions: Optional[List[str]] = None, cols: int = 2):
    if not images:
        st.warning("No images to show.")
        return

    captions = captions or [""] * len(images)
    rows = (len(images) + cols - 1) // cols
    img_cols = st.columns(cols)
    for idx, img in enumerate(images):
        with img_cols[idx % cols]:
            st.image(img, caption=captions[idx], use_column_width=True)
            # download button
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(f"Download {idx+1}", data=buf, file_name=f"generated_{idx+1}.png", mime="image/png")