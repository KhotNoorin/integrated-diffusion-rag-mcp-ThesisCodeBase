"""
evaluation/qualitative_examples.py

Generates qualitative visual comparisons:
  - Baseline vs Improved images (side-by-side)
  - Optional captions or factuality annotations
  - Supports multimodal or text-only runs
  - Saves images ready for thesis inclusion (e.g., Figure 5.3 in Results Chapter)
"""

from __future__ import annotations
import os
import json
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont

from utils.logging_utils import get_logger
from utils.visualization import save_image_grid

logger = get_logger("qualitative_examples")

RESULTS_DIR = "experiments/results"
OUTPUT_DIR = "evaluation/qualitative"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_experiment_outputs(exp_name: str) -> Dict:
    """
    Load summary JSON for given experiment.
    """
    summary_path = os.path.join(RESULTS_DIR, f"{exp_name}_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No summary found for {exp_name}")
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def create_side_by_side(
    baseline_images: List[Image.Image],
    improved_images: List[Image.Image],
    captions: List[str],
    exp_name: str,
    max_examples: int = 4,
):
    """
    Create side-by-side visual comparison between baseline and improved results.
    """
    logger.info(f"Creating qualitative comparisons for {exp_name}...")
    num_examples = min(max_examples, len(improved_images))

    for i in range(num_examples):
        base_img = baseline_images[i].convert("RGB")
        imp_img = improved_images[i].convert("RGB")

        width, height = base_img.width, base_img.height
        combined = Image.new("RGB", (width * 2 + 20, height + 80), color=(255, 255, 255))

        # Paste images side-by-side
        combined.paste(base_img, (0, 40))
        combined.paste(imp_img, (width + 20, 40))

        # Caption text
        draw = ImageDraw.Draw(combined)
        caption_text = captions[i] if i < len(captions) else "Generated sample"
        font = None
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), f"Baseline vs Improved | {exp_name}", fill=(0, 0, 0), font=font)
        draw.text((10, height + 50), caption_text, fill=(60, 60, 60), font=font)

        # Save result
        save_path = os.path.join(OUTPUT_DIR, f"{exp_name}_sample_{i+1}.png")
        combined.save(save_path)
        logger.info(f"✅ Saved: {save_path}")


def generate_examples(exp_names: List[str]):
    """
    Generates qualitative examples for one or more experiments.
    """
    for exp_name in exp_names:
        try:
            data = load_experiment_outputs(exp_name)
            improved_runs = data.get("improved_runs", [])
            baseline_runs = data.get("baseline_runs", [])
            if not improved_runs:
                logger.warning(f"No improved runs for {exp_name}")
                continue

            # Extract example image sets
            base_imgs, imp_imgs, captions = [], [], []
            for idx, (b, i) in enumerate(zip(baseline_runs, improved_runs)):
                b_imgs = b.get("outputs", {}).get("images", [])
                i_imgs = i.get("outputs", {}).get("images", [])
                if not i_imgs or not isinstance(i_imgs[0], str):
                    continue  # skip if images are not paths
                try:
                    base_imgs.append(Image.open(b_imgs[0]))
                    imp_imgs.append(Image.open(i_imgs[0]))
                    captions.append(i.get("outputs", {}).get("captions", [""])[0])
                except Exception as e:
                    logger.warning(f"Failed to load image pair {idx} for {exp_name}: {e}")

            if base_imgs and imp_imgs:
                create_side_by_side(base_imgs, imp_imgs, captions, exp_name)
            else:
                logger.warning(f"No valid images found for {exp_name}")

        except Exception as e:
            logger.exception(f"Error generating examples for {exp_name}: {e}")


if __name__ == "__main__":
    # You can specify which experiments to visualize
    experiments = [
        "ablation_diffusion",
        "full_model_integration",
        "constraint_weight_sweep"
    ]
    generate_examples(experiments)