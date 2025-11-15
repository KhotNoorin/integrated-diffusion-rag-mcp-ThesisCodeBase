"""
utils/prompt_utils.py

Prompt construction utilities for:
  - RAG + Diffusion + Multi-Constraint Prompting
  - Context injection (retrieved docs)
  - Constraint-aware prompt templates
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any
import textwrap
import random
import json


# ------------------------------------------------------------
# 🧩 Prompt Template System
# ------------------------------------------------------------

def format_prompt(
    user_query: str,
    retrieved_contexts: Optional[List[str]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    mode: str = "text",
) -> str:
    """
    Build a structured prompt depending on mode.
    Args:
        user_query: original user prompt
        retrieved_contexts: list of retrieved text chunks
        constraints: dictionary of applied constraints
        mode: one of ['text', 'image', 'multimodal']
    """
    context_section = ""
    if retrieved_contexts:
        joined_context = "\n".join(f"- {c.strip()}" for c in retrieved_contexts[:3])
        context_section = f"\n\n[Retrieved Context]\n{joined_context}"

    constraint_section = ""
    if constraints:
        formatted = json.dumps(constraints, indent=2)
        constraint_section = f"\n\n[Constraints]\n{formatted}"

    base_prompt = f"[User Query]\n{user_query}{context_section}{constraint_section}"

    if mode == "image":
        base_prompt += "\n\n[Task]\nGenerate a detailed image that aligns with the description and constraints."
    elif mode == "multimodal":
        base_prompt += "\n\n[Task]\nGenerate both a coherent caption and image consistent with the retrieved facts and style."
    else:
        base_prompt += "\n\n[Task]\nGenerate a factual, stylistically aligned text response."

    return textwrap.dedent(base_prompt).strip()


# ------------------------------------------------------------
# 🧠 Constraint Prompt Builder
# ------------------------------------------------------------

def apply_constraints_to_prompt(prompt: str, constraints: Dict[str, Any]) -> str:
    """
    Injects explicit constraint control tokens or phrases into a prompt.
    Example:
        constraints = {
            "style": "scientific",
            "ethics": "avoid bias",
            "factual": True
        }
    """
    control_texts = []
    if constraints.get("style"):
        control_texts.append(f"Style: {constraints['style']}")
    if constraints.get("ethics"):
        control_texts.append(f"Ethical Rule: {constraints['ethics']}")
    if constraints.get("factual", False):
        control_texts.append("Ensure factual correctness using retrieved context.")

    if control_texts:
        prompt += "\n\n[Control Directives]\n" + "\n".join(f"- {c}" for c in control_texts)

    return prompt


# ------------------------------------------------------------
# 🎨 Diffusion-Specific Prompting
# ------------------------------------------------------------

def diffusion_prompt_from_text(
    text_prompt: str,
    style_hint: Optional[str] = None,
    ethical_tag: Optional[str] = None,
) -> str:
    """
    Converts a text prompt into a style-guided diffusion prompt.
    """
    prompt = text_prompt
    if style_hint:
        prompt += f", in the style of {style_hint}"
    if ethical_tag:
        prompt += f" (ensure ethical representation: {ethical_tag})"
    return prompt


# ------------------------------------------------------------
# 🧩 Random Prompt Augmentation (Data Diversity)
# ------------------------------------------------------------

def augment_prompt_variations(prompt: str, num_variants: int = 3) -> List[str]:
    """
    Create small lexical variations for data augmentation.
    """
    variants = [prompt]
    synonyms = ["visualize", "illustrate", "depict", "show", "represent"]

    for _ in range(num_variants - 1):
        variant = prompt
        if "generate" in variant.lower():
            variant = variant.replace("generate", random.choice(synonyms))
        else:
            variant += f" ({random.choice(['high quality', 'realistic', 'conceptual', 'stylized'])})"
        variants.append(variant)

    return list(set(variants))


# ------------------------------------------------------------
# 🔄 Prompt Normalization
# ------------------------------------------------------------

def clean_prompt(prompt: str) -> str:
    """
    Remove redundant spaces, brackets, and normalize case.
    """
    prompt = " ".join(prompt.split())
    prompt = prompt.replace(" [", "[").replace("] ", "]")
    return prompt.strip()


# ------------------------------------------------------------
# 🧱 Unified Builder Interface
# ------------------------------------------------------------

class PromptBuilder:
    """
    Class to build multi-constraint, context-enriched prompts.
    """

    def __init__(self, mode: str = "multimodal"):
        self.mode = mode

    def build(
        self,
        query: str,
        retrieved_contexts: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        prompt = format_prompt(query, retrieved_contexts, constraints, mode=self.mode)
        if constraints:
            prompt = apply_constraints_to_prompt(prompt, constraints)
        return clean_prompt(prompt)

    def build_for_diffusion(
        self,
        query: str,
        style_hint: Optional[str] = None,
        ethical_tag: Optional[str] = None,
    ) -> str:
        return diffusion_prompt_from_text(query, style_hint, ethical_tag)


# ------------------------------------------------------------
# ✅ Example Test
# ------------------------------------------------------------

if __name__ == "__main__":
    query = "Generate an educational poster about renewable energy."
    contexts = [
        "Renewable energy sources include solar, wind, and hydroelectric power.",
        "They reduce carbon emissions and promote sustainability."
    ]
    constraints = {
        "style": "infographic",
        "ethics": "avoid misinformation",
        "factual": True
    }

    builder = PromptBuilder(mode="multimodal")
    prompt = builder.build(query, contexts, constraints)
    print("=== Constructed Prompt ===\n")
    print(prompt)

    # Example diffusion variation
    print("\n=== Diffusion Prompt ===\n")
    print(builder.build_for_diffusion(query, style_hint="flat design", ethical_tag="inclusive"))
