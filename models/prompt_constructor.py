"""
models/prompt_constructor.py

Dynamically constructs and optimizes multimodal prompts by integrating:
  - User query
  - Retrieved RAG context
  - Multi-Constraint Prompting (MCP) directives
  - Style, ethical, factual, and diversity controls

Used by:
  - MultimodalGenerator
  - DiffusionPipeline
"""

from __future__ import annotations
from typing import List, Dict, Optional, Any
import random
from utils.logging_utils import get_logger
from utils.prompt_utils import clean_prompt

logger = get_logger("prompt_constructor")


class PromptConstructor:
    """
    Builds dynamic, context-aware prompts for multimodal generation.
    """

    def __init__(
        self,
        max_context_sentences: int = 3,
        constraint_format: str = "structured",  # options: structured | inline | natural
        style_templates: Optional[Dict[str, str]] = None,
    ):
        self.max_context_sentences = max_context_sentences
        self.constraint_format = constraint_format
        self.style_templates = style_templates or {
            "cinematic": "in a highly cinematic and atmospheric style",
            "realistic": "in a detailed, realistic manner with natural lighting",
            "cyberpunk": "in a neon-lit futuristic cyberpunk world",
            "scientific": "as a precise scientific visualization",
        }

        logger.info(f"PromptConstructor initialized (format={constraint_format}).")

    # ------------------------------------------------------------
    # 🧩 Main builder
    # ------------------------------------------------------------
    def build(
        self,
        user_query: str,
        retrieved_contexts: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Construct a structured multimodal prompt.

        Steps:
          1. Clean and normalize user query
          2. Append context (if available)
          3. Integrate constraint instructions (style, factual, ethical, diversity)
        """
        constraints = constraints or {}
        retrieved_contexts = retrieved_contexts or []

        query = clean_prompt(user_query)
        base_prompt = query.strip().capitalize()

        # 1️⃣ Add retrieved RAG context
        if retrieved_contexts:
            ctx = " ".join(retrieved_contexts[: self.max_context_sentences])
            base_prompt += f"\n\nContext (from retrieved knowledge): {ctx}"

        # 2️⃣ Add constraints
        constraint_section = self._format_constraints(constraints)
        prompt = f"{base_prompt}\n\n{constraint_section}"

        # 3️⃣ Final cleanup and return
        prompt = clean_prompt(prompt)
        logger.info(f"Prompt constructed successfully (len={len(prompt)} chars).")
        return prompt

    # ------------------------------------------------------------
    # 🧠 Helper: Format constraints
    # ------------------------------------------------------------
    def _format_constraints(self, constraints: Dict[str, Any]) -> str:
        """
        Format constraints into natural or structured language.
        """
        if self.constraint_format == "structured":
            lines = ["--- CONSTRAINTS ---"]
            for key, value in constraints.items():
                if isinstance(value, bool) and value:
                    lines.append(f"{key.capitalize()}: Enabled")
                elif isinstance(value, str):
                    lines.append(f"{key.capitalize()}: {value}")
            return "\n".join(lines)

        elif self.constraint_format == "inline":
            inline_parts = []
            for key, value in constraints.items():
                if isinstance(value, bool) and value:
                    inline_parts.append(f"{key}")
                elif isinstance(value, str):
                    inline_parts.append(f"{key}: {value}")
            return f"[Constraints: {', '.join(inline_parts)}]"

        else:  # natural
            natural_prompt = "Please ensure the generation follows these conditions: "
            parts = []
            for key, value in constraints.items():
                if isinstance(value, bool) and value:
                    parts.append(f"it is {key}")
                elif isinstance(value, str):
                    parts.append(f"it has a {value} style")
            return natural_prompt + ", ".join(parts) + "."

    # ------------------------------------------------------------
    # 🎨 Style helper
    # ------------------------------------------------------------
    def apply_style(self, prompt: str, style: Optional[str]) -> str:
        """
        Apply a style phrase from template dictionary.
        """
        if not style:
            return prompt

        style_phrase = self.style_templates.get(style.lower(), f"in {style} style")
        if style_phrase.lower() not in prompt.lower():
            prompt += f", {style_phrase}."
        return prompt

    # ------------------------------------------------------------
    # 🧩 Random constraint sampling (for augmentation)
    # ------------------------------------------------------------
    def randomize_constraints(self) -> Dict[str, Any]:
        """
        Randomly selects a subset of constraints for diversity during fine-tuning.
        """
        styles = list(self.style_templates.keys())
        random_style = random.choice(styles)
        c = {
            "style": random_style,
            "ethical": bool(random.getrandbits(1)),
            "factual": bool(random.getrandbits(1)),
            "diversity": bool(random.getrandbits(1)),
        }
        logger.debug(f"Sampled random constraints: {c}")
        return c


# ------------------------------------------------------------
# ✅ Example test
# ------------------------------------------------------------
if __name__ == "__main__":
    pc = PromptConstructor(constraint_format="structured")

    query = "A group of astronauts exploring an alien planet."
    retrieved = [
        "NASA conducted multiple Mars missions using rovers.",
        "Astronaut suits are designed for mobility and protection.",
    ]
    constraints = {
        "style": "cinematic",
        "factual": True,
        "ethical": True,
        "diversity": True,
    }

    prompt = pc.build(query, retrieved, constraints)
    print("\n=== Constructed Prompt ===\n")
    print(prompt)