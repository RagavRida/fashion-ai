#!/usr/bin/env python3
"""
dataset_builder/build_prompts.py
----------------------------------
Auto-generates text prompts from DeepFashion attribute labels.

Prompt format:
  "a {sustainability_tag} {color} {category}, {attributes}, {style_keyword} style, {quality_suffix}"

Example:
  "a upcycled blue denim jacket, long sleeves, pockets, streetwear style, high realism, detailed fabric texture"

Usage:
    python dataset_builder/build_prompts.py \
        --config configs/dataset.yaml \
        --metadata data/processed/metadata_raw.jsonl \
        --output data/processed/prompts \
        --output_jsonl data/processed/metadata_with_prompts.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm


# ─── Attribute → Prompt Rules ────────────────────────────────────────────────
CATEGORY_MAP = {
    "jacket": ["jacket", "outerwear", "coat"],
    "dress": ["dress", "gown", "frock"],
    "top": ["top", "blouse", "shirt", "t-shirt", "crop top"],
    "pants": ["pants", "trousers", "jeans"],
    "shorts": ["shorts"],
    "skirt": ["skirt", "mini skirt", "maxi skirt"],
    "sweater": ["sweater", "knitwear", "pullover"],
    "suit": ["suit", "blazer", "formal set"],
    "romper": ["romper", "jumpsuit", "playsuit"],
    "other": ["garment", "clothing item"],
}

COLOR_MAP = {
    "black": "black",
    "white": "white",
    "red": "red",
    "blue": "blue",
    "denim": "denim blue",
    "navy": "navy blue",
    "grey": "grey",
    "gray": "grey",
    "green": "green",
    "beige": "beige",
    "brown": "brown",
    "yellow": "yellow",
    "pink": "pink",
    "purple": "purple",
    "orange": "orange",
    "multi": "multicolored",
    "floral": "floral patterned",
    "stripe": "striped",
    "plaid": "plaid",
    "print": "printed",
}

ATTRIBUTE_MAP = {
    # Sleeves
    "sleeveless": "sleeveless",
    "short sleeve": "short sleeves",
    "long sleeve": "long sleeves",
    "3/4 sleeve": "3/4 length sleeves",
    # Neckline
    "v-neck": "v-neck",
    "crew neck": "crew neck",
    "turtleneck": "turtleneck",
    "off-shoulder": "off-shoulder",
    "halter": "halter neck",
    # Length
    "mini": "mini length",
    "midi": "midi length",
    "maxi": "maxi length",
    "cropped": "cropped",
    # Style
    "casual": None,   # handled by style_keyword
    "formal": None,
    "fitted": "fitted",
    "oversized": "oversized",
    "loose": "loose fit",
    "slim": "slim fit",
    # Fabric-related
    "denim": "denim fabric",
    "cotton": "cotton fabric",
    "leather": "leather",
    "silk": "silk",
    "knit": "knit fabric",
    "linen": "linen",
    # Details
    "pockets": "with pockets",
    "ruffles": "with ruffles",
    "buttons": "with buttons",
    "zipper": "with zipper",
    "belt": "with belt",
    "embroidery": "with embroidery",
    "lace": "with lace trim",
}

STYLE_KEYWORDS = [
    "streetwear", "formal", "casual", "bohemian", "athleisure",
    "vintage", "minimalist", "preppy", "grunge", "romantic",
    "business casual", "summer", "winter", "resort wear",
]

SUSTAINABILITY_TAGS = [
    "upcycled", "recycled", "eco-friendly", "sustainable",
    "repurposed", "thrifted and redesigned",
]

QUALITY_SUFFIX = (
    "high realism, professional fashion photography, "
    "detailed fabric texture, sharp focus, 8k resolution"
)


# ─── Prompt Generator ────────────────────────────────────────────────────────
class PromptGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.style_keywords = config.get("prompt_generation", {}).get(
            "style_keywords", STYLE_KEYWORDS
        )
        self.sustainability_tags = config.get("prompt_generation", {}).get(
            "sustainability_tags", SUSTAINABILITY_TAGS
        )
        self.quality_suffix = config.get("prompt_generation", {}).get(
            "quality_suffix", QUALITY_SUFFIX
        )

    def _extract_category(self, raw: dict) -> str:
        """Extract garment category from raw metadata."""
        for field in ["category", "label", "class", "type"]:
            if field in raw:
                val = str(raw[field]).lower()
                for cat, keywords in CATEGORY_MAP.items():
                    if any(kw in val for kw in keywords):
                        return cat
        return "other"

    def _extract_color(self, raw: dict) -> str:
        """Extract dominant color descriptor."""
        for field in ["color", "dominant_color", "attributes"]:
            if field in raw:
                val = str(raw[field]).lower()
                for key, mapped in COLOR_MAP.items():
                    if key in val:
                        return mapped
        return ""

    def _extract_attributes(self, raw: dict) -> list[str]:
        """Extract relevant garment attributes."""
        attrs = []
        for field in ["attributes", "description", "text"]:
            if field in raw:
                val = str(raw[field]).lower()
                for key, mapped in ATTRIBUTE_MAP.items():
                    if key in val and mapped is not None:
                        attrs.append(mapped)
        return list(dict.fromkeys(attrs))[:4]  # dedupe, limit to 4

    def generate(self, raw_entry: dict) -> str:
        """Generate a training prompt for a raw metadata entry."""
        category = self._extract_category(raw_entry)
        category_name = random.choice(CATEGORY_MAP.get(category, ["garment"]))
        color = self._extract_color(raw_entry)
        attributes = self._extract_attributes(raw_entry)

        sustainability = random.choice(self.sustainability_tags)
        style = random.choice(self.style_keywords)

        # Build prompt
        parts = [f"a {sustainability}"]
        if color:
            parts.append(color)
        parts.append(category_name)

        prompt = " ".join(parts)

        if attributes:
            prompt += ", " + ", ".join(attributes)

        prompt += f", {style} style"
        prompt += f", {self.quality_suffix}"

        # Clean up spacing
        prompt = re.sub(r"\s+", " ", prompt).strip()
        return prompt

    def generate_variations(self, raw_entry: dict, n: int = 3) -> list[str]:
        """Generate N diverse prompt variations for the same image."""
        return [self.generate(raw_entry) for _ in range(n)]


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Build training prompts")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument(
        "--metadata", default="data/processed/metadata_raw.jsonl"
    )
    parser.add_argument("--output", default="data/processed/prompts")
    parser.add_argument(
        "--output_jsonl",
        default="data/processed/metadata_with_prompts.jsonl",
    )
    parser.add_argument("--n_variations", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    prompts_dir = Path(args.output)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    generator = PromptGenerator(config)

    entries = []
    with open(args.metadata) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if args.limit:
        entries = entries[: args.limit]

    logger.info(f"Generating prompts for {len(entries)} samples...")

    updated_entries = []
    for entry in tqdm(entries, desc="Generating prompts"):
        if args.n_variations > 1:
            prompts = generator.generate_variations(entry, args.n_variations)
            entry["prompt"] = prompts[0]
            entry["prompt_variations"] = prompts
        else:
            entry["prompt"] = generator.generate(entry)

        # Save individual prompt file (for reference)
        img_id = Path(entry["image_path"]).stem
        prompt_file = prompts_dir / f"{img_id}.txt"
        with open(prompt_file, "w") as f:
            f.write(entry["prompt"])

        updated_entries.append(entry)

    # Save updated JSONL
    with open(args.output_jsonl, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    logger.success(
        f"Prompts generated: {len(updated_entries)} → {args.output_jsonl}"
    )
    logger.info(f"Example prompt: {updated_entries[0]['prompt'] if updated_entries else 'N/A'}")


if __name__ == "__main__":
    main()
