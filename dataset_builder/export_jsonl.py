#!/usr/bin/env python3
"""
dataset_builder/export_jsonl.py
---------------------------------
Final step: merges all metadata sources into a clean metadata.jsonl
ready for diffusion model training.

Each entry contains:
  - image_path    : 512x512 processed image
  - prompt        : generated training prompt
  - category      : garment category
  - edge_path     : canny/hed edge map
  - mask_path     : binary garment mask
  - split         : train | val | test

Usage:
    python dataset_builder/export_jsonl.py \
        --metadata data/processed/metadata_with_masks.jsonl \
        --output data/processed/metadata.jsonl \
        --min_complete_fields image_path,prompt,edge_path,mask_path
"""

import argparse
import json
import random
from pathlib import Path

import yaml
from loguru import logger
from tqdm import tqdm


def validate_entry(entry: dict, required_fields: list[str]) -> bool:
    """Check all required fields exist and files are accessible."""
    for field in required_fields:
        if field not in entry or not entry[field]:
            return False
        # If field is a file path, verify it exists
        if field.endswith("_path"):
            if not Path(entry[field]).exists():
                return False
    return True


def assign_splits(
    entries: list[dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> list[dict]:
    """Assign train/val/test splits."""
    random.seed(seed)
    shuffled = entries.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    for i, entry in enumerate(shuffled):
        if i < n_train:
            entry["split"] = "train"
        elif i < n_train + n_val:
            entry["split"] = "val"
        else:
            entry["split"] = "test"

    return shuffled


def compute_statistics(entries: list[dict]) -> dict:
    """Compute dataset statistics."""
    from collections import Counter

    splits = Counter(e.get("split", "unknown") for e in entries)
    categories = Counter(e.get("category", "unknown") for e in entries)

    has_edges = sum(1 for e in entries if "edge_path" in e and e["edge_path"])
    has_masks = sum(1 for e in entries if "mask_path" in e and e["mask_path"])
    has_prompts = sum(1 for e in entries if "prompt" in e and e["prompt"])

    return {
        "total_samples": len(entries),
        "splits": dict(splits),
        "top_categories": dict(categories.most_common(10)),
        "has_edges": has_edges,
        "has_masks": has_masks,
        "has_prompts": has_prompts,
        "completeness_pct": round(min(has_edges, has_masks, has_prompts) / max(len(entries), 1) * 100, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Export final training metadata.jsonl")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument(
        "--metadata", default="data/processed/metadata_with_masks.jsonl"
    )
    parser.add_argument("--output", default="data/processed/metadata.jsonl")
    parser.add_argument(
        "--min_complete_fields",
        default="image_path,prompt,edge_path,mask_path",
        help="Comma-separated required fields",
    )
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    required_fields = [f.strip() for f in args.min_complete_fields.split(",")]
    logger.info(f"Required fields: {required_fields}")

    # Load all entries
    all_entries = []
    with open(args.metadata) as f:
        for line in f:
            line = line.strip()
            if line:
                all_entries.append(json.loads(line))

    logger.info(f"Total input entries: {len(all_entries)}")

    # Validate
    valid_entries = []
    skipped = 0
    for entry in tqdm(all_entries, desc="Validating entries"):
        if validate_entry(entry, required_fields):
            valid_entries.append(entry)
        else:
            skipped += 1

    logger.info(f"Valid: {len(valid_entries)} | Skipped: {skipped}")

    if args.limit:
        valid_entries = valid_entries[: args.limit]
        logger.info(f"Limited to {args.limit} samples")

    # Assign splits
    entries_with_splits = assign_splits(
        valid_entries,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # Compute and log statistics
    stats = compute_statistics(entries_with_splits)
    logger.info(f"Dataset statistics:\n{json.dumps(stats, indent=2)}")

    # Save to output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for entry in entries_with_splits:
            f.write(json.dumps(entry) + "\n")

    # Save statistics
    stats_path = output_path.parent / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.success(f"Exported {len(entries_with_splits)} samples → {output_path}")
    logger.success(f"Statistics saved → {stats_path}")
    logger.info(
        f"Training samples  : {stats['splits'].get('train', 0)}\n"
        f"Validation samples: {stats['splits'].get('val', 0)}\n"
        f"Test samples      : {stats['splits'].get('test', 0)}\n"
        f"Completeness      : {stats['completeness_pct']}%"
    )


if __name__ == "__main__":
    main()
