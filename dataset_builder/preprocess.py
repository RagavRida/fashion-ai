#!/usr/bin/env python3
"""
dataset_builder/preprocess.py
-------------------------------
Preprocesses raw DeepFashion images into 512×512 training-ready crops.

Produces:
  - data/processed/images_512/*.jpg   (center-cropped, normalized)
  - data/processed/metadata_raw.jsonl (per-sample metadata)

Usage:
    python dataset_builder/preprocess.py \
        --config configs/dataset.yaml \
        --input_dir data/raw \
        --output_dir data/processed \
        --limit 10 --dry_run
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger
from PIL import Image, ImageOps
from tqdm import tqdm


# ─── Preprocessing Logic ─────────────────────────────────────────────────────
def load_image_safe(path: str) -> Image.Image | None:
    """Load image, returning None on failure."""
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def center_crop_and_resize(
    img: Image.Image, target_size: int = 512
) -> Image.Image:
    """
    Center crop to square, then resize to target_size.
    Preserves the garment region (center-biased for clothing).
    """
    w, h = img.size
    min_dim = min(w, h)

    # Adaptive crop: shift center up slightly (garment is typically middle-upper)
    left = (w - min_dim) // 2
    top = max(0, int((h - min_dim) * 0.35))  # slightly above center
    right = left + min_dim
    bottom = top + min_dim

    img = img.crop((left, top, right, bottom))
    img = img.resize((target_size, target_size), Image.LANCZOS)
    return img


def adaptive_crop_and_resize(
    img: Image.Image, target_size: int = 512
) -> Image.Image:
    """
    Resize while maintaining aspect, then pad to square.
    Better for full-body garment shots.
    """
    img = ImageOps.contain(img, (target_size, target_size), Image.LANCZOS)
    # Pad to exact square with white background
    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))
    paste_x = (target_size - img.size[0]) // 2
    paste_y = (target_size - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def is_valid_garment_image(img: Image.Image, min_size: int = 256) -> bool:
    """Filter out images that are too small or clearly not garments."""
    w, h = img.size
    if w < min_size or h < min_size:
        return False
    # Check that image is not mostly white/black (empty)
    arr = np.array(img.convert("L"))
    std = arr.std()
    if std < 8:  # Very flat image
        return False
    return True


def process_single_image(
    args_tuple: tuple,
) -> dict | None:
    """Process a single image. Designed for multiprocessing."""
    (
        entry,
        images_dir,
        target_size,
        crop_mode,
        min_size,
    ) = args_tuple

    img = load_image_safe(entry["image_path"])
    if img is None:
        return None

    if not is_valid_garment_image(img, min_size):
        return None

    if crop_mode == "center":
        processed = center_crop_and_resize(img, target_size)
    elif crop_mode == "adaptive":
        processed = adaptive_crop_and_resize(img, target_size)
    else:
        processed = center_crop_and_resize(img, target_size)

    # Derive output filename
    src_path = Path(entry["image_path"])
    out_filename = f"{src_path.stem}_{src_path.parent.name}.jpg"
    out_path = images_dir / out_filename

    processed.save(out_path, "JPEG", quality=95)

    result = {**entry, "image_path": str(out_path), "processed": True}
    return result


# ─── Main Processing Loop ────────────────────────────────────────────────────
def load_raw_entries(input_dir: Path) -> list[dict]:
    """Load raw metadata from download step."""
    entries = []

    # Try raw labels JSONL first
    for jsonl_file in input_dir.rglob("raw_labels.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

    if not entries:
        # Fall back: scan for all images
        logger.info("No raw_labels.jsonl found. Scanning for images...")
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            for img_path in input_dir.rglob(ext):
                entries.append({"image_path": str(img_path)})

    return entries


def preprocess(
    input_dir: Path,
    output_dir: Path,
    config: dict,
    limit: int | None = None,
    dry_run: bool = False,
    num_workers: int = 8,
) -> list[dict]:
    images_dir = output_dir / "images_512"
    images_dir.mkdir(parents=True, exist_ok=True)

    entries = load_raw_entries(input_dir)
    logger.info(f"Found {len(entries)} raw entries")

    if limit:
        entries = entries[:limit]
        logger.info(f"Limited to {limit} samples")

    if dry_run:
        logger.info(f"[DRY RUN] Would process {len(entries)} images → {images_dir}")
        return entries[:5]

    target_size = config["preprocessing"]["image_size"]
    crop_mode = config["preprocessing"]["crop_mode"]
    min_size = config.get("limits", {}).get("min_image_size", 256)

    process_args = [
        (entry, images_dir, target_size, crop_mode, min_size)
        for entry in entries
    ]

    processed_entries = []
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_single_image, arg): arg
            for arg in process_args
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing images"
        ):
            result = future.result()
            if result:
                processed_entries.append(result)
            else:
                failed += 1

    logger.info(f"Processed: {len(processed_entries)} | Failed/Skipped: {failed}")

    # Save intermediate metadata
    meta_file = output_dir / "metadata_raw.jsonl"
    with open(meta_file, "w") as f:
        for entry in processed_entries:
            f.write(json.dumps(entry) + "\n")
    logger.success(f"Saved metadata → {meta_file}")

    return processed_entries


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocess DeepFashion images")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--input_dir", default="data/raw")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    preprocess(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        config=config,
        limit=args.limit,
        dry_run=args.dry_run,
        num_workers=args.num_workers,
    )
    logger.success("Preprocessing complete!")


if __name__ == "__main__":
    main()
