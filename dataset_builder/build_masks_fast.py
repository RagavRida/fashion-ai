#!/usr/bin/env python3
"""
dataset_builder/build_masks_fast.py
-------------------------------------
Fast pseudo-mask generation using Otsu thresholding + center-crop heuristic.
Replaces slow GrabCut (~14h) with near-instant processing (~2min for 42K images).

Strategy:
  1. Center-crop the middle 60% of the image (garments are typically centered)
  2. Otsu threshold on grayscale to separate garment from background
  3. Morphological cleanup (dilate+erode) to fill holes

Usage:
    python dataset_builder/build_masks_fast.py \
        --metadata data/processed/metadata_with_edges.jsonl \
        --output_masks data/processed/masks \
        --output_metadata data/processed/metadata_with_masks.jsonl \
        --num_workers 8
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm


def generate_fast_mask(image_path: str, masks_dir: Path) -> dict | None:
    """Generate a fast pseudo garment mask using Otsu + center bias."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Otsu threshold to separate garment from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Center crop mask — garments are in the middle 70% of the image
        center_mask = np.zeros_like(binary)
        y1 = int(h * 0.05)
        y2 = int(h * 0.95)
        x1 = int(w * 0.10)
        x2 = int(w * 0.90)
        center_mask[y1:y2, x1:x2] = 255

        # Combine: AND the otsu mask with center region
        combined = cv2.bitwise_and(binary, center_mask)

        # Morphological cleanup: close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

        # If mask is empty (white background), use center rectangle fallback
        if combined.sum() < (h * w * 0.05 * 255):
            combined = center_mask

        # Save mask
        src = Path(image_path)
        mask_filename = src.stem + "_mask.png"
        mask_path = masks_dir / mask_filename
        cv2.imwrite(str(mask_path), combined)

        return {"mask_path": str(mask_path)}

    except Exception as e:
        logger.warning(f"Failed {image_path}: {e}")
        return None


def build_masks(
    metadata_file: Path,
    masks_dir: Path,
    output_metadata: Path,
    num_workers: int = 8,
    limit: int | None = None,
) -> None:
    masks_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(metadata_file) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if limit:
        entries = entries[:limit]

    logger.info(f"Generating fast pseudo-masks for {len(entries)} images...")

    results = []
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(generate_fast_mask, e["image_path"], masks_dir): e
            for e in entries
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Masking"):
            entry = futures[future]
            mask_result = future.result()
            if mask_result:
                results.append({**entry, **mask_result})
            else:
                failed += 1

    logger.info(f"Masks generated: {len(results)} | Failed: {failed}")

    with open(output_metadata, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.success(f"Saved → {output_metadata}")


def main():
    parser = argparse.ArgumentParser(description="Fast pseudo-mask generation")
    parser.add_argument("--metadata", default="data/processed/metadata_with_edges.jsonl")
    parser.add_argument("--output_masks", default="data/processed/masks")
    parser.add_argument("--output_metadata", default="data/processed/metadata_with_masks.jsonl")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    build_masks(
        metadata_file=Path(args.metadata),
        masks_dir=Path(args.output_masks),
        output_metadata=Path(args.output_metadata),
        num_workers=args.num_workers,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
