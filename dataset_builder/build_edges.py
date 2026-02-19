#!/usr/bin/env python3
"""
dataset_builder/build_edges.py
--------------------------------
Extracts edge maps from processed garment images using Canny or HED.

Outputs:
  - data/processed/edges/*.png  (single-channel edge maps, white edges on black)

Usage:
    python dataset_builder/build_edges.py \
        --config configs/dataset.yaml \
        --metadata data/processed/metadata_with_prompts.jsonl \
        --output_dir data/processed/edges \
        --method canny \
        --num_workers 8
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger
from PIL import Image
from tqdm import tqdm


# ─── Canny Edge Extraction ────────────────────────────────────────────────────
def extract_canny_edges(
    img_path: str,
    output_path: str,
    low_threshold: int = 100,
    high_threshold: int = 200,
    blur_kernel: int = 3,
) -> bool:
    """
    Extract Canny edges from an image.
    
    Args:
        img_path: Input image path
        output_path: Where to save edge map
        low_threshold: Canny lower hysteresis threshold
        high_threshold: Canny upper hysteresis threshold
        blur_kernel: Gaussian blur before Canny (odd number)
    
    Returns:
        True on success, False on failure
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Optional Gaussian blur to reduce noise
        if blur_kernel > 1:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Optional: dilate edges slightly for ControlNet conditioning
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        cv2.imwrite(output_path, edges)
        return True
    except Exception as e:
        logger.warning(f"Canny failed for {img_path}: {e}")
        return False


# ─── HED (Holistically-nested Edge Detection) ─────────────────────────────────
def extract_hed_edges(img_path: str, output_path: str) -> bool:
    """
    Extract HED edges using controlnet-aux HEDdetector.
    Produces softer, more semantic edges than Canny.
    """
    try:
        from controlnet_aux import HEDdetector

        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        image = Image.open(img_path).convert("RGB")
        edge_img = hed(image, detect_resolution=512, image_resolution=512)
        edge_img.save(output_path)
        return True
    except Exception as e:
        logger.warning(f"HED failed for {img_path}: {e}")
        return False


# ─── Multi-method Edge Extraction ────────────────────────────────────────────
def extract_combined_edges(img_path: str, output_path: str, config: dict) -> bool:
    """
    Extract both Canny and HED, blend them.
    Produces richer edges for complex garments.
    """
    try:
        # Canny
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_blur = cv2.GaussianBlur(img, (3, 3), 0)
        canny = cv2.Canny(img_blur, 80, 180)

        # Try HED if available
        try:
            from controlnet_aux import HEDdetector
            hed_detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
            pil_img = Image.open(img_path).convert("RGB")
            hed_img = hed_detector(pil_img, detect_resolution=512, image_resolution=512)
            hed_arr = np.array(hed_img.convert("L"))
            combined = cv2.addWeighted(canny, 0.5, hed_arr, 0.5, 0)
        except Exception:
            combined = canny

        cv2.imwrite(output_path, combined)
        return True
    except Exception as e:
        logger.warning(f"Combined edges failed for {img_path}: {e}")
        return False


# ─── Worker Function ─────────────────────────────────────────────────────────
def process_worker(args_tuple: tuple) -> dict | None:
    img_path, output_path, method, edge_config = args_tuple
    
    output_path_str = str(output_path)
    
    if method == "canny":
        success = extract_canny_edges(
            img_path,
            output_path_str,
            low_threshold=edge_config.get("canny_low", 100),
            high_threshold=edge_config.get("canny_high", 200),
            blur_kernel=edge_config.get("blur_kernel", 3),
        )
    elif method == "hed":
        success = extract_hed_edges(img_path, output_path_str)
    elif method == "combined":
        success = extract_combined_edges(img_path, output_path_str, edge_config)
    else:
        success = extract_canny_edges(img_path, output_path_str)

    if success:
        return {"image_path": img_path, "edge_path": str(output_path)}
    return None


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract edge maps from garment images")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument(
        "--metadata", default="data/processed/metadata_with_prompts.jsonl"
    )
    parser.add_argument("--output_dir", default="data/processed/edges")
    parser.add_argument(
        "--output_jsonl",
        default="data/processed/metadata_with_edges.jsonl",
    )
    parser.add_argument("--method", default="canny", choices=["canny", "hed", "combined"])
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    edge_config = config.get("edge_extraction", {})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.metadata) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    if args.limit:
        entries = entries[: args.limit]

    logger.info(
        f"Extracting {args.method} edges for {len(entries)} images using {args.num_workers} workers..."
    )

    process_args = []
    for entry in entries:
        img_path = entry["image_path"]
        img_stem = Path(img_path).stem
        output_path = output_dir / f"{img_stem}_edges.png"
        process_args.append((img_path, output_path, args.method, edge_config))

    edge_results = {}
    failed = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_worker, arg): arg
            for arg in process_args
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting edges"
        ):
            result = future.result()
            if result:
                edge_results[result["image_path"]] = result["edge_path"]
            else:
                failed += 1

    # Merge edge paths back into entries
    updated_entries = []
    for entry in entries:
        if entry["image_path"] in edge_results:
            entry["edge_path"] = edge_results[entry["image_path"]]
            updated_entries.append(entry)

    with open(args.output_jsonl, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    logger.success(
        f"Edges extracted: {len(updated_entries)} | Failed: {failed}"
    )
    logger.success(f"Saved → {args.output_jsonl}")


if __name__ == "__main__":
    main()
