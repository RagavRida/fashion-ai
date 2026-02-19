#!/usr/bin/env python3
"""
dataset_builder/build_masks.py
--------------------------------
Generates garment segmentation masks using SAM, GrabCut, or pseudo-labeling.

Outputs:
  - data/processed/masks/*.png  (binary garment masks: 255=garment, 0=background)

Usage:
    python dataset_builder/build_masks.py \
        --config configs/dataset.yaml \
        --metadata data/processed/metadata_with_edges.jsonl \
        --output_dir data/processed/masks \
        --method sam \
        --sam_checkpoint checkpoints/sam_vit_h_4b8939.pth
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml
from loguru import logger
from PIL import Image
from tqdm import tqdm


# ─── GrabCut Masking ─────────────────────────────────────────────────────────
def generate_grabcut_mask(img_path: str, output_path: str, iterations: int = 5) -> bool:
    """
    GrabCut-based garment segmentation.
    Uses center-crop assumption: garment is in the center of the frame.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        # Assume garment occupies the middle 60% of the frame
        margin_h = int(h * 0.1)
        margin_w = int(w * 0.1)
        rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

        # Extract foreground mask
        fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        cv2.imwrite(output_path, fg_mask)
        return True
    except Exception as e:
        logger.warning(f"GrabCut failed for {img_path}: {e}")
        return False


# ─── SAM Masking ─────────────────────────────────────────────────────────────
class SAMSegmenter:
    """SAM-based garment segmentation using center-point prompt."""

    def __init__(self, model_type: str = "vit_h", checkpoint: str = None):
        self._predictor = None
        self.model_type = model_type
        self.checkpoint = checkpoint

    def _load(self):
        if self._predictor is not None:
            return
        try:
            import torch
            from segment_anything import SamPredictor, sam_model_registry

            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device=device)
            self._predictor = SamPredictor(sam)
            logger.info(f"SAM loaded on {device}")
        except ImportError:
            raise RuntimeError(
                "Install segment-anything: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

    def segment(self, img_path: str, output_path: str) -> bool:
        """Segment garment using center-point prompt."""
        self._load()
        try:
            import numpy as np
            import cv2

            image = cv2.imread(img_path)
            if image is None:
                return False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            self._predictor.set_image(image_rgb)

            h, w = image.shape[:2]
            # Use multiple center-region points for better garment coverage
            input_points = np.array([
                [w // 2, h // 2],          # center
                [w // 2, int(h * 0.35)],   # upper center (torso)
                [w // 2, int(h * 0.65)],   # lower center (waist/hip)
            ])
            input_labels = np.array([1, 1, 1])  # all foreground

            masks, scores, _ = self._predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # Take the highest-scoring mask
            best_mask = masks[np.argmax(scores)]
            fg_mask = (best_mask * 255).astype(np.uint8)

            # Cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

            cv2.imwrite(output_path, fg_mask)
            return True
        except Exception as e:
            logger.warning(f"SAM failed for {img_path}: {e}")
            return False


# ─── Pseudo-mask (center ellipse heuristic) ───────────────────────────────────
def generate_pseudo_mask(img_path: str, output_path: str) -> bool:
    """
    Fallback: generate ellipse-shaped pseudo-mask centered in frame.
    Useful when SAM/GrabCut is unavailable.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.35), int(h * 0.45))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        cv2.imwrite(output_path, mask)
        return True
    except Exception as e:
        logger.warning(f"Pseudo-mask failed for {img_path}: {e}")
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate garment segmentation masks")
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument(
        "--metadata", default="data/processed/metadata_with_edges.jsonl"
    )
    parser.add_argument("--output_dir", default="data/processed/masks")
    parser.add_argument(
        "--output_jsonl", default="data/processed/metadata_with_masks.jsonl"
    )
    parser.add_argument(
        "--method", default="sam", choices=["sam", "grabcut", "pseudo"]
    )
    parser.add_argument(
        "--sam_checkpoint", default="checkpoints/sam_vit_h_4b8939.pth"
    )
    parser.add_argument("--sam_model_type", default="vit_h")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

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

    logger.info(f"Generating {args.method} masks for {len(entries)} images...")

    # SAM runs on GPU, use single-threaded with batching
    if args.method == "sam":
        segmenter = SAMSegmenter(args.sam_model_type, args.sam_checkpoint)
        mask_results = {}
        for entry in tqdm(entries, desc="SAM segmentation"):
            img_path = entry["image_path"]
            img_stem = Path(img_path).stem
            output_path = str(output_dir / f"{img_stem}_mask.png")
            if segmenter.segment(img_path, output_path):
                mask_results[img_path] = output_path
    else:
        # GrabCut and Pseudo can be parallelized
        def worker(entry):
            img_path = entry["image_path"]
            img_stem = Path(img_path).stem
            output_path = str(output_dir / f"{img_stem}_mask.png")
            if args.method == "grabcut":
                success = generate_grabcut_mask(img_path, output_path)
            else:
                success = generate_pseudo_mask(img_path, output_path)
            return img_path, output_path if success else None

        mask_results = {}
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(worker, e): e for e in entries}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Masking"):
                img_path, mask_path = future.result()
                if mask_path:
                    mask_results[img_path] = mask_path

    # Merge mask paths into entries
    updated_entries = []
    for entry in entries:
        if entry["image_path"] in mask_results:
            entry["mask_path"] = mask_results[entry["image_path"]]
            updated_entries.append(entry)

    with open(args.output_jsonl, "w") as f:
        for entry in updated_entries:
            f.write(json.dumps(entry) + "\n")

    logger.success(
        f"Masks generated: {len(updated_entries)} / {len(entries)} → {args.output_jsonl}"
    )


if __name__ == "__main__":
    main()
