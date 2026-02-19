#!/usr/bin/env python3
"""
dataset_builder/download_deepfashion.py
----------------------------------------
Downloads and organizes DeepFashion dataset subsets.

Usage:
    python dataset_builder/download_deepfashion.py \
        --subset category_attribute \
        --output_dir data/raw \
        --verify

Note: DeepFashion requires registration at:
    http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

This script supports:
  1) Official download with credentials
  2) HuggingFace mirror (fashion-gen / deepfashion-multimodal)
  3) Kaggle API download
"""

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import requests
from loguru import logger
from tqdm import tqdm


# ─── Constants ───────────────────────────────────────────────────────────────
DEEPFASHION_SUBSETS = {
    "category_attribute": {
        "description": "Category and Attribute Prediction benchmark",
        "huggingface_id": "detection-datasets/deepfashion-multimodal",
        "kaggle_dataset": "paramaggarwal/fashion-product-images-dataset",
        "expected_images": 289222,
    },
    "in_shop": {
        "description": "In-Shop Clothes Retrieval benchmark",
        "huggingface_id": "Marqo/deepfashion-multimodal",
        "kaggle_dataset": None,
        "expected_images": 52712,
    },
    "fashion_gen": {
        "description": "FashionGen (HuggingFace hosted)",
        "huggingface_id": "rozar/FashionGen",
        "kaggle_dataset": None,
        "expected_images": 325000,
    },
}

HF_TOKEN_ENV = "HF_TOKEN"
KAGGLE_JSON_PATH = Path.home() / ".kaggle" / "kaggle.json"


# ─── Helpers ─────────────────────────────────────────────────────────────────
def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify file integrity via MD5 checksum."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest() == expected_md5


def download_file(url: str, dest: Path, desc: str = "Downloading") -> Path:
    """Stream-download a file with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    return dest


def extract_zip(zip_path: Path, dest: Path) -> None:
    """Extract zip archive with progress."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        for member in tqdm(members, desc=f"Extracting {zip_path.name}"):
            zf.extract(member, dest)


# ─── Download Strategies ─────────────────────────────────────────────────────
def download_from_huggingface(subset_info: dict, output_dir: Path) -> None:
    """Download from HuggingFace datasets hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Install: pip install datasets")

    hf_id = subset_info["huggingface_id"]
    token = os.getenv(HF_TOKEN_ENV)
    logger.info(f"Downloading {hf_id} from HuggingFace...")

    dataset = load_dataset(hf_id, token=token, cache_dir=str(output_dir / "hf_cache"))

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    labels_data = []
    for split in dataset.keys():
        logger.info(f"Processing split: {split} ({len(dataset[split])} samples)")
        for i, sample in enumerate(
            tqdm(dataset[split], desc=f"Saving {split}")
        ):
            img_path = images_dir / f"{split}_{i:06d}.jpg"
            if "image" in sample and sample["image"] is not None:
                sample["image"].save(img_path)
            elif "img" in sample and sample["img"] is not None:
                sample["img"].save(img_path)

            entry = {
                "image_path": str(img_path),
                "split": split,
                "index": i,
            }
            # Preserve any text/label fields
            for key in ["text", "description", "label", "category", "attributes"]:
                if key in sample:
                    entry[key] = sample[key]
            labels_data.append(entry)

    # Save raw labels
    labels_file = output_dir / "raw_labels.jsonl"
    with open(labels_file, "w") as f:
        for entry in labels_data:
            f.write(json.dumps(entry) + "\n")

    logger.success(
        f"Downloaded {len(labels_data)} samples → {output_dir}"
    )


def download_from_kaggle(dataset_slug: str, output_dir: Path) -> None:
    """Download from Kaggle using kaggle CLI."""
    import subprocess

    if not KAGGLE_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Kaggle credentials not found at {KAGGLE_JSON_PATH}. "
            "Get credentials at https://www.kaggle.com/settings/account"
        )

    logger.info(f"Downloading {dataset_slug} from Kaggle...")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", dataset_slug,
            "-p", str(output_dir),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle download failed: {result.stderr}")
    logger.success(f"Kaggle download complete → {output_dir}")


def verify_download(output_dir: Path, expected_images: int) -> dict:
    """Verify downloaded data integrity."""
    images = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.png"))
    labels_file = output_dir / "raw_labels.jsonl"

    result = {
        "images_found": len(images),
        "expected_images": expected_images,
        "labels_exist": labels_file.exists(),
        "coverage_pct": round(len(images) / max(expected_images, 1) * 100, 1),
    }
    logger.info(f"Verification: {json.dumps(result, indent=2)}")
    return result


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download DeepFashion dataset subsets"
    )
    parser.add_argument(
        "--subset",
        default="category_attribute",
        choices=list(DEEPFASHION_SUBSETS.keys()),
        help="Dataset subset to download",
    )
    parser.add_argument(
        "--output_dir",
        default="data/raw",
        help="Output directory for raw data",
    )
    parser.add_argument(
        "--source",
        default="huggingface",
        choices=["huggingface", "kaggle"],
        help="Download source",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded data after download",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print plan without downloading",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / args.subset
    subset_info = DEEPFASHION_SUBSETS[args.subset]

    logger.info(f"=== DeepFashion Downloader ===")
    logger.info(f"Subset  : {args.subset} — {subset_info['description']}")
    logger.info(f"Source  : {args.source}")
    logger.info(f"Output  : {output_dir}")

    if args.dry_run:
        logger.info("[DRY RUN] No files will be downloaded.")
        return

    if args.source == "huggingface":
        download_from_huggingface(subset_info, output_dir)
    elif args.source == "kaggle":
        if not subset_info["kaggle_dataset"]:
            raise ValueError(f"No Kaggle dataset for subset: {args.subset}")
        download_from_kaggle(subset_info["kaggle_dataset"], output_dir)

    if args.verify:
        verify_download(output_dir, subset_info["expected_images"])

    logger.success("Download complete!")


if __name__ == "__main__":
    main()
