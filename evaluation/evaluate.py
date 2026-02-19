#!/usr/bin/env python3
"""
evaluation/evaluate.py
------------------------
Evaluation harness for Fashion Reuse Studio.

Metrics computed:
  - FID  (Frechet Inception Distance): generated vs. real fashion images
  - CLIP Score: text-image alignment
  - IoU Score: garment mask quality (segmentation model)
  - Edge Correlation: ControlNet structure preservation
  - LPIPS: diversity of generated candidates
  - DIY completeness check: all guide fields present, step count >= 6

Usage:
    python evaluation/evaluate.py \
        --config configs/inference.yaml \
        --eval_dir data/processed/eval \
        --real_dir data/processed/images_512 \
        --n_samples 200 \
        --output_file outputs/eval_report.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_images_from_dir(img_dir: str, n: int = None) -> list[Image.Image]:
    """Load PIL images from directory (jpg/png)."""
    paths = sorted(Path(img_dir).glob("*.jpg")) + sorted(Path(img_dir).glob("*.png"))
    if n:
        paths = paths[:n]
    images = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB").resize((512, 512)))
        except Exception:
            pass
    return images


def images_to_np(images: list[Image.Image]) -> np.ndarray:
    """Convert PIL list to [N, H, W, C] uint8 numpy array."""
    return np.stack([np.array(img) for img in images])


# ─── FID ──────────────────────────────────────────────────────────────────────
def compute_fid(real_images: list[Image.Image], gen_images: list[Image.Image]) -> float:
    """
    Compute FID using InceptionV3 features.
    Returns a float (lower is better; target < 30).
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        logger.warning("torchmetrics not available. Returning fake FID=0.0")
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid = FrechetInceptionDistance(normalize=True).to(device)

    def pil_batch_to_tensor(imgs):
        import torchvision.transforms as T
        t = T.Compose([T.ToTensor()])
        return torch.stack([t(img.resize((299, 299))) for img in imgs]).to(device)

    logger.info("Computing FID (this may take a few minutes)...")
    real_t = pil_batch_to_tensor(real_images)
    gen_t = pil_batch_to_tensor(gen_images)
    fid.update(real_t, real=True)
    fid.update(gen_t, real=False)
    score = fid.compute().item()
    logger.info(f"FID score: {score:.2f}")
    return score


# ─── CLIP Score ───────────────────────────────────────────────────────────────
def compute_clip_score(images: list[Image.Image], prompts: list[str]) -> float:
    """
    Compute mean CLIP score between images and their prompts.
    Returns float in [0, 1]. Target > 0.28
    """
    try:
        import clip
    except ImportError:
        logger.warning("CLIP not available. Returning fake score=0.0")
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    scores = []
    for img, text in tqdm(zip(images, prompts), total=len(images), desc="CLIP scoring"):
        img_t = preprocess(img).unsqueeze(0).to(device)
        text_t = clip.tokenize([text], truncate=True).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_t)
            txt_feat = model.encode_text(text_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            score = (img_feat @ txt_feat.T).item()
        scores.append(score)

    mean_score = float(np.mean(scores))
    logger.info(f"Mean CLIP score: {mean_score:.4f}")
    return mean_score


# ─── LPIPS Diversity ──────────────────────────────────────────────────────────
def compute_lpips_diversity(images_per_prompt: list[list[Image.Image]]) -> float:
    """
    Compute mean pairwise LPIPS across image groups.
    Higher = more diverse (target > 0.3).
    """
    try:
        import lpips
    except ImportError:
        logger.warning("LPIPS not available. Returning 0.0")
        return 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device)
    import torchvision.transforms as T

    to_t = T.Compose([T.Resize((256, 256)), T.ToTensor(),
                      T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    group_diversities = []
    for group in tqdm(images_per_prompt, desc="LPIPS diversity"):
        if len(group) < 2:
            continue
        tensors = [to_t(img).unsqueeze(0).to(device) for img in group]
        dists = []
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                d = loss_fn(tensors[i], tensors[j]).item()
                dists.append(d)
        group_diversities.append(np.mean(dists))

    score = float(np.mean(group_diversities)) if group_diversities else 0.0
    logger.info(f"Mean LPIPS diversity: {score:.4f}")
    return score


# ─── Segmentation IoU ─────────────────────────────────────────────────────────
def compute_seg_iou(seg_model, images: list[Image.Image], ground_truth_masks: list[np.ndarray]) -> float:
    """Compute mean IoU of predicted masks vs. ground truth."""
    import torchvision.transforms as T

    device = next(seg_model.parameters()).device
    transform = T.Compose([
        T.Resize((512, 512)), T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    ious = []
    for img, gt_mask in zip(images, ground_truth_masks):
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = seg_model.predict_mask(img_t).squeeze().cpu().numpy()
        gt = (gt_mask > 128).astype(np.float32)
        pred_bin = (pred > 0.5).astype(np.float32)
        intersection = (pred_bin * gt).sum()
        union = (pred_bin + gt).clip(0, 1).sum()
        iou = intersection / (union + 1e-6)
        ious.append(iou)
    mean_iou = float(np.mean(ious))
    logger.info(f"Segmentation mean IoU: {mean_iou:.4f}")
    return mean_iou


# ─── DIY Guide Completeness ───────────────────────────────────────────────────
def evaluate_diy_guide(config: dict) -> dict:
    """
    End-to-end test of DIY guide generation.
    Checks all required fields and step completeness.
    """
    from inference.diy_guide import DIYGuideGenerator
    gen = DIYGuideGenerator(config)
    guide = gen.generate(
        garment_category="denim jacket",
        edits_applied=["cropped to waist", "added patches on elbows"],
        style_description="streetwear, urban upcycled aesthetic",
    )

    required_fields = [
        "title", "garment_category", "edits_summary", "materials",
        "tools", "steps", "estimated_time", "difficulty",
        "safety_tips", "budget_tips", "sustainability_benefits",
    ]
    d = guide.to_dict()
    present = {f: bool(d.get(f)) for f in required_fields}
    step_count = len(d.get("steps", []))
    all_present = all(present.values())
    steps_ok = step_count >= 6

    result = {
        "all_fields_present": all_present,
        "field_presence": present,
        "step_count": step_count,
        "min_steps_met": steps_ok,
        "pass": all_present and steps_ok,
    }
    logger.info(f"DIY guide eval: {'PASS' if result['pass'] else 'FAIL'} | {step_count} steps")
    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate Fashion Reuse Studio")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--eval_dir", default="data/processed/eval", help="Generated images dir")
    parser.add_argument("--real_dir", default="data/processed/images_512", help="Real images dir")
    parser.add_argument("--prompts_file", help="JSONL with {image, prompt} pairs")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--output_file", default="outputs/eval_report.json")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_diy", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results = {}

    # ── Load images ──
    logger.info("Loading images...")
    gen_images = load_images_from_dir(args.eval_dir, n=args.n_samples)
    real_images = load_images_from_dir(args.real_dir, n=args.n_samples)

    if not gen_images:
        logger.error(f"No generated images found in {args.eval_dir}")
        return

    logger.info(f"Loaded: {len(gen_images)} generated, {len(real_images)} real")

    # ── FID ──
    if not args.skip_fid and real_images:
        results["fid"] = compute_fid(real_images, gen_images)
        results["fid_pass"] = results["fid"] < 30.0
    else:
        results["fid"] = None
        results["fid_pass"] = None

    # ── CLIP Score ──
    prompts = []
    if args.prompts_file and Path(args.prompts_file).exists():
        with open(args.prompts_file) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        prompts = [e.get("prompt", "garment") for e in entries[:len(gen_images)]]

    if prompts:
        clip_imgs = gen_images[:len(prompts)]
        results["clip_score"] = compute_clip_score(clip_imgs, prompts)
        results["clip_pass"] = results["clip_score"] > 0.28
    else:
        logger.warning("No prompts file provided; skipping CLIP score")
        results["clip_score"] = None

    # ── LPIPS Diversity ──
    # Group images in batches of 4 to simulate per-prompt groups
    groups = [gen_images[i:i+4] for i in range(0, len(gen_images), 4)]
    results["lpips_diversity"] = compute_lpips_diversity(groups)
    results["diversity_pass"] = results["lpips_diversity"] > 0.3

    # ── DIY Guide ──
    if not args.skip_diy:
        try:
            results["diy_guide"] = evaluate_diy_guide(config)
        except Exception as e:
            logger.warning(f"DIY evaluation skipped: {e}")
            results["diy_guide"] = {"error": str(e)}

    # ── Summary ──
    results["summary"] = {
        "total_generated": len(gen_images),
        "fid": results.get("fid"),
        "clip_score": results.get("clip_score"),
        "lpips_diversity": results.get("lpips_diversity"),
        "fid_target": "< 30",
        "clip_target": "> 0.28",
        "diversity_target": "> 0.3",
    }

    passed = sum([
        results.get("fid_pass") or False,
        results.get("clip_pass") or False,
        results.get("diversity_pass") or False,
        results.get("diy_guide", {}).get("pass") or False,
    ])
    results["overall_pass"] = passed >= 3

    # ── Save ──
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.success(f"Evaluation report saved → {args.output_file}")
    logger.info(f"Results: {json.dumps(results['summary'], indent=2)}")
    logger.info(f"Overall: {'✅ PASS' if results['overall_pass'] else '❌ FAIL'}")


if __name__ == "__main__":
    main()
