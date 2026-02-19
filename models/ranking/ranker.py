#!/usr/bin/env python3
"""
models/ranking/ranker.py
--------------------------
Multi-metric ranking engine.

Scores candidates on:
  1. CLIPScore (prompt ↔ image semantic similarity)
  2. Mask Alignment (generated mask vs control mask IoU)
  3. Edge Alignment (edge correlation)
  4. Aesthetic Score (LAION aesthetic predictor proxy)
  5. LPIPS diversity (penalize near-copies)

Usage:
    from models.ranking.ranker import CandidateRanker

    ranker = CandidateRanker()
    ranked = ranker.rank(
        images=[pil_img1, pil_img2, ...],
        prompt="upcycled denim jacket, streetwear",
        control_mask=mask_np,          # optional
        control_edges=edges_np,        # optional
        top_k=4,
    )
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from PIL import Image


@dataclass
class RankResult:
    image: Image.Image
    index: int
    clip_score: float = 0.0
    mask_score: float = 0.0
    edge_score: float = 0.0
    aesthetic_score: float = 0.0
    lpips_penalty: float = 0.0
    total_score: float = 0.0
    scores: dict = field(default_factory=dict)


class CLIPScorer:
    """Compute cosine similarity between CLIP image and text embeddings."""

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai"):
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is not None:
            return
        import open_clip
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        logger.info(f"CLIP scorer loaded: {self.model_name} on {self.device}")

    def score(self, images: list[Image.Image], prompts: list[str]) -> list[float]:
        """Return CLIPScore for each (image, prompt) pair."""
        self._load()
        with torch.no_grad():
            # Encode text
            tokens = self._tokenizer(prompts).to(self.device)
            text_feats = self._model.encode_text(tokens)
            text_feats = F.normalize(text_feats, dim=-1)

            scores = []
            for img in images:
                img_tensor = self._preprocess(img).unsqueeze(0).to(self.device)
                img_feats = self._model.encode_image(img_tensor)
                img_feats = F.normalize(img_feats, dim=-1)
                sim = (img_feats * text_feats[0]).sum().item()
                # Scale to [0, 1]: raw cosine is [-1, 1]
                scores.append((sim + 1) / 2)

        return scores


class LPIPSScorer:
    """LPIPS perceptual distance scorer."""

    def __init__(self):
        self._loss_fn = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._loss_fn is not None:
            return
        import lpips
        self._loss_fn = lpips.LPIPS(net="alex").to(self.device)
        logger.info("LPIPS scorer loaded")

    def distance(self, img_a: Image.Image, img_b: Image.Image) -> float:
        """Return perceptual distance between two images (lower = more similar)."""
        self._load()
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        with torch.no_grad():
            a = transform(img_a).unsqueeze(0).to(self.device)
            b = transform(img_b).unsqueeze(0).to(self.device)
            dist = self._loss_fn(a, b).item()
        return dist


def compute_mask_alignment(gen_mask: np.ndarray, control_mask: np.ndarray) -> float:
    """IoU between generated garment mask and control mask."""
    if gen_mask is None or control_mask is None:
        return 0.5  # neutral

    gen_bin = (gen_mask > 127).astype(np.float32)
    ctrl_bin = (control_mask > 127).astype(np.float32)

    # Resize if needed
    if gen_bin.shape != ctrl_bin.shape:
        from PIL import Image
        gen_pil = Image.fromarray(gen_bin.astype(np.uint8) * 255)
        gen_pil = gen_pil.resize(
            (ctrl_bin.shape[1], ctrl_bin.shape[0]), Image.NEAREST
        )
        gen_bin = np.array(gen_pil).astype(np.float32) / 255.0

    intersection = (gen_bin * ctrl_bin).sum()
    union = np.clip(gen_bin + ctrl_bin, 0, 1).sum()
    return float(intersection / (union + 1e-8))


def compute_edge_alignment(gen_edges: np.ndarray, ctrl_edges: np.ndarray) -> float:
    """Pearson correlation between edge maps."""
    if gen_edges is None or ctrl_edges is None:
        return 0.5

    a = gen_edges.flatten().astype(np.float32)
    b = ctrl_edges.flatten().astype(np.float32)

    if a.shape != b.shape:
        from PIL import Image
        gen_pil = Image.fromarray(gen_edges.astype(np.uint8))
        gen_pil = gen_pil.resize(
            (ctrl_edges.shape[1], ctrl_edges.shape[0]), Image.BILINEAR
        )
        a = np.array(gen_pil).flatten().astype(np.float32)
        b = ctrl_edges.flatten().astype(np.float32)

    # Pearson correlation → scale to [0, 1]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.5
    corr = np.corrcoef(a, b)[0, 1]
    return float((corr + 1) / 2)


class CandidateRanker:
    """
    Multi-metric ranker for fashion generation candidates.

    Weights (configurable):
      - clip_weight:      0.40  (semantic prompt matching)
      - mask_weight:      0.25  (structure consistency)
      - edge_weight:      0.20  (edge alignment)
      - aesthetic_weight: 0.15  (visual quality)
    """

    def __init__(
        self,
        clip_weight: float = 0.40,
        mask_weight: float = 0.25,
        edge_weight: float = 0.20,
        aesthetic_weight: float = 0.15,
        lpips_lower_bound: float = 0.3,
        clip_model: str = "ViT-L-14",
    ):
        self.clip_weight = clip_weight
        self.mask_weight = mask_weight
        self.edge_weight = edge_weight
        self.aesthetic_weight = aesthetic_weight
        self.lpips_lower_bound = lpips_lower_bound

        self._clip_scorer = CLIPScorer(model_name=clip_model)
        self._lpips_scorer = LPIPSScorer()

    def _compute_aesthetic_score(self, image: Image.Image) -> float:
        """
        Simple aesthetic proxy: penalize images with extreme color distributions.
        Replace with LAION aesthetic predictor for production.
        """
        arr = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        # Std of brightness → boring if too low or too high
        brightness = arr.mean(axis=2)
        brightness_std = brightness.std()
        # Ideal std range: [0.1, 0.3]
        score = math.exp(-((brightness_std - 0.2) ** 2) / (2 * 0.05**2))
        return float(score)

    def _compute_lpips_penalty(
        self, image: Image.Image, reference: Optional[Image.Image]
    ) -> float:
        """Penalize if image is too close to reference (LPIPS < lower_bound)."""
        if reference is None:
            return 0.0
        try:
            dist = self._lpips_scorer.distance(image, reference)
            # Penalize if too similar (dist < lower_bound)
            if dist < self.lpips_lower_bound:
                return (self.lpips_lower_bound - dist) / self.lpips_lower_bound
            return 0.0
        except Exception:
            return 0.0

    def rank(
        self,
        images: list[Image.Image],
        prompt: str,
        control_mask: Optional[np.ndarray] = None,
        control_edges: Optional[np.ndarray] = None,
        reference_image: Optional[Image.Image] = None,
        segmentation_model=None,
        top_k: int = 4,
    ) -> list[RankResult]:
        """
        Rank candidates and return top_k.

        Args:
            images: List of generated PIL Images
            prompt: Text prompt used for generation
            control_mask: Original garment binary mask (H,W uint8)
            control_edges: Original edge map (H,W uint8)
            reference_image: Input garment image (for LPIPS diversity)
            segmentation_model: GarmentUNet for mask extraction from generations
            top_k: Number of top candidates to return

        Returns:
            List of RankResult sorted by total_score (descending)
        """
        if not images:
            return []

        logger.info(f"Ranking {len(images)} candidates (top_k={top_k})")

        # Compute CLIP scores
        prompts_batch = [prompt] * len(images)
        try:
            clip_scores = self._clip_scorer.score(images, prompts_batch)
        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}. Using neutral scores.")
            clip_scores = [0.5] * len(images)

        results = []
        for i, image in enumerate(images):
            # Aesthetic score
            aesthetic = self._compute_aesthetic_score(image)

            # LPIPS penalty vs reference
            lpips_penalty = self._compute_lpips_penalty(image, reference_image)

            # Mask alignment
            mask_score = 0.5
            if control_mask is not None and segmentation_model is not None:
                try:
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.Resize((512, 512)),
                        T.ToTensor(),
                        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ])
                    device = next(segmentation_model.parameters()).device
                    img_t = transform(image.convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        pred_mask = segmentation_model.predict_mask(img_t)
                    gen_mask_np = (
                        pred_mask.squeeze().cpu().numpy() * 255
                    ).astype(np.uint8)
                    mask_score = compute_mask_alignment(gen_mask_np, control_mask)
                except Exception as e:
                    logger.warning(f"Mask alignment failed: {e}")

            # Edge alignment
            edge_score = 0.5
            if control_edges is not None:
                try:
                    import cv2
                    gen_gray = np.array(image.convert("L"))
                    gen_edges = cv2.Canny(gen_gray, 100, 200)
                    edge_score = compute_edge_alignment(gen_edges, control_edges)
                except Exception as e:
                    logger.warning(f"Edge alignment failed: {e}")

            # Total score
            total = (
                self.clip_weight * clip_scores[i]
                + self.mask_weight * mask_score
                + self.edge_weight * edge_score
                + self.aesthetic_weight * aesthetic
                - 0.15 * lpips_penalty  # small penalty for being too close to input
            )

            result = RankResult(
                image=image,
                index=i,
                clip_score=round(clip_scores[i], 4),
                mask_score=round(mask_score, 4),
                edge_score=round(edge_score, 4),
                aesthetic_score=round(aesthetic, 4),
                lpips_penalty=round(lpips_penalty, 4),
                total_score=round(total, 4),
                scores={
                    "clip": clip_scores[i],
                    "mask": mask_score,
                    "edge": edge_score,
                    "aesthetic": aesthetic,
                    "lpips_penalty": lpips_penalty,
                    "total": total,
                },
            )
            results.append(result)

        # Sort descending
        results.sort(key=lambda r: r.total_score, reverse=True)
        top = results[:top_k]

        logger.info(
            f"Top {top_k} scores: {[r.total_score for r in top]}"
        )
        return top
