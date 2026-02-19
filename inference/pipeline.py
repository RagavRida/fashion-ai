#!/usr/bin/env python3
"""
inference/pipeline.py
----------------------
Master FashionPipeline class that orchestrates all inference modes:
  - generate:          prompt-only → N images
  - redesign:          image-only  → N variations
  - redesign_prompt:   image+prompt → N images
  - refine:            previous output + new instruction → updated image
  - style_transfer:    garment + style reference → redesigned garment

All modes run ranking to return top-K outputs.

Usage:
    from inference.pipeline import FashionPipeline
    pipe = FashionPipeline.from_config("configs/inference.yaml")

    # Prompt-only
    results = pipe.generate("upcycled denim jacket, streetwear style", n_images=4)
    
    # Image-based redesign
    results = pipe.redesign(garment_image, n_images=4)
    
    # Refinement
    results = pipe.refine(prev_image, "make sleeves shorter")
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from loguru import logger
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ranking.ranker import CandidateRanker


# ─── Pipeline ────────────────────────────────────────────────────────────────
class FashionPipeline:
    """
    Unified fashion inference pipeline.
    
    Models loaded on first use (lazy loading) to minimize VRAM usage.
    """

    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        # Lazy-loaded components
        self._pipe = None                # Main diffusion pipeline
        self._controlnet = None          # ControlNet model
        self._image_proj = None          # IP-Adapter projection
        self._seg_model = None           # Garment segmentation U-Net
        self._clip_processor = None      # CLIP processor for IP-Adapter
        self._image_encoder = None       # CLIP image encoder

        self.ranker = CandidateRanker(
            clip_weight=config["ranking"]["clip_weight"],
            mask_weight=config["ranking"]["mask_alignment_weight"],
            edge_weight=config["ranking"]["edge_alignment_weight"],
            aesthetic_weight=config["ranking"]["aesthetic_weight"],
        )
        self.gen_config = config["generation"]
        self.neg_prompt = config["generation"]["negative_prompt"]
        self.prompt_suffix = config["generation"]["prompt_suffix"]

    @classmethod
    def from_config(cls, config_path: str) -> "FashionPipeline":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _load_main_pipeline(self):
        """Lazy-load the main diffusion pipeline with LoRA and ControlNet."""
        if self._pipe is not None:
            return

        from diffusers import (
            AutoencoderKL,
            ControlNetModel,
            StableDiffusionXLControlNetPipeline,
            StableDiffusionXLPipeline,
        )
        models_config = self.config["models"]

        logger.info("Loading diffusion pipeline...")

        # Load ControlNet if available
        controlnet_path = models_config.get("controlnet_weights")
        if controlnet_path and Path(controlnet_path).exists():
            logger.info(f"Loading ControlNet from {controlnet_path}")
            self._controlnet = ControlNetModel.from_pretrained(
                controlnet_path, torch_dtype=self.dtype
            )
            self._pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                models_config["base_model"],
                controlnet=self._controlnet,
                vae=AutoencoderKL.from_pretrained(
                    models_config["vae_model"], torch_dtype=self.dtype
                ),
                torch_dtype=self.dtype,
            )
        else:
            logger.info("Loading SDXL pipeline (no ControlNet)")
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                models_config["base_model"], torch_dtype=self.dtype
            )

        # Load LoRA weights
        lora_path = models_config.get("lora_weights")
        if lora_path and Path(lora_path).exists():
            self._pipe.load_lora_weights(lora_path)
            self._pipe.fuse_lora(lora_scale=models_config.get("lora_scale", 0.85))
            logger.info(f"LoRA weights loaded (scale={models_config.get('lora_scale', 0.85)})")

        # Enable memory optimizations
        self._pipe = self._pipe.to(self.device)
        try:
            self._pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        self._pipe.enable_vae_slicing()

        logger.success("Diffusion pipeline loaded!")

    def _load_seg_model(self):
        """Lazy-load the garment segmentation U-Net."""
        if self._seg_model is not None:
            return

        seg_path = self.config["models"].get("segmentation_model")
        if seg_path and Path(seg_path).exists():
            from models.segmentation.unet import GarmentUNet
            self._seg_model = GarmentUNet.from_checkpoint(seg_path)
            self._seg_model = self._seg_model.to(self.device).eval()
            logger.info("Segmentation model loaded")
        else:
            logger.warning("Segmentation model not found, mask scoring disabled")

    def _extract_edges(self, image: Image.Image) -> np.ndarray:
        """Extract Canny edges from a PIL image."""
        import cv2
        arr = np.array(image.convert("L"))
        low = self.config["edge_extraction"]["canny_low"]
        high = self.config["edge_extraction"]["canny_high"]
        edges = cv2.Canny(arr, low, high)
        return edges  # np.ndarray, shape (H, W), uint8

    def _edges_to_pil(self, edges: np.ndarray) -> Image.Image:
        """Convert edge map array to 3-channel PIL for ControlNet input."""
        edges_3ch = np.stack([edges] * 3, axis=-1)
        return Image.fromarray(edges_3ch)

    def _get_garment_mask(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract garment mask using segmentation model."""
        self._load_seg_model()
        if self._seg_model is None:
            return None

        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        img_t = transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mask = self._seg_model.predict_mask(img_t)
        mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        return mask_np

    def _build_prompt(self, prompt: str) -> str:
        """Append quality suffix if not already present."""
        if self.prompt_suffix.split(",")[0].strip() in prompt:
            return prompt
        return f"{prompt}, {self.prompt_suffix}"

    def _generate_images(
        self,
        prompt: str,
        control_image: Optional[Image.Image] = None,
        init_image: Optional[Image.Image] = None,
        strength: float = 0.75,
        n_images: int = 8,
    ) -> list[Image.Image]:
        """Core image generation."""
        self._load_main_pipeline()

        full_prompt = self._build_prompt(prompt)
        negative_prompt = self.neg_prompt

        gen_kwargs = dict(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=n_images,
            num_inference_steps=self.gen_config["num_inference_steps"],
            guidance_scale=self.gen_config["guidance_scale"],
            height=self.gen_config["resolution"],
            width=self.gen_config["resolution"],
        )

        seed = self.gen_config.get("seed")
        if seed:
            gen_kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)

        # Add ControlNet conditioning if available
        if control_image is not None and self._controlnet is not None:
            gen_kwargs["image"] = control_image
            gen_kwargs["controlnet_conditioning_scale"] = self.config["models"]["controlnet_scale"]

        # img2img if init_image provided
        if init_image is not None:
            from diffusers import StableDiffusionXLImg2ImgPipeline
            img2img_pipe = StableDiffusionXLImg2ImgPipeline(**self._pipe.components)
            img2img_pipe = img2img_pipe.to(self.device)
            gen_kwargs["image"] = init_image
            gen_kwargs["strength"] = strength
            del gen_kwargs["height"]
            del gen_kwargs["width"]
            result = img2img_pipe(**gen_kwargs)
        else:
            result = self._pipe(**gen_kwargs)

        return result.images

    # ─── Public Generation Modes ─────────────────────────────────────────────
    def generate(
        self, prompt: str, n_images: int = None, top_k: int = None
    ) -> list[Image.Image]:
        """
        Mode A: Prompt-only fashion generation.
        Returns top_k ranked images.
        """
        n_gen = n_images or self.gen_config["num_images_per_prompt"]
        k = top_k or self.gen_config["top_k_return"]

        logger.info(f"[generate] prompt='{prompt[:80]}' n={n_gen}")
        images = self._generate_images(prompt, n_images=n_gen)

        ranked = self.ranker.rank(images, prompt, top_k=k)
        return [r.image for r in ranked]

    def redesign(
        self,
        garment_image: Image.Image,
        prompt: Optional[str] = None,
        n_images: int = None,
        top_k: int = None,
    ) -> list[Image.Image]:
        """
        Mode B: Image-based garment redesign.
        If no prompt given, generates a generic variation prompt.
        """
        n_gen = n_images or self.gen_config["num_images_per_prompt"]
        k = top_k or self.gen_config["top_k_return"]

        if not prompt:
            prompt = "redesigned garment, upcycled fashion, creative variation"

        logger.info(f"[redesign] prompt='{prompt[:60]}' n={n_gen}")

        # Extract edges for ControlNet
        edges = self._extract_edges(garment_image)
        edge_pil = self._edges_to_pil(edges)
        control_mask = self._get_garment_mask(garment_image)

        images = self._generate_images(
            prompt=prompt,
            control_image=edge_pil,
            init_image=garment_image,
            strength=self.gen_config["strength"],
            n_images=n_gen,
        )

        ranked = self.ranker.rank(
            images, prompt,
            control_mask=control_mask,
            control_edges=edges,
            reference_image=garment_image,
            segmentation_model=self._seg_model,
            top_k=k,
        )
        return [r.image for r in ranked]

    def refine(
        self,
        prev_image: Image.Image,
        refinement_prompt: str,
        original_prompt: str = "",
        n_images: int = 4,
        top_k: int = None,
    ) -> list[Image.Image]:
        """
        Mode C: Refinement via img2img with inpaint for localized edits.
        Parses refinement instruction to choose between global and local edit.
        """
        k = top_k or self.gen_config["top_k_return"]

        # Build combined prompt
        full_prompt = (
            f"{original_prompt}, {refinement_prompt}" if original_prompt
            else refinement_prompt
        )

        logger.info(f"[refine] instruction='{refinement_prompt[:60]}'")

        # Use lower strength for localized edits
        is_color_edit = any(
            word in refinement_prompt.lower()
            for word in ["color", "pastel", "darker", "lighter", "vibrant"]
        )
        is_structural = any(
            word in refinement_prompt.lower()
            for word in ["sleeve", "pocket", "collar", "shorter", "longer", "crop"]
        )

        strength = (
            self.gen_config["inpaint_strength"] if is_structural
            else self.gen_config["strength"]
        )

        edges = self._extract_edges(prev_image)
        edge_pil = self._edges_to_pil(edges)

        images = self._generate_images(
            prompt=full_prompt,
            control_image=edge_pil if self._controlnet else None,
            init_image=prev_image,
            strength=strength,
            n_images=n_images,
        )

        ranked = self.ranker.rank(
            images, full_prompt,
            control_edges=edges,
            reference_image=prev_image,
            top_k=k,
        )
        return [r.image for r in ranked]
