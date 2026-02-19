#!/usr/bin/env python3
"""
scripts/test_inference.py
─────────────────────────
Quick end-to-end inference test:
  - SDXL base + LoRA (fashion fine-tune)
  - ControlNet (canny edge conditioning)
  - Saves outputs/test_inference/ directory with a few sample images

Usage:
  python scripts/test_inference.py
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL    = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH     = "checkpoints/fashion_lora/unet_lora_final"
CONTROLNET    = "checkpoints/fashion_controlnet"
OUTPUT_DIR    = Path("outputs/test_inference")
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float16

PROMPTS = [
    "upcycled denim jacket, cropped silhouette, frayed edges, streetwear style, fashion photography",
    "vintage floral dress with modern lace trim, sustainable fashion, studio lighting",
    "oversized patchwork blazer, mixed fabrics, avant-garde, editorial look",
]

NEGATIVE = (
    "blurry, low quality, distorted, deformed, ugly, watermark, cartoon, bad anatomy"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_canny(img_pil: Image.Image, low=100, high=200, size=512) -> Image.Image:
    img = img_pil.resize((size, size)).convert("RGB")
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)


def make_blank_canny(size=512) -> Image.Image:
    """Use a blank canny map when no input image is provided."""
    return Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))


# ── Load pipeline ─────────────────────────────────────────────────────────────

print("Loading ControlNet …")
controlnet = ControlNetModel.from_pretrained(CONTROLNET, torch_dtype=DTYPE)

print("Loading SDXL pipeline + ControlNet …")
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16",
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

print("Loading LoRA …")
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.85)
print("LoRA fused ✓")

# ── Inference ─────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
canny = make_blank_canny(512)           # no input garment image → blank canny

generator = torch.Generator(device=DEVICE).manual_seed(42)

for idx, prompt in enumerate(PROMPTS, 1):
    print(f"\n[{idx}/{len(PROMPTS)}] Generating: {prompt[:60]}…")
    result = pipe(
        prompt=prompt + ", high realism, professional fashion photography, 8k",
        negative_prompt=NEGATIVE,
        image=canny,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=0.5,  # mild canny for text-only generation
        generator=generator,
    )
    out_path = OUTPUT_DIR / f"sample_{idx:02d}.png"
    result.images[0].save(out_path)
    print(f"  ✓ Saved → {out_path}")

print(f"\n✅ All done — check {OUTPUT_DIR}/")
