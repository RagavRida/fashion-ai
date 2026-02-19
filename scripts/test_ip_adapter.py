#!/usr/bin/env python3
"""
scripts/test_ip_adapter.py
───────────────────────────
Quick IP-Adapter inference test:
  - Takes one of the training images as the style reference
  - Generates 3 outfits conditioned on that reference image
  - Saves outputs/test_ip_adapter/

Usage:
  python3 scripts/test_ip_adapter.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "stabilityai/stable-diffusion-xl-base-1.0"
LORA_PATH        = "checkpoints/fashion_lora/unet_lora_final"
IMAGE_PROJ_PATH  = "checkpoints/fashion_ip_adapter/image_proj.pth"
IP_ATTN_PATH     = "checkpoints/fashion_ip_adapter/ip_adapter_attn.pth"
IMAGE_ENCODER_ID = "openai/clip-vit-large-patch14"
OUTPUT_DIR       = Path("outputs/test_ip_adapter")
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE            = torch.float16

PROMPTS = [
    "redesign this garment in bohemian festival style, flowy, earthy tones",
    "modern minimalist version of this outfit, clean lines, monochrome",
    "streetwear remix of this garment, oversized, urban aesthetic",
]
NEGATIVE = "blurry, low quality, distorted, deformed, ugly, watermark, cartoon"

# ── Load models ───────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_ip_adapter import ImageProjModel, IPAdapterAttnProcessor

from diffusers import (
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

print("Loading SDXL pipeline …")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL, torch_dtype=DTYPE, use_safetensors=True, variant="fp16"
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

print("Loading LoRA …")
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora(lora_scale=0.85)

print("Loading CLIP image encoder …")
image_encoder  = CLIPVisionModelWithProjection.from_pretrained(IMAGE_ENCODER_ID).to(DEVICE, dtype=DTYPE)
clip_processor = CLIPImageProcessor.from_pretrained(IMAGE_ENCODER_ID)

print("Loading IP-Adapter weights …")
cross_attn_dim = pipe.unet.config.cross_attention_dim  # 2048 for SDXL
clip_embed_dim = image_encoder.config.projection_dim   # 1024

image_proj = ImageProjModel(
    cross_attention_dim=cross_attn_dim,
    clip_embeddings_dim=clip_embed_dim,
    clip_extra_context_tokens=16,
).to(DEVICE, dtype=torch.float32)
image_proj.load_state_dict(torch.load(IMAGE_PROJ_PATH, map_location=DEVICE))
image_proj.eval()

# Install IP-Adapter attention processors
ip_attn_state = torch.load(IP_ATTN_PATH, map_location=DEVICE)
attn_procs = {}
unet_sd = pipe.unet.state_dict()
for name in pipe.unet.attn_processors.keys():
    if "attn2" in name:
        to_k_key = name.replace(".processor", ".to_k.weight")
        hidden_size = unet_sd[to_k_key].shape[0] if to_k_key in unet_sd else cross_attn_dim
        proc = IPAdapterAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attn_dim,
            num_tokens=16,
        )
        if name in ip_attn_state:
            proc.load_state_dict(ip_attn_state[name])
        attn_procs[name] = proc
    else:
        attn_procs[name] = AttnProcessor2_0()
pipe.unet.set_attn_processor(attn_procs)

# Cast all IP-Adapter modules to fp16 to match the pipeline's autocast dtype
image_proj = image_proj.to(DEVICE, dtype=DTYPE)
for proc in pipe.unet.attn_processors.values():
    if isinstance(proc, IPAdapterAttnProcessor):
        proc.to(dtype=DTYPE)
print("IP-Adapter loaded ✓")


# ── Pick a reference image from the dataset ───────────────────────────────────
ref_candidates = sorted(Path("data/processed/images_512").glob("*.jpg"))[:5]
if not ref_candidates:
    ref_candidates = sorted(Path("outputs/test_inference").glob("*.png"))
ref_image = Image.open(ref_candidates[0]).convert("RGB")
ref_image.save("outputs/test_ip_adapter_ref.jpg")
print(f"Reference image: {ref_candidates[0]}")

# ── Encode reference with CLIP ────────────────────────────────────────────────
clip_input = clip_processor(images=ref_image, return_tensors="pt")
clip_pixels = clip_input.pixel_values.to(DEVICE, dtype=DTYPE)
with torch.no_grad():
    clip_feats = image_encoder(clip_pixels).image_embeds.float()  # [1, 1024]

ip_tokens = image_proj(clip_feats)  # [1, 16, 2048]

# ── Generate images ───────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
generator = torch.Generator(device=DEVICE).manual_seed(42)

for idx, prompt in enumerate(PROMPTS, 1):
    print(f"\n[{idx}/{len(PROMPTS)}] {prompt[:60]}…")

    # We pass ip_tokens via a custom cross_attention_kwargs hook
    # The IPAdapterAttnProcessor splits encoder_hidden_states by num_tokens
    # Simple approach: append ip_tokens to the text-encoder output via prompt_embeds

    # Get text embeds using pipe's encode_prompt helper
    (prompt_embeds, neg_embeds,
     pooled_prompt_embeds, neg_pooled_embeds) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        negative_prompt=NEGATIVE,
        negative_prompt_2=NEGATIVE,
        device=DEVICE,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    # Append ip_tokens to prompt_embeds (ip processor splits from the end)
    ip_tok = ip_tokens.to(DEVICE, dtype=prompt_embeds.dtype)

    # Positive: concat text + ip_tokens
    prompt_embeds_ip    = torch.cat([prompt_embeds, ip_tok], dim=1)
    # Negative: concat with zeros (uncond image)
    neg_ip_zeros        = torch.zeros_like(ip_tok)
    neg_embeds_ip       = torch.cat([neg_embeds, neg_ip_zeros], dim=1)

    result = pipe(
        prompt_embeds=prompt_embeds_ip,
        negative_prompt_embeds=neg_embeds_ip,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=neg_pooled_embeds,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator,
        height=512,
        width=512,
    )
    out_path = OUTPUT_DIR / f"ip_sample_{idx:02d}.png"
    result.images[0].save(out_path)
    print(f"  ✓ Saved → {out_path}")

print(f"\n✅ All done — check {OUTPUT_DIR}/")
print(f"   Reference image → outputs/test_ip_adapter_ref.jpg")
