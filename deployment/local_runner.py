"""
deployment/local_runner.py
---------------------------
CLI script to run Fashion Reuse Studio locally with quantized models.

Usage
-----
# Prompt-only (text → image)
python deployment/local_runner.py \
    --mode local_int8 \
    --prompt "Upcycle this denim jacket into a cropped jacket with patches" \
    --output outputs/

# Image redesign (image → redesigned image)
python deployment/local_runner.py \
    --mode local_int8 \
    --image inputs/jacket.jpg \
    --output outputs/

# Image + prompt redesign
python deployment/local_runner.py \
    --mode local_int8 \
    --image inputs/jacket.jpg \
    --prompt "Convert into a bohemian crop top with floral embroidery" \
    --output outputs/

# Refinement (previous output + new instruction)
python deployment/local_runner.py \
    --mode local_int8 \
    --refine \
    --image outputs/design_0.jpg \
    --prompt "Make sleeves shorter and add distressed texture" \
    --output outputs/refined/

Options
-------
--mode          auto | cloud_fp16 | local_int8 | local_int4  [default: auto]
--prompt        text description
--image         path to input garment image (jpg/png)
--output        directory to save generated images
--refine        flag: treat --image as previous output to refine
--n_images      number of images to generate per run [default: 4]
--steps         override default inference steps for mode
--guidance      guidance scale [default: 7.5]
--strength      img2img strength for refine mode [default: 0.6]
--seed          random seed for reproducibility [default: None]
--save_grid     also save a 2x2 grid image of all outputs
--model         base model ID or path [default: stabilityai/stable-diffusion-xl-base-1.0]
--lora          path to LoRA weights
--controlnet    path to ControlNet weights directory
--ip_adapter    path to IP-Adapter weights directory
--no_controlnet disable ControlNet even if path provided
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from deployment.quantization import load_pipeline_quantized, DEFAULT_STEPS


# ─── Helpers ──────────────────────────────────────────────────────────────────
def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    # Resize to SDXL-friendly 1024×1024 (or 512×512 for low VRAM)
    if img.width != 1024 or img.height != 1024:
        img = img.resize((1024, 1024), Image.LANCZOS)
    return img


def resize_for_vram(img: Image.Image, mode: str) -> Image.Image:
    """Downscale to 512×512 for int4 to save VRAM."""
    if mode == "local_int4":
        return img.resize((512, 512), Image.LANCZOS)
    return img


def save_outputs(images: list, output_dir: str, prefix: str, save_grid: bool = False) -> list[str]:
    """Save generated images and return file paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for i, img in enumerate(images):
        path = out / f"{prefix}_{i}.jpg"
        img.save(str(path), "JPEG", quality=95)
        paths.append(str(path))
        logger.info(f"Saved: {path}")

    if save_grid and len(images) >= 4:
        _save_grid(images[:4], out / f"{prefix}_grid.jpg")

    return paths


def _save_grid(images: list[Image.Image], path: Path) -> None:
    """Combine 4 images into a 2×2 grid."""
    w, h = images[0].size
    grid = Image.new("RGB", (w * 2, h * 2), color=(10, 10, 20))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for img, pos in zip(images, positions):
        grid.paste(img, pos)
    grid.save(str(path), "JPEG", quality=90)
    logger.info(f"Grid saved: {path}")


def log_generation_summary(
    mode: str, gen_type: str, n_images: int,
    elapsed: float, vram_peak: float, paths: list[str]
) -> None:
    logger.success(
        f"\n{'─'*50}\n"
        f"  Mode        : {mode}\n"
        f"  Type        : {gen_type}\n"
        f"  Images      : {n_images}\n"
        f"  Time        : {elapsed:.1f}s ({elapsed/n_images:.1f}s/image)\n"
        f"  VRAM peak   : {vram_peak:.2f} GB\n"
        f"  Saved to    : {Path(paths[0]).parent}\n"
        f"{'─'*50}"
    )


# ─── Generation Functions ─────────────────────────────────────────────────────
NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, signature, cropped, worst quality, extra limbs"
)


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
def run_generate(pipe, args) -> list[Image.Image]:
    """Text-prompt only generation."""
    logger.info(f"[GENERATE] prompt='{args.prompt[:60]}...'")
    steps = args.steps or pipe.default_steps

    result = pipe.base_pipe(
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=steps,
        guidance_scale=args.guidance,
        num_images_per_prompt=args.n_images,
        generator=torch.Generator(pipe.device).manual_seed(args.seed) if args.seed else None,
    )
    return result.images


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
def run_redesign(pipe, args) -> list[Image.Image]:
    """Image-based redesign using img2img."""
    logger.info(f"[REDESIGN] image='{args.image}', prompt='{args.prompt or '(auto)'}'")
    steps = args.steps or pipe.default_steps
    image = resize_for_vram(load_image(args.image), pipe.mode)

    # Build a style prompt if none given
    prompt = args.prompt or (
        "fashionable upcycled garment, premium quality, detailed fabric texture, "
        "sustainable fashion, high-end design, studio lighting"
    )

    result = pipe.img2img_pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        strength=args.strength,
        num_inference_steps=steps,
        guidance_scale=args.guidance,
        num_images_per_prompt=args.n_images,
        generator=torch.Generator(pipe.device).manual_seed(args.seed) if args.seed else None,
    )
    return result.images


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
def run_redesign_with_controlnet(pipe, args) -> list[Image.Image]:
    """
    Image + prompt redesign using ControlNet edge conditioning.
    Falls back to img2img if ControlNet not loaded.
    """
    if pipe.controlnet is None:
        logger.warning("ControlNet not loaded — falling back to img2img redesign")
        return run_redesign(pipe, args)

    import cv2
    import numpy as np

    logger.info(f"[REDESIGN+CONTROLNET] image='{args.image}', prompt='{args.prompt}'")
    steps = args.steps or pipe.default_steps
    image = resize_for_vram(load_image(args.image), pipe.mode)

    # Extract Canny edges for ControlNet conditioning
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edge_img = Image.fromarray(edges_rgb)

    result = pipe.base_pipe(
        prompt=args.prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=edge_img,
        num_inference_steps=steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=0.75,
        num_images_per_prompt=args.n_images,
        generator=torch.Generator(pipe.device).manual_seed(args.seed) if args.seed else None,
    )
    return result.images


@torch.no_grad()
@torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available())
def run_refine(pipe, args) -> list[Image.Image]:
    """Refinement loop — apply a new prompt to an existing generated image."""
    logger.info(f"[REFINE] base='{args.image}', instruction='{args.prompt}'")
    steps = args.steps or max(pipe.default_steps, 20)
    image = resize_for_vram(load_image(args.image), pipe.mode)

    # Use a lower strength for subtle refinements
    strength = args.strength if args.strength != 0.6 else 0.5
    prompt = f"{args.prompt}, high fashion, detailed, premium quality"

    result = pipe.img2img_pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=image,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=args.guidance,
        num_images_per_prompt=args.n_images,
        generator=torch.Generator(pipe.device).manual_seed(args.seed) if args.seed else None,
    )
    return result.images


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fashion Reuse Studio — Local Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", default="auto",
                   choices=["auto", "cloud_fp16", "local_int8", "local_int4"],
                   help="Quantization mode")
    p.add_argument("--prompt", type=str, default=None,
                   help="Text generation/refinement prompt")
    p.add_argument("--image", type=str, default=None,
                   help="Input garment image path")
    p.add_argument("--output", type=str, default="outputs/local/",
                   help="Output directory for generated images")
    p.add_argument("--refine", action="store_true",
                   help="Refine mode: --image is base output, --prompt is refinement instruction")
    p.add_argument("--n_images", type=int, default=4,
                   help="Number of images to generate")
    p.add_argument("--steps", type=int, default=None,
                   help="Override default inference steps for the mode")
    p.add_argument("--guidance", type=float, default=7.5,
                   help="Classifier-free guidance scale")
    p.add_argument("--strength", type=float, default=0.6,
                   help="img2img strength (0.0–1.0)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--save_grid", action="store_true",
                   help="Also save a 2×2 grid image of all outputs")
    p.add_argument("--model", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="Base model HuggingFace ID or local path")
    p.add_argument("--lora", type=str, default=None,
                   help="Path to LoRA safetensors file")
    p.add_argument("--controlnet", type=str, default=None,
                   help="Path to ControlNet weights directory")
    p.add_argument("--ip_adapter", type=str, default=None,
                   help="Path to IP-Adapter weights directory")
    p.add_argument("--no_controlnet", action="store_true",
                   help="Disable ControlNet even if --controlnet path given")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate inputs ──
    has_image = args.image is not None
    has_prompt = args.prompt is not None

    if not has_image and not has_prompt:
        logger.error("Provide at least --prompt or --image (or both).")
        sys.exit(1)

    if args.refine and not has_image:
        logger.error("--refine requires --image (the previous output to refine).")
        sys.exit(1)

    if args.refine and not has_prompt:
        logger.error("--refine requires --prompt (the refinement instruction).")
        sys.exit(1)

    if has_image and not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)

    # ── Determine run type ──
    if args.refine:
        run_type = "refine"
    elif has_image and has_prompt:
        run_type = "redesign_prompt" if not args.no_controlnet else "redesign"
    elif has_image:
        run_type = "redesign"
    else:
        run_type = "generate"

    logger.info(f"Run type: {run_type}")

    # ── Load pipeline ──
    load_start = time.time()
    pipe = load_pipeline_quantized(
        mode=args.mode,
        base_model=args.model,
        lora_path=args.lora,
        controlnet_path=args.controlnet,
        ip_adapter_path=args.ip_adapter,
        load_controlnet=not args.no_controlnet,
    )
    load_time = time.time() - load_start
    logger.info(f"Pipeline loaded in {load_time:.1f}s")

    # Reset VRAM peak counter before generation
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ── Run generation ──
    gen_start = time.time()

    if run_type == "generate":
        images = run_generate(pipe, args)
    elif run_type == "redesign":
        images = run_redesign(pipe, args)
    elif run_type == "redesign_prompt":
        images = run_redesign_with_controlnet(pipe, args)
    elif run_type == "refine":
        images = run_refine(pipe, args)

    elapsed = time.time() - gen_start
    vram_peak = pipe.vram_peak_gb

    # ── Save outputs ──
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{run_type}_{pipe.mode}_{ts}"
    paths = save_outputs(images, args.output, prefix, save_grid=args.save_grid)

    # ── Summary ──
    log_generation_summary(pipe.mode, run_type, len(images), elapsed, vram_peak, paths)

    # Save metadata JSON alongside outputs
    meta = {
        "mode": pipe.mode,
        "run_type": run_type,
        "prompt": args.prompt,
        "image": args.image,
        "n_images": len(images),
        "steps": args.steps or pipe.default_steps,
        "guidance": args.guidance,
        "strength": args.strength,
        "seed": args.seed,
        "generation_time_secs": round(elapsed, 2),
        "vram_peak_gb": round(vram_peak, 3),
        "output_files": paths,
    }
    meta_path = Path(args.output) / f"{prefix}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved: {meta_path}")


if __name__ == "__main__":
    main()
