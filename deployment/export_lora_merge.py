"""
deployment/export_lora_merge.py
--------------------------------
Merge LoRA weights into the base SDXL model for faster inference.

After merging, the output model is a standalone SDXL checkpoint that
doesn't require LoRA loading at inference time — this saves ~0.5s per run.

Usage
-----
python deployment/export_lora_merge.py \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora checkpoints/fashion_lora/fashion_lora.safetensors \
    --output checkpoints/merged_model/ \
    --lora_scale 0.85

Options
-------
--base_model    HuggingFace model ID or local path
--lora          path to LoRA safetensors file
--output        directory to save merged model
--lora_scale    LoRA merge strength [default: 0.85]
--dtype         output dtype: float16 | bfloat16 | float32 [default: float16]
--verify        run a quick verification generate after merging
--verify_prompt text prompt for verification [default: fashion upcycling]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge LoRA into base SDXL model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base_model", required=True,
                   help="HuggingFace model ID or local path to base SDXL model")
    p.add_argument("--lora", required=True,
                   help="Path to LoRA safetensors file")
    p.add_argument("--output", default="checkpoints/merged_model/",
                   help="Output directory for merged model")
    p.add_argument("--lora_scale", type=float, default=0.85,
                   help="LoRA merge scale (0.0–1.0)")
    p.add_argument("--dtype", default="float16",
                   choices=list(DTYPE_MAP.keys()),
                   help="Output model dtype")
    p.add_argument("--verify", action="store_true",
                   help="Run a quick generation after merging to verify the model")
    p.add_argument("--verify_prompt", type=str,
                   default="fashionable upcycled denim jacket, premium quality, studio lighting",
                   help="Prompt for post-merge verification")
    return p.parse_args()


def load_and_merge(args) -> tuple:
    """Load base pipeline, apply LoRA, fuse, and return pipeline + metadata."""
    from diffusers import StableDiffusionXLPipeline, UniPCMultistepScheduler

    dtype = DTYPE_MAP[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base model from: {args.base_model}")
    logger.info(f"Device: {device}, dtype: {args.dtype}")

    start = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    load_time = time.time() - start
    logger.info(f"Base model loaded in {load_time:.1f}s")

    # ── Apply LoRA ──
    lora_path = Path(args.lora)
    if not lora_path.exists():
        logger.error(f"LoRA file not found: {lora_path}")
        sys.exit(1)

    logger.info(f"Loading LoRA from: {lora_path}")
    pipe.load_lora_weights(str(lora_path))

    logger.info(f"Fusing LoRA with scale={args.lora_scale}")
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.unload_lora_weights()   # Remove LoRA adapters — weights now baked in

    logger.info("LoRA fused and unloaded — weights are now merged")

    merge_metadata = {
        "base_model": args.base_model,
        "lora_path": str(args.lora),
        "lora_scale": args.lora_scale,
        "output_dtype": args.dtype,
        "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return pipe, device, merge_metadata


def save_merged_model(pipe, output_dir: str, metadata: dict) -> None:
    """Save the merged pipeline using safetensors format."""
    out = Path(output_dir)

    if out.exists():
        logger.warning(f"Output directory already exists: {out}. Overwriting.")
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving merged model to: {out}")
    save_start = time.time()

    pipe.save_pretrained(str(out), safe_serialization=True)
    save_time = time.time() - save_start
    logger.info(f"Model saved in {save_time:.1f}s")

    # Save merge metadata
    meta_path = out / "merge_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Merge metadata saved: {meta_path}")

    # Report disk usage
    total_size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    logger.info(f"Total saved size: {total_size / (1024**3):.2f} GB")


def verify_merged_model(pipe, device: str, prompt: str) -> None:
    """Run a quick test generation to verify the merged model works."""
    logger.info(f"Verifying merged model with prompt: '{prompt[:60]}...'")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    try:
        import xformers  # noqa
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pipe.enable_attention_slicing()

    gen_start = time.time()
    with torch.no_grad(), torch.autocast(device, dtype=torch.float16, enabled=(device == "cuda")):
        result = pipe(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted",
            num_inference_steps=20,
            guidance_scale=7.5,
            num_images_per_prompt=1,
        )
    gen_time = time.time() - gen_start

    verify_path = Path("outputs/merge_verify.jpg")
    verify_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(str(verify_path), "JPEG", quality=95)

    logger.success(
        f"Verification passed! Image saved: {verify_path}\n"
        f"  Generation time: {gen_time:.1f}s"
    )


def main():
    args = parse_args()

    logger.info("=" * 55)
    logger.info("  Fashion Reuse Studio — LoRA Merge Export")
    logger.info("=" * 55)
    logger.info(f"  Base model  : {args.base_model}")
    logger.info(f"  LoRA        : {args.lora}")
    logger.info(f"  Output      : {args.output}")
    logger.info(f"  Scale       : {args.lora_scale}")
    logger.info(f"  Dtype       : {args.dtype}")

    total_start = time.time()

    # ── Load + merge ──
    pipe, device, metadata = load_and_merge(args)

    # ── Save ──
    save_merged_model(pipe, args.output, metadata)

    # ── Verify (optional) ──
    if args.verify:
        verify_merged_model(pipe, device, args.verify_prompt)
    else:
        logger.info(
            "Skipping verification. Run with --verify to test the merged model.\n"
            "To use the merged model:\n"
            f"  python deployment/local_runner.py --model {args.output} --mode cloud_fp16 ..."
        )

    total_time = time.time() - total_start
    logger.success(f"\nMerge complete in {total_time:.1f}s")
    logger.info(f"Merged model saved at: {Path(args.output).resolve()}")
    logger.info("Usage after merge (no LoRA needed at inference):")
    logger.info(
        f"  python deployment/local_runner.py \\\n"
        f"    --model {args.output} \\\n"
        f"    --mode local_int8 \\\n"
        f"    --prompt 'your fashion prompt' \\\n"
        f"    --output outputs/"
    )


if __name__ == "__main__":
    main()
