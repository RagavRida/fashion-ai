"""
deployment/benchmark.py
------------------------
Benchmark the Fashion Reuse Studio pipeline across all quantization modes.

Measures per mode:
  - Model load time
  - Generation time per image (mean, min, max over N runs)
  - Peak VRAM usage (GB)
  - Allocated VRAM after loading (GB)
  - Qualitative CLIP score (text-image agreement, if clip installed)

Usage
-----
# Benchmark all 3 modes (requires GPU with enough VRAM for each)
python deployment/benchmark.py \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --modes cloud_fp16 local_int8 local_int4 \
    --n_runs 3 \
    --n_images 4 \
    --output outputs/benchmark_report.json

# Benchmark only local modes (for 8-12 GB GPU)
python deployment/benchmark.py \
    --modes local_int8 local_int4 \
    --n_runs 3

Options
-------
--model         base model ID or local/merged model path
--lora          path to LoRA safetensors (optional)
--controlnet    path to ControlNet weights (optional)
--modes         space-separated list of modes to benchmark
--n_runs        number of generation runs per mode [default: 3]
--n_images      images per run [default: 4]
--steps         fixed inference steps for all modes [default: use mode default]
--output        output JSON report path [default: outputs/benchmark_report.json]
--prompt        benchmark prompt [default: fashion upcycling subject]
--no_clip       skip CLIP scoring
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from deployment.quantization import load_pipeline_quantized, DEFAULT_STEPS, get_total_vram_gb


# ─── CLIP scoring ─────────────────────────────────────────────────────────────
def _clip_score_images(images, prompt: str) -> Optional[float]:
    """Compute mean CLIP cosine similarity between images and prompt."""
    try:
        import clip
    except ImportError:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    text = clip.tokenize([prompt], truncate=True).to(device)

    scores = []
    for img in images:
        img_t = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img_t)
            txt_feat = model.encode_text(text)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
            score = (img_feat @ txt_feat.T).item()
        scores.append(score)
    return float(mean(scores))


# ─── VRAM measurement ─────────────────────────────────────────────────────────
def reset_vram_stats():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def vram_allocated_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 3)


def vram_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# ─── Single mode benchmark ────────────────────────────────────────────────────
def benchmark_mode(
    mode: str,
    prompt: str,
    n_runs: int,
    n_images: int,
    fixed_steps: Optional[int],
    model: str,
    lora_path: Optional[str],
    controlnet_path: Optional[str],
    no_clip: bool,
) -> dict:
    """Run benchmark for one mode and return a results dict."""
    logger.info(f"\n{'═'*55}")
    logger.info(f"  Benchmarking: {mode.upper()}")
    logger.info(f"{'═'*55}")
    logger.info(f"  Runs: {n_runs} × {n_images} images | Steps: {fixed_steps or 'mode default'}")

    result = {
        "mode": mode,
        "n_runs": n_runs,
        "n_images_per_run": n_images,
        "error": None,
    }

    reset_vram_stats()

    # ── Load pipeline ──
    try:
        load_start = time.time()
        pipe = load_pipeline_quantized(
            mode=mode,
            base_model=model,
            lora_path=lora_path,
            controlnet_path=controlnet_path,
            load_controlnet=False,   # faster benchmarking without ControlNet
        )
        load_time = time.time() - load_start
        vram_after_load = vram_allocated_gb()

        result["load_time_secs"] = round(load_time, 2)
        result["vram_allocated_after_load_gb"] = round(vram_after_load, 3)
        result["default_steps"] = pipe.default_steps
        result["actual_steps"] = fixed_steps or pipe.default_steps
        result["gpu_available"] = torch.cuda.is_available()
        result["total_vram_gb"] = round(get_total_vram_gb(), 2)

        logger.info(f"Load time: {load_time:.1f}s | VRAM after load: {vram_after_load:.2f} GB")

    except Exception as e:
        logger.error(f"Failed to load pipeline in {mode}: {e}")
        result["error"] = str(e)
        return result

    # ── Generation runs ──
    gen_times = []
    all_images = []
    steps = fixed_steps or pipe.default_steps

    NEGATIVE = "low quality, blurry, distorted, deformed, ugly"

    for run_idx in range(n_runs):
        reset_vram_stats()
        logger.info(f"  Run {run_idx + 1}/{n_runs}...")

        try:
            run_start = time.time()
            with torch.no_grad(), torch.autocast(
                pipe.device, dtype=torch.float16, enabled=(pipe.device == "cuda")
            ):
                out = pipe.base_pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE,
                    num_inference_steps=steps,
                    guidance_scale=7.5,
                    num_images_per_prompt=n_images,
                )
            run_time = time.time() - run_start
            peak = vram_peak_gb()

            gen_times.append(run_time)
            all_images.extend(out.images)

            logger.info(
                f"    ✓ {run_time:.1f}s | {run_time/n_images:.1f}s/image | "
                f"VRAM peak: {peak:.2f} GB"
            )

        except torch.cuda.OutOfMemoryError as oom:
            logger.error(f"    OOM on run {run_idx + 1}: {oom}")
            result["error"] = f"CUDA OOM: {oom}"
            break
        except Exception as e:
            logger.error(f"    Error on run {run_idx + 1}: {e}")
            result["error"] = str(e)
            break

    if not gen_times:
        return result

    # ── Compute stats ──
    per_image_times = [t / n_images for t in gen_times]

    result["generation"] = {
        "total_times_secs": [round(t, 2) for t in gen_times],
        "mean_run_secs": round(mean(gen_times), 2),
        "stdev_run_secs": round(stdev(gen_times), 3) if len(gen_times) > 1 else 0.0,
        "min_run_secs": round(min(gen_times), 2),
        "max_run_secs": round(max(gen_times), 2),
        "mean_per_image_secs": round(mean(per_image_times), 2),
        "min_per_image_secs": round(min(per_image_times), 2),
        "max_per_image_secs": round(max(per_image_times), 2),
    }

    # Re-measure peak after all runs
    result["vram_peak_gb"] = round(vram_peak_gb(), 3)

    # ── CLIP scoring ──
    if not no_clip and all_images:
        logger.info("  Computing CLIP score...")
        clip_score = _clip_score_images(all_images[:8], prompt)
        result["clip_score"] = round(clip_score, 4) if clip_score else None
        if clip_score:
            logger.info(f"  CLIP score: {clip_score:.4f}")

    # ── Save sample images ──
    sample_dir = Path("outputs/benchmark_samples") / mode
    sample_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(all_images[:4]):
        img.save(str(sample_dir / f"sample_{i}.jpg"), "JPEG", quality=90)
    result["sample_dir"] = str(sample_dir)

    # ── Cleanup ──
    pipe.clear_cache()
    del pipe
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        f"\n  {'─'*45}\n"
        f"  Mode         : {mode}\n"
        f"  Load time    : {result['load_time_secs']}s\n"
        f"  Mean gen time: {result['generation']['mean_run_secs']}s "
        f"({result['generation']['mean_per_image_secs']}s/image)\n"
        f"  VRAM peak    : {result['vram_peak_gb']} GB\n"
        f"  CLIP score   : {result.get('clip_score', 'N/A')}\n"
        f"  {'─'*45}"
    )

    return result


# ─── Comparison table ─────────────────────────────────────────────────────────
def print_comparison_table(results: list[dict]) -> None:
    """Print a human-readable comparison table."""
    logger.info("\n" + "═" * 75)
    logger.info("  BENCHMARK COMPARISON TABLE")
    logger.info("═" * 75)

    header = f"{'Mode':<18} {'Load(s)':<10} {'Gen/run(s)':<12} {'s/img':<8} {'VRAM pk':<10} {'CLIP':<8}"
    logger.info(header)
    logger.info("─" * 75)

    for r in results:
        if r.get("error") and "generation" not in r:
            logger.info(f"{r['mode']:<18} FAILED: {r['error']}")
            continue
        gen = r.get("generation", {})
        logger.info(
            f"{r['mode']:<18} "
            f"{r.get('load_time_secs', '–'):<10} "
            f"{gen.get('mean_run_secs', '–'):<12} "
            f"{gen.get('mean_per_image_secs', '–'):<8} "
            f"{r.get('vram_peak_gb', '–'):<10} "
            f"{r.get('clip_score', 'N/A'):<8}"
        )

    logger.info("═" * 75)


# ─── Main ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Fashion Reuse Studio — Quantization Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",
                   default="stabilityai/stable-diffusion-xl-base-1.0",
                   help="Base model HuggingFace ID or local path")
    p.add_argument("--lora", type=str, default=None)
    p.add_argument("--controlnet", type=str, default=None)
    p.add_argument("--modes", nargs="+",
                   default=["cloud_fp16", "local_int8", "local_int4"],
                   help="Modes to benchmark")
    p.add_argument("--n_runs", type=int, default=3,
                   help="Number of generation runs per mode")
    p.add_argument("--n_images", type=int, default=4,
                   help="Images per run")
    p.add_argument("--steps", type=int, default=None,
                   help="Fixed steps (overrides mode default)")
    p.add_argument("--output", default="outputs/benchmark_report.json",
                   help="Output JSON report path")
    p.add_argument("--prompt",
                   default=(
                       "fashionable upcycled denim jacket, cropped, streetwear aesthetic, "
                       "detailed texture, studio photography, premium quality"
                   ),
                   help="Benchmark generation prompt")
    p.add_argument("--no_clip", action="store_true",
                   help="Skip CLIP scoring")
    return p.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 55)
    logger.info("  Fashion Reuse Studio — Benchmark")
    logger.info("=" * 55)
    logger.info(f"  Modes       : {', '.join(args.modes)}")
    logger.info(f"  Runs        : {args.n_runs} × {args.n_images} images")
    logger.info(f"  GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"  Total VRAM  : {get_total_vram_gb():.1f} GB")

    all_results = []
    benchmark_start = time.time()

    for mode in args.modes:
        r = benchmark_mode(
            mode=mode,
            prompt=args.prompt,
            n_runs=args.n_runs,
            n_images=args.n_images,
            fixed_steps=args.steps,
            model=args.model,
            lora_path=args.lora,
            controlnet_path=args.controlnet,
            no_clip=args.no_clip,
        )
        all_results.append(r)

    total_time = time.time() - benchmark_start

    # ── Print comparison table ──
    print_comparison_table(all_results)

    # ── Save JSON report ──
    report = {
        "benchmark_info": {
            "model": args.model,
            "prompt": args.prompt,
            "n_runs": args.n_runs,
            "n_images_per_run": args.n_images,
            "total_vram_gb": round(get_total_vram_gb(), 2),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "total_benchmark_time_secs": round(total_time, 1),
        },
        "results": all_results,
        "summary": {
            r["mode"]: {
                "s_per_image": r.get("generation", {}).get("mean_per_image_secs"),
                "vram_peak_gb": r.get("vram_peak_gb"),
                "clip_score": r.get("clip_score"),
                "error": r.get("error"),
            }
            for r in all_results
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.success(f"\nBenchmark report saved: {out_path}")
    logger.success(f"Total benchmark time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
