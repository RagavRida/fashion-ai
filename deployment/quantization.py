"""
deployment/quantization.py
---------------------------
Quantized model loading for Fashion Reuse Studio.

Supports three deployment modes:
  - cloud_fp16   : full fp16 (best quality, needs 16+ GB VRAM)
  - local_int8   : bitsandbytes 8-bit UNet + CLIP (8–12 GB VRAM)
  - local_int4   : NF4 4-bit UNet, 8-bit CLIP (6–8 GB VRAM)

ControlNet and IP-Adapter always run in fp16 by default.

Public API
----------
  detect_mode()                  → str  (auto-picks mode based on VRAM)
  load_pipeline_quantized(...)   → QuantizedPipeline
"""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import bitsandbytes as bnb  # noqa: F401
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    logger.warning("bitsandbytes not installed. int8/int4 modes unavailable.")

try:
    import xformers  # noqa: F401
    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    UniPCMultistepScheduler,
)
from transformers import (
    BitsAndBytesConfig,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
)


# ─── Constants ────────────────────────────────────────────────────────────────
VRAM_THRESHOLD_INT8 = 12.0   # GB — use int8 below this
VRAM_THRESHOLD_INT4 = 8.0    # GB — use int4 below this

DEFAULT_STEPS = {
    "cloud_fp16": 50,
    "local_int8": 25,
    "local_int4": 20,
}

DEFAULT_GUIDANCE = 7.5


# ─── VRAM Detection ───────────────────────────────────────────────────────────
def get_available_vram_gb() -> float:
    """Return free VRAM in GB on the active CUDA device, or 0 for CPU."""
    if not torch.cuda.is_available():
        return 0.0
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return free / (1024 ** 3)


def get_total_vram_gb() -> float:
    """Return total VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    device = torch.cuda.current_device()
    _, total = torch.cuda.mem_get_info(device)
    return total / (1024 ** 3)


def detect_mode() -> str:
    """
    Auto-detect the best quantization mode based on available VRAM.

    Returns one of: 'cloud_fp16', 'local_int8', 'local_int4'
    """
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPU detected. Falling back to CPU (very slow).")
        return "local_int4"

    vram = get_total_vram_gb()
    logger.info(f"Total VRAM detected: {vram:.1f} GB")

    if vram >= VRAM_THRESHOLD_INT8:
        mode = "local_int8" if vram < 16.0 else "cloud_fp16"
    else:
        mode = "local_int4"

    logger.info(f"Auto-selected mode: {mode}")
    return mode


# ─── BitsAndBytes Configs ─────────────────────────────────────────────────────
def _make_bnb_int8_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(load_in_8bit=True)


def _make_bnb_int4_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 GB
    )


# ─── Model Loaders ────────────────────────────────────────────────────────────
def _load_text_encoders(base_model: str, mode: str, device: str):
    """
    Load CLIP text encoders with quantization appropriate for mode.

    - cloud_fp16  → fp16
    - local_int8  → 8-bit
    - local_int4  → 8-bit (4-bit for text encoders degrades quality too much)
    """
    common = dict(pretrained_model_name_or_path=base_model, subfolder="text_encoder")
    common2 = dict(pretrained_model_name_or_path=base_model, subfolder="text_encoder_2")

    if mode == "cloud_fp16":
        te1 = CLIPTextModel.from_pretrained(**common, torch_dtype=torch.float16).to(device)
        te2 = CLIPTextModelWithProjection.from_pretrained(**common2, torch_dtype=torch.float16).to(device)
    elif BNB_AVAILABLE and mode in ("local_int8", "local_int4"):
        bnb_cfg = _make_bnb_int8_config()
        te1 = CLIPTextModel.from_pretrained(
            **common, quantization_config=bnb_cfg, device_map="auto"
        )
        te2 = CLIPTextModelWithProjection.from_pretrained(
            **common2, quantization_config=bnb_cfg, device_map="auto"
        )
    else:
        logger.warning("bitsandbytes not available — loading text encoders in fp16")
        te1 = CLIPTextModel.from_pretrained(**common, torch_dtype=torch.float16).to(device)
        te2 = CLIPTextModelWithProjection.from_pretrained(**common2, torch_dtype=torch.float16).to(device)

    return te1, te2


def _load_unet_quantized(base_model: str, mode: str, device: str):
    """Load UNet with the appropriate quantization configuration."""
    from diffusers import UNet2DConditionModel

    if mode == "cloud_fp16":
        unet = UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet", torch_dtype=torch.float16
        ).to(device)

    elif mode == "local_int8":
        if not BNB_AVAILABLE:
            logger.warning("bitsandbytes missing — loading UNet in fp16")
            return _load_unet_quantized(base_model, "cloud_fp16", device)
        bnb_cfg = _make_bnb_int8_config()
        unet = UNet2DConditionModel.from_pretrained(
            base_model,
            subfolder="unet",
            quantization_config=bnb_cfg,
            device_map="auto",
        )

    elif mode == "local_int4":
        if not BNB_AVAILABLE:
            logger.warning("bitsandbytes missing — loading UNet in fp16")
            return _load_unet_quantized(base_model, "cloud_fp16", device)
        bnb_cfg = _make_bnb_int4_config()
        warnings.warn(
            "local_int4 mode: expect ~15-25% quality reduction vs fp16. "
            "Increase num_inference_steps to partially compensate.",
            UserWarning,
        )
        unet = UNet2DConditionModel.from_pretrained(
            base_model,
            subfolder="unet",
            quantization_config=bnb_cfg,
            device_map="auto",
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return unet


def _load_vae(base_model: str, device: str) -> AutoencoderKL:
    """VAE is always loaded in fp16 — quantizing it causes visible artifacts."""
    # Use SDXL-specific VAE fix for numerical stability
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        ).to(device)
        logger.info("Loaded fp16-fix VAE from madebyollin/sdxl-vae-fp16-fix")
    except Exception:
        vae = AutoencoderKL.from_pretrained(
            base_model, subfolder="vae", torch_dtype=torch.float16
        ).to(device)
        logger.info("Loaded VAE from base model")
    return vae


def _apply_memory_optimizations(pipe, mode: str) -> None:
    """Apply attention / memory optimizations to the pipeline."""
    if XFORMERS_AVAILABLE:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("xformers memory-efficient attention enabled")
            return
        except Exception as e:
            logger.warning(f"xformers failed ({e}), falling back to attention slicing")

    # Fallback — always safe
    pipe.enable_attention_slicing(slice_size="auto")
    logger.info("Attention slicing enabled (xformers fallback)")

    # For int4 on very low VRAM, also enable CPU offload
    if mode == "local_int4":
        try:
            pipe.enable_model_cpu_offload()
            logger.info("CPU offload enabled for local_int4 mode")
        except Exception as e:
            logger.warning(f"CPU offload failed: {e}")


# ─── QuantizedPipeline ────────────────────────────────────────────────────────
@dataclass
class QuantizedPipeline:
    """
    Wrapper around loaded diffusion pipelines with embedding caches.

    Attributes
    ----------
    mode : deployment mode string
    base_pipe : text2img or controlnet pipeline (used for generate)
    img2img_pipe : img2img pipeline (used for redesign / refine)
    controlnet : ControlNetModel if loaded
    device : target device string
    default_steps : number of inference steps for this mode
    _embed_cache : cached (prompt → text embeddings) dict
    _ip_embed_cache : cached (image_hash → IP embeddings) dict
    """
    mode: str
    base_pipe: object
    img2img_pipe: object
    controlnet: Optional[object]
    device: str
    default_steps: int
    _embed_cache: dict = field(default_factory=dict, repr=False)
    _ip_embed_cache: dict = field(default_factory=dict, repr=False)

    def get_text_embeddings(self, prompt: str, negative_prompt: str = ""):
        """Return cached or freshly-computed text embeddings."""
        key = (prompt, negative_prompt)
        if key not in self._embed_cache:
            with torch.autocast(self.device if self.device != "cpu" else "cpu",
                                dtype=torch.float16, enabled=(self.device != "cpu")):
                result = self.base_pipe.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                )
            self._embed_cache[key] = result
            # Limit cache size to avoid OOM
            if len(self._embed_cache) > 32:
                oldest = next(iter(self._embed_cache))
                del self._embed_cache[oldest]
        return self._embed_cache[key]

    def get_ip_embeddings(self, image_hash: str, image):
        """Return cached or freshly-computed IP-Adapter image embeddings."""
        if image_hash not in self._ip_embed_cache:
            # IP-Adapter embed extraction via pipeline's image encoder
            if hasattr(self.base_pipe, "image_encoder") and self.base_pipe.image_encoder:
                import torchvision.transforms as T
                transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                                       T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
                img_t = transform(image).unsqueeze(0).to(self.device).half()
                with torch.no_grad():
                    embeds = self.base_pipe.image_encoder(img_t).image_embeds
                self._ip_embed_cache[image_hash] = embeds
                if len(self._ip_embed_cache) > 16:
                    oldest = next(iter(self._ip_embed_cache))
                    del self._ip_embed_cache[oldest]
        return self._ip_embed_cache.get(image_hash)

    def clear_cache(self):
        """Free embedding caches + trigger CUDA memory cleanup."""
        self._embed_cache.clear()
        self._ip_embed_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("Embedding caches cleared")

    @property
    def vram_used_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    @property
    def vram_peak_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 3)


# ─── Main Entry Point ─────────────────────────────────────────────────────────
def load_pipeline_quantized(
    mode: str = "auto",
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    lora_path: Optional[str] = None,
    controlnet_path: Optional[str] = None,
    ip_adapter_path: Optional[str] = None,
    load_controlnet: bool = True,
) -> QuantizedPipeline:
    """
    Build and return a QuantizedPipeline for the specified mode.

    Parameters
    ----------
    mode : 'auto' | 'cloud_fp16' | 'local_int8' | 'local_int4'
    base_model : HuggingFace model ID or local path
    lora_path : path to LoRA safetensors (optional)
    controlnet_path : path to ControlNet weights directory (optional)
    ip_adapter_path : path to IP-Adapter weights directory (optional)
    load_controlnet : if False, skip ControlNet (faster for prompt-only mode)

    Returns
    -------
    QuantizedPipeline ready for inference
    """
    if mode == "auto":
        mode = detect_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    logger.info(f"Loading pipeline — mode={mode}, device={device}")
    logger.info(f"VRAM available: {get_available_vram_gb():.1f} GB / "
                f"{get_total_vram_gb():.1f} GB total")

    # ── VRAM pre-checks ──
    vram = get_total_vram_gb()
    if mode == "cloud_fp16" and vram < 13.0 and device == "cuda":
        logger.warning(
            f"cloud_fp16 needs ~14 GB VRAM but only {vram:.1f} GB detected. "
            "Auto-downgrading to local_int8."
        )
        mode = "local_int8"

    if mode == "local_int8" and vram < 6.0 and device == "cuda":
        logger.warning(
            f"local_int8 needs ~8 GB VRAM but only {vram:.1f} GB detected. "
            "Auto-downgrading to local_int4."
        )
        mode = "local_int4"

    # ── Load components ──
    logger.info("Loading tokenizers...")
    tok1 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2")

    logger.info("Loading text encoders...")
    te1, te2 = _load_text_encoders(base_model, mode, device)

    logger.info("Loading VAE...")
    vae = _load_vae(base_model, device)

    logger.info("Loading UNet...")
    unet = _load_unet_quantized(base_model, mode, device)

    # ── ControlNet ──
    controlnet = None
    if load_controlnet and controlnet_path and Path(controlnet_path).exists():
        logger.info(f"Loading ControlNet from {controlnet_path}...")
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=dtype
        ).to(device)
        controlnet.eval()
        logger.info("ControlNet loaded in fp16")
    elif load_controlnet and controlnet_path:
        logger.warning(f"ControlNet path not found: {controlnet_path}. Skipping.")

    # ── Assemble base pipeline ──
    logger.info("Assembling base pipeline...")
    pipe_kwargs = dict(
        pretrained_model_name_or_path=base_model,
        vae=vae,
        text_encoder=te1,
        text_encoder_2=te2,
        tokenizer=tok1,
        tokenizer_2=tok2,
        unet=unet,
        torch_dtype=dtype,
    )

    if controlnet:
        base_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            **pipe_kwargs, controlnet=controlnet
        )
    else:
        base_pipe = StableDiffusionXLPipeline.from_pretrained(**pipe_kwargs)

    # Better scheduler for fewer steps
    base_pipe.scheduler = UniPCMultistepScheduler.from_config(
        base_pipe.scheduler.config
    )

    if device == "cuda" and mode == "cloud_fp16":
        base_pipe = base_pipe.to(device)

    # ── Apply optimizations ──
    _apply_memory_optimizations(base_pipe, mode)

    # ── LoRA weights ──
    if lora_path and Path(lora_path).exists():
        logger.info(f"Loading LoRA weights from {lora_path}...")
        try:
            base_pipe.load_lora_weights(lora_path)
            base_pipe.fuse_lora(lora_scale=0.8)
            logger.info("LoRA weights fused successfully")
        except Exception as e:
            logger.warning(f"LoRA loading failed: {e}. Continuing without LoRA.")
    else:
        logger.info("No LoRA path provided or path not found — using base SDXL")

    # ── IP-Adapter ──
    if ip_adapter_path and Path(ip_adapter_path).exists():
        logger.info(f"Loading IP-Adapter from {ip_adapter_path}...")
        try:
            base_pipe.load_ip_adapter(
                ip_adapter_path,
                subfolder="models",
                weight_name="ip-adapter_sdxl.bin",
            )
            base_pipe.set_ip_adapter_scale(0.6)
            logger.info("IP-Adapter loaded")
        except Exception as e:
            logger.warning(f"IP-Adapter loading failed: {e}. Skipping.")
    else:
        logger.info("No IP-Adapter path — running without reference conditioning")

    # ── img2img pipeline (shares loaded components) ──
    logger.info("Assembling img2img pipeline...")
    img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=vae,
        text_encoder=te1,
        text_encoder_2=te2,
        tokenizer=tok1,
        tokenizer_2=tok2,
        unet=base_pipe.unet,   # shared UNet — no extra VRAM
        scheduler=UniPCMultistepScheduler.from_config(base_pipe.scheduler.config),
    )
    if device == "cuda" and mode == "cloud_fp16":
        img2img_pipe = img2img_pipe.to(device)
    _apply_memory_optimizations(img2img_pipe, mode)

    # ── Log final VRAM ──
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"VRAM allocated after loading: {allocated:.2f} GB")

    return QuantizedPipeline(
        mode=mode,
        base_pipe=base_pipe,
        img2img_pipe=img2img_pipe,
        controlnet=controlnet,
        device=device,
        default_steps=DEFAULT_STEPS[mode],
    )


# ─── Standalone usage example ─────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    logger.info(f"Testing pipeline load with mode={mode}")
    pipe = load_pipeline_quantized(mode=mode)
    logger.success(f"Pipeline loaded successfully: mode={pipe.mode}, device={pipe.device}")
    logger.info(f"VRAM used: {pipe.vram_used_gb:.2f} GB")
