#!/usr/bin/env python3
"""
api/app.py
-----------
Fashion Reuse Studio — FastAPI Backend

Endpoints:
  GET  /health                 → system health check
  POST /generate               → prompt-only generation
  POST /redesign               → image-only redesign
  POST /redesign_prompt        → image + prompt redesign
  POST /refine                 → refinement loop
  POST /diy_guide              → DIY household instructions

Start:
    python api/app.py
    # or
    uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import base64
import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from api.schemas import (
    DIYGuideRequest,
    DIYGuideResponse,
    DIYStep,
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ImageResponse,
    RefineRequest,
    RefineResponse,
    RedesignPromptRequest,
    RedesignPromptResponse,
    RedesignRequest,
    RedesignResponse,
)

# ─── Config ──────────────────────────────────────────────────────────────────
CONFIG_PATH = os.getenv("FASHION_CONFIG", "configs/inference.yaml")
APP_VERSION = "1.0.0"


# ─── Global State ─────────────────────────────────────────────────────────────
class AppState:
    pipeline = None
    diy_generator = None
    config = None


state = AppState()


def load_config() -> dict:
    config_path = Path(CONFIG_PATH)
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using defaults")
        return {
            "models": {"base_model": "stabilityai/stable-diffusion-xl-base-1.0"},
            "generation": {
                "resolution": 512,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "num_images_per_prompt": 8,
                "top_k_return": 4,
                "strength": 0.75,
                "inpaint_strength": 0.6,
                "negative_prompt": "blurry, low quality, distorted",
                "prompt_suffix": "high realism, professional fashion photography",
                "seed": None,
            },
            "ranking": {
                "clip_weight": 0.40,
                "mask_alignment_weight": 0.25,
                "edge_alignment_weight": 0.20,
                "aesthetic_weight": 0.15,
                "lpips_lower_bound": 0.3,
            },
            "edge_extraction": {"canny_low": 100, "canny_high": 200},
            "diy_guide": {
                "llm_provider": "openai",
                "openai_model": "gpt-4o",
                "max_tokens": 1500,
                "temperature": 0.3,
            },
        }
    with open(config_path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    logger.info("Fashion Reuse Studio API starting up...")
    state.config = load_config()

    # Lazy import to avoid long startup on cold machines
    try:
        from inference.pipeline import FashionPipeline
        state.pipeline = FashionPipeline(state.config)
        logger.success("FashionPipeline initialized (models load lazily on first request)")
    except Exception as e:
        logger.warning(f"Pipeline initialization failed: {e}. Will retry on first request.")

    try:
        from inference.diy_guide import DIYGuideGenerator
        state.diy_generator = DIYGuideGenerator(state.config)
        logger.success("DIYGuideGenerator initialized")
    except Exception as e:
        logger.warning(f"DIY generator initialization failed: {e}")

    yield

    logger.info("Shutting down Fashion Reuse Studio API")


# ─── App Instance ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fashion Reuse Studio API",
    description="CNN + Diffusion + ControlNet + IP-Adapter fashion upcycling system",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def b64_to_pil(b64_str: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        data = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid base64 image: {e}",
        )


def pil_list_to_response(images: list, prompt: str = "") -> list[ImageResponse]:
    return [ImageResponse.from_pil(img, index=i) for i, img in enumerate(images)]


def get_pipeline():
    """Get pipeline, initializing if needed."""
    if state.pipeline is None:
        config = state.config or load_config()
        from inference.pipeline import FashionPipeline
        state.pipeline = FashionPipeline(config)
    return state.pipeline


def get_diy_generator():
    """Get DIY generator, initializing if needed."""
    if state.diy_generator is None:
        config = state.config or load_config()
        from inference.diy_guide import DIYGuideGenerator
        state.diy_generator = DIYGuideGenerator(config)
    return state.diy_generator


# ─── Error Handler ────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            request_id=str(uuid.uuid4()),
        ).model_dump(),
    )


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """System health check with GPU info."""
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)

    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        models_loaded={
            "pipeline": state.pipeline is not None,
            "diy_generator": state.diy_generator is not None,
        },
        cuda_available=cuda_available,
        gpu_name=gpu_name,
    )


# ─── POST /generate ───────────────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse, tags=["Generation"])
async def generate(request: GenerateRequest):
    """
    Generate 4-8 fashion images from a text prompt.
    
    Example: 'Upcycle a denim jacket into a cropped streetwear jacket with patches'
    """
    logger.info(f"[/generate] prompt='{request.prompt[:80]}' n={request.n_images}")
    start = time.time()

    try:
        pipe = get_pipeline()
        images = pipe.generate(
            prompt=request.prompt,
            n_images=request.n_images * 2,  # generate 2x, return top-n
            top_k=request.n_images,
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    elapsed = round(time.time() - start, 2)
    logger.info(f"[/generate] completed in {elapsed}s, returned {len(images)} images")

    return GenerateResponse(
        images=pil_list_to_response(images, request.prompt),
        prompt_used=request.prompt,
        generation_time_secs=elapsed,
    )


# ─── POST /redesign ───────────────────────────────────────────────────────────
@app.post("/redesign", response_model=RedesignResponse, tags=["Generation"])
async def redesign(request: RedesignRequest):
    """
    Upload a garment image and receive 4-8 redesign variations.
    No prompt required — system auto-generates design variations.
    """
    logger.info(f"[/redesign] n_images={request.n_images}")
    start = time.time()

    garment_image = b64_to_pil(request.image_b64)

    auto_prompts = [
        "redesigned garment, upcycled streetwear, modern aesthetic",
        "eco-friendly redesign, sustainable fashion, creative variation",
        "formal style transformation, professional look, refined design",
        "bohemian upcycle, artistic pattern, free-spirited style",
    ]

    try:
        pipe = get_pipeline()
        images = pipe.redesign(
            garment_image=garment_image,
            prompt=auto_prompts[0],
            n_images=request.n_images * 2,
            top_k=request.n_images,
        )
    except Exception as e:
        logger.error(f"Redesign failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redesign failed: {e}")

    elapsed = round(time.time() - start, 2)
    return RedesignResponse(
        images=pil_list_to_response(images),
        auto_prompts=auto_prompts,
        generation_time_secs=elapsed,
    )


# ─── POST /redesign_prompt ────────────────────────────────────────────────────
@app.post("/redesign_prompt", response_model=RedesignPromptResponse, tags=["Generation"])
async def redesign_prompt(request: RedesignPromptRequest):
    """
    Upload a garment image + prompt. System redesigns the garment
    while preserving its identity and applying the prompt instructions.
    """
    logger.info(f"[/redesign_prompt] prompt='{request.prompt[:60]}'")
    start = time.time()

    garment_image = b64_to_pil(request.image_b64)

    try:
        pipe = get_pipeline()
        images = pipe.redesign(
            garment_image=garment_image,
            prompt=request.prompt,
            n_images=request.n_images * 2,
            top_k=request.n_images,
        )
    except Exception as e:
        logger.error(f"Redesign+prompt failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Redesign failed: {e}")

    elapsed = round(time.time() - start, 2)
    return RedesignPromptResponse(
        images=pil_list_to_response(images, request.prompt),
        prompt_used=request.prompt,
        generation_time_secs=elapsed,
    )


# ─── POST /refine ─────────────────────────────────────────────────────────────
@app.post("/refine", response_model=RefineResponse, tags=["Refinement"])
async def refine(request: RefineRequest):
    """
    Apply a refinement instruction to a previously generated image.
    
    Examples: 'make sleeves shorter', 'change to pastel colors', 'add pockets'
    The system uses inpainting to preserve identity while applying changes.
    """
    logger.info(f"[/refine] instruction='{request.refinement_prompt[:60]}'")
    start = time.time()

    prev_image = b64_to_pil(request.previous_image_b64)

    try:
        pipe = get_pipeline()
        images = pipe.refine(
            prev_image=prev_image,
            refinement_prompt=request.refinement_prompt,
            original_prompt=request.original_prompt or "",
            n_images=request.n_images * 2,
            top_k=request.n_images,
        )
    except Exception as e:
        logger.error(f"Refinement failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Refinement failed: {e}")

    elapsed = round(time.time() - start, 2)
    return RefineResponse(
        images=pil_list_to_response(images, request.refinement_prompt),
        refinement_applied=request.refinement_prompt,
        generation_time_secs=elapsed,
    )


# ─── POST /diy_guide ──────────────────────────────────────────────────────────
@app.post("/diy_guide", response_model=DIYGuideResponse, tags=["DIY"])
async def diy_guide(request: DIYGuideRequest):
    """
    Generate household-friendly DIY upcycling instructions for a design.
    
    Returns step-by-step instructions, materials, tools, difficulty rating,
    safety tips, and sustainability benefits.
    """
    logger.info(f"[/diy_guide] garment='{request.garment_category}' edits={request.edits_applied}")
    start = time.time()

    try:
        gen = get_diy_generator()
        guide = gen.generate(
            garment_category=request.garment_category,
            edits_applied=request.edits_applied,
            style_description=request.style_description or "",
            difficulty_target=request.difficulty_target.value,
        )
    except Exception as e:
        logger.error(f"DIY guide generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"DIY guide failed: {e}")

    guide_dict = guide.to_dict()

    return DIYGuideResponse(
        title=guide_dict["title"],
        garment_category=guide_dict["garment_category"],
        edits_summary=guide_dict["edits_summary"],
        materials=guide_dict["materials"],
        tools=guide_dict["tools"],
        steps=[DIYStep(**s) for s in guide_dict["steps"]],
        estimated_time=guide_dict["estimated_time"],
        difficulty=guide_dict["difficulty"],
        safety_tips=guide_dict["safety_tips"],
        budget_tips=guide_dict["budget_tips"],
        sustainability_benefits=guide_dict["sustainability_benefits"],
    )


# ─── POST /upload (multipart form) ────────────────────────────────────────────
@app.post("/upload_redesign", tags=["Generation"])
async def upload_redesign(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    n_images: int = Form(4),
):
    """
    Upload endpoint accepting multipart/form-data (alternative to base64).
    Accepts image file + optional prompt.
    """
    logger.info(f"[/upload_redesign] file={file.filename} prompt='{prompt}'")
    start = time.time()

    contents = await file.read()
    garment_image = Image.open(io.BytesIO(contents)).convert("RGB")

    try:
        pipe = get_pipeline()
        images = pipe.redesign(
            garment_image=garment_image,
            prompt=prompt or "upcycled fashion redesign, creative variation",
            n_images=n_images * 2,
            top_k=n_images,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = round(time.time() - start, 2)
    image_responses = [ImageResponse.from_pil(img, i) for i, img in enumerate(images)]

    return {
        "images": [img.model_dump() for img in image_responses],
        "prompt_used": prompt,
        "generation_time_secs": elapsed,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting Fashion Reuse Studio API on {host}:{port}")
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true",
        workers=1,  # Keep 1 worker to share GPU model in memory
    )
