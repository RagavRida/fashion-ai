#!/usr/bin/env python3
"""
api/schemas.py
---------------
Pydantic request/response models for all API endpoints.
"""

import base64
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, validator


# ─── Enums ────────────────────────────────────────────────────────────────────
class Difficulty(str, Enum):
    easy = "Easy"
    medium = "Medium"
    hard = "Hard"


# ─── Shared ──────────────────────────────────────────────────────────────────
class ImageResponse(BaseModel):
    """A single generated image as base64."""
    image_b64: str = Field(..., description="Base64-encoded JPEG image")
    index: int = Field(..., description="Position in generation batch")
    scores: Optional[dict] = Field(None, description="Ranking scores (CLIP, mask, edge, aesthetic)")

    @classmethod
    def from_pil(cls, image, index: int = 0, scores: dict = None) -> "ImageResponse":
        import io
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return cls(image_b64=b64, index=index, scores=scores)


# ─── /generate ────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="Fashion generation prompt",
        example="Upcycle a denim jacket into a cropped streetwear jacket with patches",
        min_length=5,
        max_length=500,
    )
    n_images: int = Field(default=4, ge=1, le=8, description="Number of images to return")
    style_reference_b64: Optional[str] = Field(
        None, description="Optional base64 style reference image"
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class GenerateResponse(BaseModel):
    images: list[ImageResponse]
    prompt_used: str
    generation_time_secs: float


# ─── /redesign ────────────────────────────────────────────────────────────────
class RedesignRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 input garment image (JPEG/PNG)")
    n_images: int = Field(default=4, ge=1, le=8)

    @validator("image_b64")
    def validate_base64(cls, v):
        try:
            base64.b64decode(v)
        except Exception:
            raise ValueError("image_b64 must be valid base64")
        return v


class RedesignResponse(BaseModel):
    images: list[ImageResponse]
    auto_prompts: list[str] = Field(
        default_factory=list,
        description="Auto-generated design variation descriptions"
    )
    generation_time_secs: float


# ─── /redesign_prompt ─────────────────────────────────────────────────────────
class RedesignPromptRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 input garment image")
    prompt: str = Field(
        ...,
        description="Redesign instruction",
        example="Convert into a formal blazer with gold buttons",
        min_length=3,
        max_length=500,
    )
    n_images: int = Field(default=4, ge=1, le=8)
    reference_style_b64: Optional[str] = Field(
        None, description="Optional style reference image"
    )


class RedesignPromptResponse(BaseModel):
    images: list[ImageResponse]
    prompt_used: str
    generation_time_secs: float


# ─── /refine ──────────────────────────────────────────────────────────────────
class RefineRequest(BaseModel):
    previous_image_b64: str = Field(..., description="Base64 image to refine")
    refinement_prompt: str = Field(
        ...,
        description="Refinement instruction",
        example="Make the sleeves shorter and add a floral embroidery pattern",
        min_length=3,
        max_length=500,
    )
    original_prompt: Optional[str] = Field(
        None, description="Original generation prompt (for context)"
    )
    n_images: int = Field(default=4, ge=1, le=8)


class RefineResponse(BaseModel):
    images: list[ImageResponse]
    refinement_applied: str
    generation_time_secs: float


# ─── /diy_guide ───────────────────────────────────────────────────────────────
class DIYStep(BaseModel):
    step: int
    instruction: str
    tip: Optional[str] = None


class DIYGuideRequest(BaseModel):
    garment_category: str = Field(
        ...,
        description="Category of garment",
        example="denim jacket",
    )
    edits_applied: list[str] = Field(
        default_factory=list,
        description="List of design changes made",
        example=["cropped to waist", "added patches on elbows", "distressed sleeves"],
    )
    style_description: Optional[str] = Field(
        None,
        description="Style context from the design",
        example="streetwear, urban, upcycled aesthetic",
    )
    difficulty_target: Difficulty = Field(default=Difficulty.medium)
    final_image_b64: Optional[str] = Field(
        None, description="Optional base64 of final generated design"
    )


class DIYGuideResponse(BaseModel):
    title: str
    garment_category: str
    edits_summary: str
    materials: list[str]
    tools: list[str]
    steps: list[DIYStep]
    estimated_time: str
    difficulty: str
    safety_tips: list[str]
    budget_tips: list[str]
    sustainability_benefits: list[str]


# ─── Health ──────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    models_loaded: dict = Field(default_factory=dict)
    cuda_available: bool = False
    gpu_name: Optional[str] = None


# ─── Error ───────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
