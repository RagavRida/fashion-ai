#!/usr/bin/env python3
"""
inference/diy_guide.py
-----------------------
DIY Household Instructions Generator.

Given a garment description + applied edits, generates:
  - Title
  - Materials needed
  - Tools needed
  - Step-by-step instructions
  - Estimated time
  - Difficulty rating
  - Safety + budget tips
  - Sustainability benefits

Uses OpenAI / Anthropic / local Ollama based on config.

Usage:
    from inference.diy_guide import DIYGuideGenerator

    gen = DIYGuideGenerator.from_config("configs/inference.yaml")
    guide = gen.generate(
        garment_category="denim jacket",
        edits_applied=["cropped", "added patches", "distressed sleeves"],
        style_description="streetwear aesthetic, urban, upcycled",
    )
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Output Schema ────────────────────────────────────────────────────────────
@dataclass
class DIYStep:
    step_number: int
    instruction: str
    tip: Optional[str] = None


@dataclass
class DIYGuide:
    title: str
    garment_category: str
    edits_summary: str
    materials: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    steps: list[DIYStep] = field(default_factory=list)
    estimated_time: str = "2-4 hours"
    difficulty: str = "Medium"
    safety_tips: list[str] = field(default_factory=list)
    budget_tips: list[str] = field(default_factory=list)
    sustainability_benefits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "garment_category": self.garment_category,
            "edits_summary": self.edits_summary,
            "materials": self.materials,
            "tools": self.tools,
            "steps": [
                {
                    "step": s.step_number,
                    "instruction": s.instruction,
                    "tip": s.tip,
                }
                for s in self.steps
            ],
            "estimated_time": self.estimated_time,
            "difficulty": self.difficulty,
            "safety_tips": self.safety_tips,
            "budget_tips": self.budget_tips,
            "sustainability_benefits": self.sustainability_benefits,
        }


# ─── Prompt Template ─────────────────────────────────────────────────────────
DIY_SYSTEM_PROMPT = """You are an expert fashion upcycling instructor and DIY guide writer.
You create clear, practical, household-friendly guides for upcycling fashion items.
Your guides are suitable for beginners and require only household tools — no industrial machines.
Always be encouraging, safety-conscious, and sustainability-focused.
Output must be valid JSON exactly matching the required schema."""

DIY_USER_TEMPLATE = """
Generate a detailed DIY household upcycling guide for this fashion transformation:

**Garment Category:** {garment_category}
**Edits Applied in Design:** {edits_applied}
**Style Description:** {style_description}
**Difficulty Target:** {difficulty_target}

Return a JSON object with EXACTLY this structure:
{{
  "title": "Short creative title for this upcycle project (max 10 words)",
  "edits_summary": "One sentence summarizing what was changed",
  "materials": ["list", "of", "materials", "needed"],
  "tools": ["scissors", "needle and thread", etc.],
  "steps": [
    {{
      "step": 1,
      "instruction": "Detailed step instruction (2-3 sentences). Include exact measurements where applicable.",
      "tip": "Optional helpful tip for this step or null"
    }}
  ],
  "estimated_time": "e.g. 2-3 hours",
  "difficulty": "Easy | Medium | Hard",
  "safety_tips": ["Always cut away from your body", etc.],
  "budget_tips": ["Use fabric from old jeans for patches", etc.],
  "sustainability_benefits": ["Diverts clothing from landfill", etc.]
}}

Requirements:
- Include 6-10 detailed steps
- Materials must be household-accessible (no industrial equipment)
- Include exact measurements and techniques where applicable
- Safety tips must mention needle/scissor safety
- Budget tips must use free/cheap alternatives
- Return ONLY the JSON object, no markdown or extra text
"""


# ─── Generator ────────────────────────────────────────────────────────────────
class DIYGuideGenerator:
    def __init__(self, config: dict):
        self.config = config
        self.diy_config = config.get("diy_guide", {})
        self.provider = self.diy_config.get("llm_provider", "openai")
        self.max_tokens = self.diy_config.get("max_tokens", 1500)
        self.temperature = self.diy_config.get("temperature", 0.3)

    @classmethod
    def from_config(cls, config_path: str) -> "DIYGuideGenerator":
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _call_openai(self, prompt: str) -> str:
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=self.diy_config.get("openai_model", "gpt-4o"),
            messages=[
                {"role": "system", "content": DIY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.diy_config.get("anthropic_model", "claude-3-5-sonnet-20241022"),
            max_tokens=self.max_tokens,
            system=DIY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    def _call_local(self, prompt: str) -> str:
        import httpx
        endpoint = self.diy_config.get("local_endpoint", "http://localhost:11434/api/generate")
        model = self.diy_config.get("local_model", "llama3.2")
        response = httpx.post(
            endpoint,
            json={
                "model": model,
                "prompt": f"{DIY_SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "format": "json",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["response"]

    def _llm_call(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "local":
            return self._call_local(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _fallback_guide(
        self,
        garment_category: str,
        edits_applied: list[str],
        style_description: str,
    ) -> dict:
        """Fallback guide when LLM is unavailable."""
        edits_str = ", ".join(edits_applied)
        return {
            "title": f"Upcycled {garment_category.title()} Transformation",
            "edits_summary": f"Transform your {garment_category} with: {edits_str}",
            "materials": [
                "Existing garment",
                "Matching thread",
                "Fabric scissors",
                "Measuring tape",
                "Chalk or fabric marker",
                "Iron and ironing board",
                "Pins",
            ],
            "tools": [
                "Sharp fabric scissors",
                "Needle (size 14 universal)",
                "Sewing machine or hand needle",
                "Steam iron",
                "Seam ripper",
            ],
            "steps": [
                {
                    "step": 1,
                    "instruction": "Lay the garment flat on a clean surface. Mark all areas to be modified with chalk or a fabric marker. Double-check measurements before cutting.",
                    "tip": "Always measure twice, cut once.",
                },
                {
                    "step": 2,
                    "instruction": "Cut along the marked lines using sharp fabric scissors. Cut 1.5cm (0.6 inches) beyond your finished line for seam allowance.",
                    "tip": "Keep scissors clean and sharp for clean cuts.",
                },
                {
                    "step": 3,
                    "instruction": "To prevent fraying, fold the raw edges over by 0.5cm and press with an iron, then fold again by 1cm and pin in place.",
                    "tip": "Use fabric glue as an alternative to sewing for quick hems.",
                },
                {
                    "step": 4,
                    "instruction": "Stitch the hem using a straight stitch 0.2cm from the fold. Use matching thread color for an invisible finish.",
                    "tip": "Practice your stitch on a scrap fabric first.",
                },
                {
                    "step": 5,
                    "instruction": "Add patches, embellishments, or structural changes per your design. Pin all additions in place before permanently attaching.",
                    "tip": "Use iron-on adhesive for patches as a quick alternative.",
                },
                {
                    "step": 6,
                    "instruction": "Press the finished garment with a steam iron to set all seams and remove wrinkles. Check all seams are secure.",
                    "tip": "Use a damp cloth between iron and fabric for delicate materials.",
                },
            ],
            "estimated_time": "2-4 hours",
            "difficulty": "Medium",
            "safety_tips": [
                "Always cut away from your body",
                "Keep needles in a dedicated needle cushion",
                "Use a thimble when hand-sewing",
                "Keep iron away from flammable materials",
                "Work in a well-lit area",
            ],
            "budget_tips": [
                "Use old jeans or shirts for patch material",
                "Fabric glue is cheaper than sewing for small fixes",
                "Buy thread in bulk — it keeps forever",
                "Thrift stores often sell buttons, zippers, and trim cheaply",
            ],
            "sustainability_benefits": [
                "Extends the lifespan of existing clothing",
                "Diverts textile waste from landfill",
                "Reduces demand for new manufacturing",
                "Customized fit reduces future returns and waste",
            ],
        }

    def generate(
        self,
        garment_category: str,
        edits_applied: list[str],
        style_description: str = "",
        difficulty_target: str = "Medium",
    ) -> DIYGuide:
        """
        Generate a DIY household upcycling guide.

        Args:
            garment_category: Type of garment (e.g., "denim jacket")
            edits_applied: List of edits made in the design (e.g., ["cropped", "added patches"])
            style_description: Style context (e.g., "streetwear, urban")
            difficulty_target: Target difficulty level

        Returns:
            DIYGuide struct with all fields populated
        """
        edits_str = ", ".join(edits_applied) if edits_applied else "general redesign"
        prompt = DIY_USER_TEMPLATE.format(
            garment_category=garment_category,
            edits_applied=edits_str,
            style_description=style_description or "modern fashion",
            difficulty_target=difficulty_target,
        )

        logger.info(f"Generating DIY guide for: {garment_category} | edits: {edits_str}")

        try:
            raw_response = self._llm_call(prompt)
            data = json.loads(raw_response)
        except Exception as e:
            logger.warning(f"LLM call failed ({e}), using fallback guide")
            data = self._fallback_guide(garment_category, edits_applied, style_description)

        # Parse steps
        steps = []
        for step_data in data.get("steps", []):
            steps.append(DIYStep(
                step_number=step_data.get("step", len(steps) + 1),
                instruction=step_data.get("instruction", ""),
                tip=step_data.get("tip"),
            ))

        guide = DIYGuide(
            title=data.get("title", f"Upcycled {garment_category} Guide"),
            garment_category=garment_category,
            edits_summary=data.get("edits_summary", edits_str),
            materials=data.get("materials", []),
            tools=data.get("tools", []),
            steps=steps,
            estimated_time=data.get("estimated_time", "2-4 hours"),
            difficulty=data.get("difficulty", difficulty_target),
            safety_tips=data.get("safety_tips", []),
            budget_tips=data.get("budget_tips", []),
            sustainability_benefits=data.get("sustainability_benefits", []),
        )

        logger.success(
            f"DIY guide generated: '{guide.title}' | {len(guide.steps)} steps | {guide.difficulty}"
        )
        return guide
