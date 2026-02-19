# Fashion Reuse Studio 🌿♻️

> **Production-grade AI fashion upcycling system** — Generate, redesign, and refine garments from text prompts or uploaded images, then get household DIY instructions to recreate designs in real life.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

---

## ✨ System Capabilities

| Feature | Description |
|---------|-------------|
| **Prompt → Fashion** | Generate 8 garment designs from a text description |
| **Image → Redesign** | Upload any garment and get 8 AI upcycled variations |
| **Image + Prompt** | Combine reference image with specific redesign instructions |
| **Refinement Loop** | Chat-style iterative refinement: "make sleeves shorter", "add embroidery" |
| **DIY Guide** | LLM-generated household upcycling instructions for any AI design |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Fashion Reuse Studio                         │
│                                                                     │
│  Frontend (React)           API (FastAPI)        Models              │
│  ┌─────────────────┐        ┌──────────────┐    ┌─────────────────┐ │
│  │  Mode Tabs      │──────▶ │ /generate    │───▶│ SDXL + LoRA     │ │
│  │  Upload Zone    │        │ /redesign    │───▶│ ControlNet      │ │
│  │  Gallery /4up   │◀────── │ /redesign_p  │───▶│ IP-Adapter      │ │
│  │  Refine Chat    │──────▶ │ /refine      │    └────────┬────────┘ │
│  │  DIY Panel      │◀────── │ /diy_guide   │            │          │
│  └─────────────────┘        └──────────────┘    ┌─────────────────┐ │
│                                                  │ Ranker (CLIP,   │ │
│  Dataset Pipeline                                │ Mask, Edge,     │ │
│  ┌─────────────────────────────────────────┐     │ LPIPS)          │ │
│  │ DeepFashion → preprocess → edges →      │     └────────┬────────┘ │
│  │ masks → prompts → metadata.jsonl        │              │          │
│  └─────────────────────────────────────────┘    ┌─────────────────┐ │
│                                                  │ DIY Guide LLM   │ │
│  Training (4-Stage)                              │ (GPT-4o/Claude  │ │
│  ┌─────────────────────────────────────────┐     │  /Ollama)        │ │
│  │ 1. GarmentUNet (segmentation)           │     └─────────────────┘ │
│  │ 2. SDXL LoRA (DeepFashion fine-tune)    │                         │
│  │ 3. ControlNet (Canny edge conditioning) │                         │
│  │ 4. IP-Adapter (reference conditioning)  │                         │
│  └─────────────────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions
- **LoRA first** (Phase 2) to learn fashion-domain priors, then ControlNet (Phase 3) loads LoRA weights
- **CFG-scale conditioning dropout** during ControlNet training for flexible inference
- **Multi-metric ranker**: CLIP relevance + mask IoU + edge correlation + aesthetic score − LPIPS penalty
- **Fallback DIY guide** works without any LLM API key (pre-baked instructions)

---

## 📁 Project Structure

```
fashion-ai/
├── configs/              # All YAML configuration files
│   ├── dataset.yaml
│   ├── train_lora.yaml
│   ├── train_controlnet.yaml
│   ├── train_ip_adapter.yaml
│   ├── train_segmentation.yaml
│   └── inference.yaml
├── dataset_builder/      # Dataset processing pipeline
│   ├── download_deepfashion.py
│   ├── preprocess.py
│   ├── build_prompts.py
│   ├── build_edges.py
│   ├── build_masks.py
│   └── export_jsonl.py
├── models/               # Model architecture code
│   ├── segmentation/unet.py      # GarmentUNet
│   └── ranking/ranker.py         # CandidateRanker
├── training/             # Training scripts + pipeline
│   ├── train_segmentation.py
│   ├── train_lora.py
│   ├── train_controlnet.py
│   ├── train_ip_adapter.py
│   └── automated_train_pipeline.sh
├── inference/            # Inference engine
│   ├── pipeline.py       # FashionPipeline (all modes)
│   └── diy_guide.py      # DIYGuideGenerator
├── api/                  # FastAPI backend
│   ├── app.py
│   └── schemas.py
├── frontend/             # React UI
│   ├── public/index.html
│   └── src/
│       ├── App.jsx
│       ├── index.js
│       └── index.css
├── evaluation/
│   └── evaluate.py       # FID, CLIP, LPIPS, IoU, DIY eval
├── data/                 # (gitignored) data directory
├── checkpoints/          # (gitignored) trained model weights
├── outputs/              # (gitignored) generated images
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and setup
git clone <your-repo>
cd fashion-ai

# Create conda env (Python 3.10+, CUDA 12.1)
conda create -n fashion-ai python=3.10 -y
conda activate fashion-ai
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install Python dependencies
pip install -r requirements.txt

# Install accelerate + configure for your GPU
pip install accelerate
accelerate config    # Follow prompts for your GPU setup

# (Optional) Login to W&B for training logs
wandb login

# (Optional) Set up LLM API key for DIY guides
export OPENAI_API_KEY=sk-...
# or use local Ollama: set diy_guide.llm_provider: local in inference.yaml
```

### 2. Dataset Preparation

```bash
# Option A: Download DeepFashion from HuggingFace
python dataset_builder/download_deepfashion.py --source huggingface --output_dir data/raw

# One-shot preprocessing (all stages)
bash training/automated_train_pipeline.sh --prep-dataset --end-stage 0
```

### 3. Training (4-Stage Pipeline)

```bash
# Run all 4 training stages sequentially (GPU required)
bash training/automated_train_pipeline.sh --gpu-ids 0,1,2,3

# Or run individual stages:
# Stage 1: Segmentation CNN (~2h on A100)
accelerate launch training/train_segmentation.py --config configs/train_segmentation.yaml

# Stage 2: LoRA fine-tuning (~6h on A100)
accelerate launch training/train_lora.py --config configs/train_lora.yaml

# Stage 3: ControlNet (~8h on A100)
accelerate launch training/train_controlnet.py --config configs/train_controlnet.yaml

# Stage 4: IP-Adapter (~4h on A100)
accelerate launch training/train_ip_adapter.py --config configs/train_ip_adapter.yaml

# Resume from checkpoint (skips completed stages automatically)
bash training/automated_train_pipeline.sh --start-stage 3
```

### 4. Start the API Server

```bash
# Using pre-trained/public models (without fine-tuning)
python api/app.py
# or: uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# API docs available at: http://localhost:8000/docs
```

### 5. Start the Frontend

```bash
cd frontend
npm install
npm start
# Open: http://localhost:3000
```

---

## 🔌 API Reference

### `GET /health`
System health check and GPU info.

### `POST /generate`
```json
{
  "prompt": "Upcycle a denim jacket into a cropped streetwear jacket with patches",
  "n_images": 4
}
```
Returns top 4 ranked images (base64 JPEG).

### `POST /redesign`
```json
{
  "image_b64": "<base64 garment image>",
  "n_images": 4
}
```
Returns 4 AI-generated redesign variations.

### `POST /redesign_prompt`
```json
{
  "image_b64": "<base64 garment image>",
  "prompt": "Convert into a formal blazer with gold buttons",
  "n_images": 4
}
```

### `POST /refine`
```json
{
  "previous_image_b64": "<base64 image>",
  "refinement_prompt": "Make sleeves shorter, add embroidery",
  "original_prompt": "denim jacket",
  "n_images": 4
}
```

### `POST /diy_guide`
```json
{
  "garment_category": "denim jacket",
  "edits_applied": ["cropped to waist", "added patches"],
  "style_description": "streetwear, urban",
  "difficulty_target": "Medium"
}
```
Returns step-by-step DIY instructions with materials, tools, steps, safety and budget tips.

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| FID | < 30 | vs DeepFashion2 val set |
| CLIP Score | > 0.28 | text-image alignment |
| LPIPS Diversity | > 0.3 | within each generation batch |
| Segmentation IoU | > 0.82 | on fashion dataset |
| DIY Guide Steps | ≥ 6 | all required fields present |
| Inference Speed | < 30s / 4 images | A100 GPU |

### Run Evaluation
```bash
python evaluation/evaluate.py \
  --config configs/inference.yaml \
  --eval_dir outputs/generated_samples \
  --real_dir data/processed/images_512 \
  --n_samples 200
```

---

## 🔧 Configuration

Key configs in `configs/inference.yaml`:

```yaml
models:
  base_model: stabilityai/stable-diffusion-xl-base-1.0
  lora_weights: checkpoints/fashion_lora/fashion_lora.safetensors
  controlnet_weights: checkpoints/fashion_controlnet/
  
diy_guide:
  llm_provider: openai   # openai | anthropic | local (Ollama)
  openai_model: gpt-4o

generation:
  num_inference_steps: 50
  guidance_scale: 7.5
  num_images_per_prompt: 8
  top_k_return: 4
```

---

## 📚 Documentation

- `TRAINING.md` — Detailed training guide for each stage
- `GPU_SETUP.md` — Multi-GPU setup, memory optimization
- `DEMO.md` — Interactive demo and API examples

---

## 🌱 Sustainability

Fashion Reuse Studio is designed to **reduce textile waste** by:
- Making garment upcycling accessible to everyone
- Generating household-friendly DIY instructions (no industrial equipment)
- Providing budget-conscious material alternatives
- Demonstrating sustainability benefits for each transformation

---

## License
MIT License — see [LICENSE](LICENSE)
