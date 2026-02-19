# Local Deployment Guide — Fashion Reuse Studio

Run Fashion Reuse Studio on a **consumer GPU (6–12 GB VRAM)** using quantized models.

---

## GPU Requirements by Mode

| Mode | Min VRAM | Best for | Quality |
|------|----------|----------|---------|
| `cloud_fp16` | 14+ GB | A100, H100, RTX 3090/4090 | ⭐⭐⭐ Best |
| `local_int8` | 8–12 GB | RTX 3070/4070, A10G | ⭐⭐ Good |
| `local_int4` | 6–8 GB | RTX 3060, 2060S, M2 Max | ⭐ Acceptable |

> **`auto` mode** (default) detects your VRAM and picks the best mode automatically.

---

## 1. Install Dependencies

```bash
# Activate your environment first
conda activate fashion-ai

# Core (if not already installed)
pip install -r requirements.txt

# bitsandbytes — required for int8/int4 modes
pip install bitsandbytes>=0.43.0

# xformers — for memory-efficient attention
pip install xformers>=0.0.23

# Verify bitsandbytes CUDA works
python -c "import bitsandbytes as bnb; print(bnb.cuda_setup.get_compute_capability())"
```

---

## 2. Quick Start: Auto Mode (Recommended)

```bash
# Auto-selects mode based on your GPU VRAM
python deployment/local_runner.py \
    --mode auto \
    --prompt "Upcycle this denim jacket into a cropped jacket with patches" \
    --output outputs/
```

---

## 3. All Generation Modes

### A — Prompt Only (Text → Fashion)
```bash
python deployment/local_runner.py \
    --mode local_int8 \
    --prompt "Transform a plain white shirt into a bohemian crop top" \
    --n_images 4 \
    --output outputs/
```

### B — Image Redesign (Garment → Redesigned)
```bash
python deployment/local_runner.py \
    --mode local_int8 \
    --image inputs/jacket.jpg \
    --output outputs/
```

### C — Image + Prompt (Best Quality)
```bash
python deployment/local_runner.py \
    --mode local_int8 \
    --image inputs/jacket.jpg \
    --prompt "Convert into a bohemian crop top with floral embroidery" \
    --output outputs/
```

### D — Refinement Loop
```bash
# Give it a previous output image and a new instruction
python deployment/local_runner.py \
    --mode local_int8 \
    --refine \
    --image outputs/redesign_local_int8_20240219_123456_0.jpg \
    --prompt "Make sleeves shorter and add distressed texture" \
    --output outputs/refined/
```

### With Your Trained Weights
```bash
python deployment/local_runner.py \
    --mode local_int8 \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora checkpoints/fashion_lora/fashion_lora.safetensors \
    --controlnet checkpoints/fashion_controlnet/ \
    --ip_adapter checkpoints/fashion_ip_adapter/ \
    --image inputs/jacket.jpg \
    --prompt "Cropped streetwear jacket with embroidered patches" \
    --output outputs/
```

---

## 4. Merge LoRA for Faster Inference

Merging LoRA into the base model eliminates LoRA loading overhead (~0.5s per run):

```bash
python deployment/export_lora_merge.py \
    --base_model stabilityai/stable-diffusion-xl-base-1.0 \
    --lora checkpoints/fashion_lora/fashion_lora.safetensors \
    --output checkpoints/merged_model/ \
    --lora_scale 0.85 \
    --verify

# Use the merged model (no --lora flag needed)
python deployment/local_runner.py \
    --model checkpoints/merged_model/ \
    --mode local_int8 \
    --prompt "Upcycled denim jacket" \
    --output outputs/
```

---

## 5. Benchmark Your GPU

Runs all 3 modes and outputs a comparison table:

```bash
# Full benchmark (needs enough VRAM for cloud_fp16)
python deployment/benchmark.py \
    --modes cloud_fp16 local_int8 local_int4 \
    --n_runs 3

# Only local modes (recommended for 8–12 GB GPUs)
python deployment/benchmark.py \
    --modes local_int8 local_int4 \
    --n_runs 3 \
    --output outputs/bench_report.json
```

Example output:

```
═══════════════════════════════════════════════════════════════════
  BENCHMARK COMPARISON TABLE
═══════════════════════════════════════════════════════════════════
Mode               Load(s)    Gen/run(s)   s/img    VRAM pk    CLIP
───────────────────────────────────────────────────────────────────
cloud_fp16         18.2       38.1         9.5      13.8       0.312
local_int8         22.4       52.3         13.1     7.9        0.303
local_int4         24.1       44.6         11.2     5.4        0.281
═══════════════════════════════════════════════════════════════════
```

---

## 6. Speed / Quality Trade-offs

| Mode | s/image | VRAM | CLIP score | Notes |
|------|---------|------|------------|-------|
| `cloud_fp16` | ~10s | 14 GB | 0.31+ | Full precision, best outputs |
| `local_int8` | ~13s | 8 GB | ~0.30 | Minimal quality loss |
| `local_int4` | ~11s | 5.5 GB | ~0.28 | Visible softness, acceptable |

> **Tip**: Increase `--steps` by 5–10 in `local_int4` to partially recover quality.
> The default is 20 steps — try `--steps 30` for better results.

---

## 7. Troubleshooting

### bitsandbytes CUDA error
```
RuntimeError: CUDA error: no kernel image is available for execution on device
```
**Fix**: Install the matching bitsandbytes CUDA version:
```bash
pip install bitsandbytes --index-url https://huggingface.github.io/bitsandbytes-wheels/
# or for CUDA 12.x
pip install bitsandbytes>=0.43.0
```

### OOM (Out of Memory)
```
torch.cuda.OutOfMemoryError
```
**Fix options**:
1. Switch to a lower mode: `--mode local_int4`
2. Reduce image count: `--n_images 1`
3. Enable CPU offload — set in `configs/inference.yaml`: `cpu_offload: true`
4. Add `--no_controlnet` flag to skip ControlNet

### xformers Not Available
xformers automatically falls back to **attention slicing** — generation still works, just ~15% slower. Install xformers for your CUDA version:
```bash
pip install xformers==0.0.23+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

### bitsandbytes Not Installed (CPU-only machine)
The pipeline automatically falls back to fp16 with a warning. CPU generation is very slow (~5 min/image).

### HuggingFace Model Download Fails
```bash
# Set HF token if model is gated
export HUGGING_FACE_HUB_TOKEN=hf_...
huggingface-cli login
```

---

## 8. Config Reference (`configs/inference.yaml`)

Key settings relevant to local deployment:

```yaml
generation:
  num_inference_steps: 25    # Reduce to 15 for local_int4
  guidance_scale: 7.5
  num_images_per_prompt: 8
  top_k_return: 4

local_deployment:
  default_mode: auto          # auto | cloud_fp16 | local_int8 | local_int4
  cpu_offload: false          # Enable for GPUs with < 6 GB VRAM (very slow)
  attention_slicing: auto     # auto | true | false
  xformers: true

diy_guide:
  llm_provider: local         # Use local Ollama (no API key needed)
  local_model: llama3.2:3b    # Fast local model
```

---

## 9. Verified GPU Compatibility

| GPU | VRAM | Recommended Mode | Notes |
|-----|------|-----------------|-------|
| RTX 4090 | 24 GB | cloud_fp16 | Fastest consumer GPU |
| RTX 3090 / 4080 | 24 / 16 GB | cloud_fp16 | ✅ Fully tested |
| RTX 4070 Ti | 12 GB | local_int8 | ✅ Recommended for int8 |
| RTX 3070 / 4070 | 8 GB | local_int8 | ✅ Works well |
| RTX 3060 | 12 GB | local_int8 | ✅ Works well |
| RTX 2060 Super | 8 GB | local_int8 | ✅ With xformers |
| GTX 1080 Ti | 11 GB | local_int8 | ⚠️ Slow, no int4 support |
| M2 Max / M3 Max | 30–96 GB | cloud_fp16 | Use MPS device |
| A10G (cloud) | 24 GB | cloud_fp16 | AWS/GCP standard |
| A100 (cloud) | 40/80 GB | cloud_fp16 | Best for batch training |
