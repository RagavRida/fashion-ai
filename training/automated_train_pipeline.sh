#!/usr/bin/env bash
# =============================================================================
# training/automated_train_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────
# Automated sequential training pipeline:
#   Stage 0: Dataset preprocessing (if not done)
#   Stage 1: Segmentation CNN training
#   Stage 2: LoRA fine-tuning (DeepFashion → fashion-aware diffusion)
#   Stage 3: ControlNet training (edge conditioning)
#   Stage 4: IP-Adapter training (reference consistency)
#
# Usage:
#   bash training/automated_train_pipeline.sh [--start-stage 1] [--gpu-ids 0,1,2,3]
#
# Requirements:
#   - accelerate configured: accelerate config
#   - CUDA available
#   - Dataset preprocessed and metadata.jsonl ready
#   - W&B logged in: wandb login
# =============================================================================

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/outputs/logs"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Defaults
START_STAGE=1
END_STAGE=4
GPU_IDS="0"
NUM_GPUS=1
DO_DATASET_PREP=false
DRY_RUN=false
SKIP_EXISTING=true

# ─── Parse CLI Args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage)    START_STAGE="$2"; shift 2 ;;
    --end-stage)      END_STAGE="$2"; shift 2 ;;
    --gpu-ids)        GPU_IDS="$2"; NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l); shift 2 ;;
    --prep-dataset)   DO_DATASET_PREP=true; shift ;;
    --dry-run)        DRY_RUN=true; shift ;;
    --no-skip)        SKIP_EXISTING=false; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# ─── Helpers ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log_section() {
  echo ""
  echo "============================================================"
  echo "  $*"
  echo "============================================================"
}

check_stage_done() {
  local sentinel="$1"
  [[ "$SKIP_EXISTING" == "true" && -f "$sentinel" ]]
}

mark_stage_done() {
  local sentinel="$1"
  touch "$sentinel"
}

run_accelerate() {
  local script="$1"; shift
  local extra_args=("$@")
  
  if [[ "$DRY_RUN" == "true" ]]; then
    log "[DRY RUN] Would run: accelerate launch --num_processes=$NUM_GPUS $script ${extra_args[*]}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$GPU_IDS" accelerate launch \
    --num_processes="$NUM_GPUS" \
    --mixed_precision=fp16 \
    "$script" \
    "${extra_args[@]}"
}

# ─── Setup ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
cd "$ROOT_DIR"

log_section "Fashion Reuse Studio — Automated Training Pipeline"
log "Root:        $ROOT_DIR"
log "GPU IDs:     $GPU_IDS ($NUM_GPUS GPU(s))"
log "Stages:      $START_STAGE → $END_STAGE"
log "Timestamp:   $TIMESTAMP"
log "Dry run:     $DRY_RUN"

# ─── Pre-flight checks ────────────────────────────────────────────────────────
log "Checking prerequisites..."

# Python
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" \
  || { echo "ERROR: CUDA not available"; exit 1; }
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
log "CUDA version: $CUDA_VERSION"

# Metadata
if [[ ! -f "data/processed/metadata.jsonl" ]]; then
  if [[ "$DO_DATASET_PREP" == "true" ]]; then
    log "metadata.jsonl not found. Running dataset preprocessing..."
  else
    echo "ERROR: data/processed/metadata.jsonl not found. Run:"
    echo "  bash training/automated_train_pipeline.sh --prep-dataset"
    exit 1
  fi
fi

# ─── Stage 0: Dataset Preprocessing (optional) ───────────────────────────────
if [[ "$DO_DATASET_PREP" == "true" ]]; then
  log_section "Stage 0: Dataset Preprocessing"
  SENTINEL="${CHECKPOINT_DIR}/.stage0_done"

  if check_stage_done "$SENTINEL"; then
    log "✓ Stage 0 already complete (sentinel found)"
  else
    log "Preprocessing raw images..."
    python3 dataset_builder/preprocess.py \
      --config configs/dataset.yaml \
      --input_dir data/raw \
      --output_dir data/processed \
      2>&1 | tee "${LOG_DIR}/stage0_preprocess_${TIMESTAMP}.log"

    log "Generating prompts..."
    python3 dataset_builder/build_prompts.py \
      --config configs/dataset.yaml \
      --metadata data/processed/metadata_raw.jsonl \
      2>&1 | tee "${LOG_DIR}/stage0_prompts_${TIMESTAMP}.log"

    log "Extracting edge maps..."
    python3 dataset_builder/build_edges.py \
      --config configs/dataset.yaml \
      --metadata data/processed/metadata_with_prompts.jsonl \
      --method canny \
      2>&1 | tee "${LOG_DIR}/stage0_edges_${TIMESTAMP}.log"

    log "Generating segmentation masks..."
    python3 dataset_builder/build_masks.py \
      --config configs/dataset.yaml \
      --metadata data/processed/metadata_with_edges.jsonl \
      --method grabcut \
      2>&1 | tee "${LOG_DIR}/stage0_masks_${TIMESTAMP}.log"

    log "Exporting final metadata.jsonl..."
    python3 dataset_builder/export_jsonl.py \
      --config configs/dataset.yaml \
      2>&1 | tee "${LOG_DIR}/stage0_export_${TIMESTAMP}.log"

    mark_stage_done "$SENTINEL"
    log "✓ Stage 0 complete"
  fi
fi

# ─── Stage 1: Segmentation CNN ────────────────────────────────────────────────
if [[ "$START_STAGE" -le 1 && "$END_STAGE" -ge 1 ]]; then
  log_section "Stage 1: Segmentation CNN Training"
  SENTINEL="${CHECKPOINT_DIR}/.stage1_done"
  SEG_OUTPUT="${CHECKPOINT_DIR}/segmentation/best_model.pth"

  if check_stage_done "$SENTINEL"; then
    log "✓ Stage 1 already complete (sentinel found)"
  else
    log "Training GarmentUNet..."
    run_accelerate training/train_segmentation.py \
      --config configs/train_segmentation.yaml \
      2>&1 | tee "${LOG_DIR}/stage1_segmentation_${TIMESTAMP}.log"

    # Verify output
    if [[ ! -f "$SEG_OUTPUT" ]]; then
      echo "WARNING: Segmentation checkpoint not found at $SEG_OUTPUT"
    else
      log "✓ Segmentation model saved: $SEG_OUTPUT"
    fi

    mark_stage_done "$SENTINEL"
    log "✓ Stage 1 complete"
  fi
fi

# ─── Stage 2: LoRA Fine-Tuning ────────────────────────────────────────────────
if [[ "$START_STAGE" -le 2 && "$END_STAGE" -ge 2 ]]; then
  log_section "Stage 2: LoRA Fine-Tuning (SDXL on DeepFashion)"
  SENTINEL="${CHECKPOINT_DIR}/.stage2_done"
  LORA_OUTPUT="${CHECKPOINT_DIR}/fashion_lora/fashion_lora.safetensors"

  if check_stage_done "$SENTINEL"; then
    log "✓ Stage 2 already complete (sentinel found)"
  else
    log "Fine-tuning diffusion model with LoRA..."
    run_accelerate training/train_lora.py \
      --config configs/train_lora.yaml \
      2>&1 | tee "${LOG_DIR}/stage2_lora_${TIMESTAMP}.log"

    if [[ ! -f "$LORA_OUTPUT" ]]; then
      echo "WARNING: LoRA weights not found at $LORA_OUTPUT"
    else
      log "✓ LoRA weights saved: $LORA_OUTPUT"
    fi

    mark_stage_done "$SENTINEL"
    log "✓ Stage 2 complete"
  fi
fi

# ─── Stage 3: ControlNet Training ─────────────────────────────────────────────
if [[ "$START_STAGE" -le 3 && "$END_STAGE" -ge 3 ]]; then
  log_section "Stage 3: ControlNet Training (Edge Conditioning)"
  SENTINEL="${CHECKPOINT_DIR}/.stage3_done"
  CTRL_OUTPUT="${CHECKPOINT_DIR}/fashion_controlnet/config.json"

  if check_stage_done "$SENTINEL"; then
    log "✓ Stage 3 already complete (sentinel found)"
  else
    # Ensure LoRA is ready
    if [[ ! -d "${CHECKPOINT_DIR}/fashion_lora" ]]; then
      echo "ERROR: LoRA checkpoint not found. Run Stage 2 first."
      exit 1
    fi

    log "Training ControlNet on edge maps..."
    run_accelerate training/train_controlnet.py \
      --config configs/train_controlnet.yaml \
      2>&1 | tee "${LOG_DIR}/stage3_controlnet_${TIMESTAMP}.log"

    if [[ ! -f "$CTRL_OUTPUT" ]]; then
      echo "WARNING: ControlNet config not found at $CTRL_OUTPUT"
    else
      log "✓ ControlNet saved: ${CHECKPOINT_DIR}/fashion_controlnet/"
    fi

    mark_stage_done "$SENTINEL"
    log "✓ Stage 3 complete"
  fi
fi

# ─── Stage 4: IP-Adapter Training ─────────────────────────────────────────────
if [[ "$START_STAGE" -le 4 && "$END_STAGE" -ge 4 ]]; then
  log_section "Stage 4: IP-Adapter Training (Reference Conditioning)"
  SENTINEL="${CHECKPOINT_DIR}/.stage4_done"
  IPA_OUTPUT="${CHECKPOINT_DIR}/fashion_ip_adapter/image_proj.pth"

  if check_stage_done "$SENTINEL"; then
    log "✓ Stage 4 already complete (sentinel found)"
  else
    if [[ ! -d "${CHECKPOINT_DIR}/fashion_controlnet" ]]; then
      echo "WARNING: ControlNet not found. Proceeding with base model only."
    fi

    log "Training IP-Adapter..."
    run_accelerate training/train_ip_adapter.py \
      --config configs/train_ip_adapter.yaml \
      2>&1 | tee "${LOG_DIR}/stage4_ip_adapter_${TIMESTAMP}.log"

    if [[ ! -f "$IPA_OUTPUT" ]]; then
      echo "WARNING: IP-Adapter weights not found at $IPA_OUTPUT"
    else
      log "✓ IP-Adapter saved: $IPA_OUTPUT"
    fi

    mark_stage_done "$SENTINEL"
    log "✓ Stage 4 complete"
  fi
fi

# ─── Pipeline Complete ─────────────────────────────────────────────────────────
log_section "✅ All Stages Complete!"
echo ""
echo "Checkpoints:"
echo "  Segmentation : ${CHECKPOINT_DIR}/segmentation/best_model.pth"
echo "  LoRA         : ${CHECKPOINT_DIR}/fashion_lora/fashion_lora.safetensors"
echo "  ControlNet   : ${CHECKPOINT_DIR}/fashion_controlnet/"
echo "  IP-Adapter   : ${CHECKPOINT_DIR}/fashion_ip_adapter/"
echo ""
echo "Next steps:"
echo "  python api/app.py                  # Start API server"
echo "  cd frontend && npm install && npm start  # Start UI"
echo "  python inference/generate.py       # Test inference"
echo ""
echo "Logs: ${LOG_DIR}/"
