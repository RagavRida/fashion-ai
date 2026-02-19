#!/bin/bash
# Fashion Reuse Studio — BUDGET Training Script ($30 / ~10h on A100)
# Strategy:
#   - Fast pseudo-masks (Otsu, 2min) instead of GrabCut (14h)
#   - 10K sample subset instead of 42K
#   - Stage 1: Segmentation (15 epochs, ~20min)
#   - Stage 2: LoRA SDXL (12K steps, ~3.5h) ← most important stage
#   - SKIP Stage 3+4 (ControlNet/IP-Adapter) — add later if budget allows
#
# Total estimate: ~5h GPU time = ~$12-15 on Shadeform A100 @ $2.5/hr

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"
export HF_HOME="$HOME/.cache/huggingface"

cd ~/fashion-ai
mkdir -p logs checkpoints outputs/logs

LOG=logs/training.log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "=========================================="
log "Fashion Reuse Studio — BUDGET Training Run"
log "GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))')"
log "VRAM: $(python3 -c 'import torch; print(round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")')"
log "Budget mode: 10K samples, fast masks, LoRA only"
log "=========================================="

# ── Step 1: Fast pseudo-masks (replaces 14h GrabCut with ~2min) ────────────
SENTINEL_MASKS="data/processed/.fast_masks_done"
if [ ! -f "$SENTINEL_MASKS" ]; then
    log "Step 1: Generating fast pseudo-masks (Otsu threshold)..."
    python3 dataset_builder/build_masks_fast.py \
        --metadata data/processed/metadata_with_edges.jsonl \
        --output_masks data/processed/masks \
        --output_metadata data/processed/metadata_with_masks.jsonl \
        --num_workers 8 \
        --limit 10000 2>&1 | tee -a "$LOG"
    touch "$SENTINEL_MASKS"
    log "Step 1 DONE ✓ — $(ls data/processed/masks/ | wc -l) masks"
else
    log "Step 1: Fast masks already done ✓"
fi

# ── Step 2: Export final metadata.jsonl (10K subset) ───────────────────────
SENTINEL_JSONL="data/processed/.fast_jsonl_done"
if [ ! -f "$SENTINEL_JSONL" ]; then
    log "Step 2: Exporting 10K-sample metadata.jsonl..."
    python3 dataset_builder/export_jsonl.py \
        --metadata data/processed/metadata_with_masks.jsonl \
        --output data/processed/metadata.jsonl \
        --limit 10000 2>&1 | tee -a "$LOG"
    touch "$SENTINEL_JSONL"
    TOTAL=$(wc -l < data/processed/metadata.jsonl 2>/dev/null || echo "0")
    log "Step 2 DONE ✓ — $TOTAL training samples"
else
    log "Step 2: metadata.jsonl already exported ✓"
fi

# ── Step 3: Stage 1 — Segmentation CNN (15 epochs, ~20min on A100) ─────────
rm -f checkpoints/.stage1_done 2>/dev/null || true
log "=========================================="
log "Stage 1: Segmentation CNN (15 epochs, 10K samples)"
log "=========================================="
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    training/train_segmentation.py \
    --config configs/train_segmentation.yaml \
    2>&1 | tee -a "$LOG"
log "Stage 1 DONE ✓"

# ── Step 4: Stage 2 — LoRA Fine-Tuning (12K steps, ~3.5h on A100) ──────────
log "=========================================="
log "Stage 2: LoRA SDXL Fine-Tuning (12K steps)"
log "=========================================="
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes=1 \
    --mixed_precision=fp16 \
    training/train_lora.py \
    --config configs/train_lora.yaml \
    2>&1 | tee -a "$LOG"
log "Stage 2 DONE ✓"

log "=========================================="
log "BUDGET TRAINING COMPLETE! 🎉"
log "  Segmentation : checkpoints/segmentation/"
log "  LoRA weights : checkpoints/fashion_lora/"
log "  Total cost   : ~5-6h × \$2.5/hr ≈ \$12-15"
log "  Next steps   : python api/app.py to test inference"
log "=========================================="
