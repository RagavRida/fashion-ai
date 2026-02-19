#!/bin/bash
# Fashion Reuse Studio — Full Training Launcher
# Run with: nohup bash run_training.sh > logs/nohup.log 2>&1 &

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"

cd ~/fashion-ai
mkdir -p logs data/raw data/processed checkpoints

LOG=logs/training.log

echo "==========================================" | tee "$LOG"
echo "Training started: $(date)"                  | tee -a "$LOG"
python3 -c "import torch; print('GPU:', torch.cuda.get_device_name(0), '| VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')" | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"

# Stage 0: Download dataset
echo "[$(date)] Downloading DeepFashion dataset..." | tee -a "$LOG"
python3 dataset_builder/download_deepfashion.py --output_dir data/raw 2>&1 | tee -a "$LOG"

if [ $? -ne 0 ]; then
    echo "[$(date)] WARNING: Dataset download failed, trying Kaggle fallback..." | tee -a "$LOG"
    python3 dataset_builder/download_deepfashion.py --source kaggle --output_dir data/raw 2>&1 | tee -a "$LOG"
fi

echo "[$(date)] Dataset step done" | tee -a "$LOG"

# Stages 1-4: Full training pipeline
echo "[$(date)] Starting 4-stage training (prep + seg + lora + controlnet + ip-adapter)..." | tee -a "$LOG"
bash training/automated_train_pipeline.sh \
    --gpu-ids 0 \
    --prep-dataset \
    --start-stage 0 \
    2>&1 | tee -a "$LOG"

echo "==========================================" | tee -a "$LOG"
echo "[$(date)] ALL TRAINING COMPLETE!"           | tee -a "$LOG"
echo "==========================================" | tee -a "$LOG"
