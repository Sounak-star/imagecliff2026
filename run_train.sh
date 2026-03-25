#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_train.sh  —  Complete training pipeline for ImageCLEF-MR2026-OpenQA-Visual
#
# Run from inside your virtual environment:
#   source ~/ImageCLEF-2026-Multimod/venv/bin/activate
#   bash run_train.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on error

# ── Paths (adjust if needed) ──────────────────────────────────────────────────
DATASET_DIR="./dataset_local"          # created by download.py
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct" # ~5 GB, open-source, fits in 50 GB
OUTPUT_DIR="./output/qwen2vl-finetuned"
PREDICTIONS="./predictions.json"

# ── 1. Install dependencies ───────────────────────────────────────────────────
echo ">>> Installing dependencies..."
pip install -q -r requirements.txt

# ── 2. Check dataset exists ───────────────────────────────────────────────────
if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset not found at $DATASET_DIR"
    echo "Run: python download.py"
    exit 1
fi
echo ">>> Dataset found at $DATASET_DIR"

# ── 3. Check disk space (should be > 20 GB free) ──────────────────────────────
FREE_GB=$(df -BG . | awk 'NR==2 {print $4}' | tr -d 'G')
echo ">>> Free disk space: ${FREE_GB} GB"
if [ "$FREE_GB" -lt 15 ]; then
    echo "WARNING: Less than 15 GB free. Model weights + checkpoints may fail."
fi

# ── 4. Fine-tune ──────────────────────────────────────────────────────────────
echo ">>> Starting fine-tuning..."
python train.py \
    --dataset_dir      "$DATASET_DIR"  \
    --model_name       "$MODEL_NAME"   \
    --output_dir       "$OUTPUT_DIR"   \
    --num_train_epochs 5               \
    --per_device_train_batch_size 1    \
    --gradient_accumulation_steps 8    \
    --learning_rate    2e-4            \
    --max_seq_len      1024            \
    --save_steps       66              \
    --eval_steps       66              \
    --logging_steps    10              \
    --bf16

echo ">>> Training complete. Model saved to $OUTPUT_DIR"

# ── 5. Evaluate on dev split ──────────────────────────────────────────────────
echo ">>> Evaluating on dev split..."
python evaluate.py \
    --model_dir    "$OUTPUT_DIR"      \
    --base_model   "$MODEL_NAME"      \
    --dataset_dir  "$DATASET_DIR"     \
    --split        dev                \
    --output_file  "$PREDICTIONS"

echo ">>> Done! Predictions saved to $PREDICTIONS"
echo ">>> Submit predictions to: https://ai4media-bench.aimultimedialab.ro/competitions/15/"
