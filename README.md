# ImageCLEF-MR2026-OpenQA-Visual — Training Code

Fine-tuning **[Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)** (open-source VLM) on the [ImageCLEF-MR2026-OpenQA-Visual](https://huggingface.co/datasets/SU-FMI-AI/ImageCLEF-MR2026-OpenQA-Visual) dataset.

> **Rules compliance**: Open-weights models  only ✓ | Pre-trained VLMs allowed ✓ | No proprietary APIs ✓

## Dataset Overview

| Split | Examples |
|-------|----------|
| train | 528      |
| dev   | 240      |

- **Columns**: `image` (PIL), `answer` (str), `subject`, `language`, `question_id`, + visual type flags  
- **No separate question text** — the question is embedded inside the image (scanned exam paper)  
- **Multilingual**: Bulgarian, Chinese, and others

## File Structure

```
ImageCLEF-2026-Multimod/
├── dataset_local/        ← already downloaded by download.py
├── venv/                 ← virtual environment (already set up)
├── download.py           ← original download script
├── train.py              ← fine-tuning script  (this repo)
├── evaluate.py           ← evaluation / inference  (this repo)
├── run_train.sh          ← one-command pipeline  (this repo)
└── requirements.txt      ← dependencies  (this repo)
```

## Storage estimate (50.5 GB budget)

| Item | Size |
|------|------|
| Dataset (on disk) | ~1.4 GB |
| Qwen2-VL-2B weights | ~5 GB |
| LoRA checkpoints (2 saves) | ~0.5 GB |
| OS + venv + libs | ~10 GB |
| **Total** | **~17 GB** ✓ |

## Quick Start

```bash
# Activate your venv
source ~/ImageCLEF-2026-Multimod/venv/bin/activate

# Copy these scripts into your project folder
cp train.py evaluate.py run_train.sh requirements.txt ~/ImageCLEF-2026-Multimod/

# Run the full pipeline (install deps + train + evaluate)
cd ~/ImageCLEF-2026-Multimod
bash run_train.sh
```

### Or run steps manually

```bash
# Install deps
pip install -r requirements.txt

# Train
python train.py \
    --dataset_dir ./dataset_local \
    --model_name  Qwen/Qwen2-VL-2B-Instruct \
    --output_dir  ./output/qwen2vl-finetuned \
    --num_train_epochs 5 \
    --bf16

# Evaluate
python evaluate.py \
    --model_dir   ./output/qwen2vl-finetuned \
    --base_model  Qwen/Qwen2-VL-2B-Instruct \
    --dataset_dir ./dataset_local \
    --split dev \
    --output_file predictions.json
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_dir` | `./dataset_local` | Local dataset path (from `download.py`) |
| `--model_name` | `Qwen/Qwen2-VL-2B-Instruct` | Open-source VLM to fine-tune |
| `--output_dir` | `./output/qwen2vl-finetuned` | Save path for checkpoints |
| `--num_train_epochs` | 5 | Epochs (528 samples, fast to train) |
| `--gradient_accumulation_steps` | 8 | Effective batch = 1×8 = 8 |
| `--no_lora` | False | Full fine-tune (more VRAM) |
| `--no_4bit` | False | Disable QLoRA quantization |

## Metrics

Evaluated on the `dev` split with:
- **Exact Match** (case-insensitive)
- **ROUGE-1 / ROUGE-2 / ROUGE-L**

Submit `predictions.json` to the [AI4Media-Bench leaderboard](https://ai4media-bench.aimultimedialab.ro/competitions/15/).
