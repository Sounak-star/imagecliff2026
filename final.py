"""
Inference script for ImageCLEF-MR2026-OpenQA-Visual
Adapted from the 2025 MCQ script — same model, same 4-bit loading.

Changes from 2025:
  - Loads dataset from dataset_local/ (HuggingFace format) instead of JSON
  - Open-ended QA prompt instead of MCQ (A/B/C/D/E)
  - Outputs free-form text answer instead of letter extraction
  - Uses question_id instead of sample_id

Usage:
    python final.py
    python final.py --split dev
    python final.py --split train
"""

import json
import torch
from PIL import Image
from datasets import load_from_disk
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = "./dataset_local"
OUTPUT_FILE = "run.json"
MODEL_NAME  = "Qwen/Qwen2.5-VL-72B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev"],
                        help="Dataset split to run inference on")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output JSON file for predictions")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load model (same as 2025 — 4-bit quantized) ──────────────────────────
    print("🔥 Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_enable_fp32_cpu_offload=True, # Allow spilling to CPU RAM
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
    )

    processor = AutoProcessor.from_pretrained(args.model, use_fast=True)

    # ── Load dataset from disk ────────────────────────────────────────────────
    print(f"📦 Loading dataset ({args.split}) from {args.dataset_dir}...")
    dataset = load_from_disk(args.dataset_dir)
    data = dataset[args.split]
    print(f"   → {len(data)} examples")

    results = []

    # ── Process each example ──────────────────────────────────────────────────
    for idx, row in enumerate(data):
        question_id = row["question_id"]
        language    = row["language"]
        subject     = row["subject"]
        image       = row["image"]  # PIL Image from HuggingFace dataset

        # Ensure PIL Image
        if not isinstance(image, Image.Image):
            print(f"⚠️ Skipping {question_id}: image is not PIL")
            continue
        image = image.convert("RGB")

        try:
            # ── Build prompt (open-ended QA, NOT MCQ) ─────────────────────────
            prompt = (
                f"You are an expert at answering visual exam questions. "
                f"This is an exam question in {language} about {subject}.\n\n"
                f"Step 1: Carefully read the question shown in the image.\n"
                f"Step 2: Analyze any diagrams, graphs, tables, or visual content.\n"
                f"Step 3: Reason through the question carefully.\n"
                f"Step 4: Provide a concise answer in the same language as the question. "
                f"Only give the answer, do not explain.\n"
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # ── Prepare inputs (same as 2025) ─────────────────────────────────
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            # Move inputs to the same device as the model's embedding layer (usually cuda:0 or cpu)
            # Since the model might be split across CPU and GPU, we send inputs to the device 
            # where the model expects the first layer's input.
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # ── Inference (same as 2025) ──────────────────────────────────────
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                response = processor.batch_decode(generated_ids, skip_special_tokens=True)
                text = response[0]

                # Extract assistant response (same logic as 2025)
                last_assistant_index = text.rfind("assistant")
                if last_assistant_index != -1:
                    answer = text[last_assistant_index + len("assistant"):].strip()
                else:
                    answer = text.strip()

            print(f"[{idx+1}/{len(data)}] {question_id} → {answer}")

            # ── Save result ───────────────────────────────────────────────────
            results.append({
                "question_id": question_id,
                "language": language,
                "answer": answer,
            })

            # Clean up GPU memory
            del inputs, generated_ids, response, image_inputs, video_inputs
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error processing {question_id}: {e}")
            results.append({
                "question_id": question_id,
                "language": language,
                "answer": "",
            })

    # ── Save output ───────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
