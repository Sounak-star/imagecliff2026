"""
Evaluation / inference script for ImageCLEF-MR2026-OpenQA-Visual.
Generates predictions from the fine-tuned model and computes metrics.

Usage:
    python evaluate.py \
        --model_dir ./output/qwen2vl-finetuned \
        --dataset_dir ./dataset_local \
        --split dev \
        --output_file predictions.json
"""

import argparse
import json
import logging
import io

import torch
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
import evaluate as hf_evaluate
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATASET = "./dataset_local"

SYSTEM_PROMPT = (
    "You are a multilingual visual question answering assistant. "
    "You are given an image that contains an exam question along with a visual element "
    "(diagram, graph, table, or figure). "
    "Study the image carefully and answer the question concisely and accurately."
)


def load_pil_image(img_field) -> Image.Image:
    if isinstance(img_field, Image.Image):
        return img_field.convert("RGB")
    if isinstance(img_field, dict):
        if img_field.get("bytes"):
            return Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        if img_field.get("path"):
            return Image.open(img_field["path"]).convert("RGB")
    if isinstance(img_field, (bytes, bytearray)):
        return Image.open(io.BytesIO(img_field)).convert("RGB")
    raise ValueError(f"Cannot load image: unsupported type {type(img_field)}")


def generate_answer(model, processor, image: Image.Image, max_new_tokens: int = 128) -> str:
    """Run inference on one image and return the model's predicted answer."""
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Look at the image carefully and answer the question shown."},
            ],
        },
    ]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text_prompt],
        images=[[image]],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )
    # Only decode the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned VLM on ImageCLEF-MR2026")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the fine-tuned model/adapter directory")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Base model name (required when using LoRA adapters)")
    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET,
                        help="Path to dataset saved with save_to_disk()")
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev"])
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--output_file", type=str, default="predictions.json")
    parser.add_argument("--is_lora", action="store_true", default=True,
                        help="The model_dir contains LoRA adapters (default: True)")
    parser.add_argument("--no_lora", action="store_true",
                        help="model_dir contains a fully merged model")
    args = parser.parse_args()

    use_lora = args.is_lora and not args.no_lora

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)

    logger.info("Loading model...")
    if use_lora:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, args.model_dir)
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading dataset ({args.split}) from: {args.dataset_dir}")
    dataset = load_from_disk(args.dataset_dir)[args.split]
    logger.info(f"Evaluating on {len(dataset)} examples")

    # ── Generate predictions ──────────────────────────────────────────────────
    predictions, references, results = [], [], []

    for example in tqdm(dataset, desc="Generating"):
        img  = load_pil_image(example["image"])
        pred = generate_answer(model, processor, img, args.max_new_tokens)
        gold = str(example["answer"])

        predictions.append(pred)
        references.append(gold)
        results.append({
            "question_id": example["question_id"],
            "subject":     example["subject"],
            "language":    example["language"],
            "prediction":  pred,
            "reference":   gold,
            "exact_match": pred.strip().lower() == gold.strip().lower(),
        })

    # ── Compute metrics ────────────────────────────────────────────────────────
    rouge = hf_evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    exact_match  = sum(r["exact_match"] for r in results) / len(results)

    metrics = {
        "exact_match": round(exact_match, 4),
        "rouge1":      round(rouge_scores["rouge1"], 4),
        "rouge2":      round(rouge_scores["rouge2"], 4),
        "rougeL":      round(rouge_scores["rougeL"], 4),
        "num_examples": len(results),
    }

    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k:<20}: {v}")
    logger.info("=" * 50)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {"metrics": metrics, "predictions": results}
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    logger.info(f"Predictions saved to: {args.output_file}")


if __name__ == "__main__":
    main()
