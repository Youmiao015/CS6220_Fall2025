#!/usr/bin/env python3
"""
Generative evaluation for SLAKE using Qwen2-VL-2B with RAG.
Retrieves similar examples and uses them in-context for answer generation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from retriever import Retriever, RetrieverConfig

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "result" / "slake_generative_predictions.json"


def load_split(json_dir: str, split: str, limit: Optional[int] = None) -> List[Dict]:
    """Load dataset split from JSON file."""
    split_file = Path(json_dir) / f"{split}.json"
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data = data[:limit]

    return data


def build_chat_messages(
    exemplars: List[Dict],
    test_case: Dict,
    images_root: Path,
    answer_instruction: Optional[str] = None,
) -> tuple[List[Dict], List[str]]:
    """
    Build chat messages and image paths for Qwen2-VL.

    Args:
        exemplars: Retrieved similar training examples
        test_case: Current test example
        images_root: Root directory for images

    Returns:
        (messages, image_paths) for Qwen2-VL processing
    """
    messages = []
    image_paths = []

    system_content = [
        {
            "type": "text",
            "text": "You are a medical VQA assistant. Use the retrieved exemplars to help answer the question about the final medical image."
        },
        {"type": "text", "text": "Always respond with a single short answer and no explanation."},
    ]
    if answer_instruction:
        system_content.append({"type": "text", "text": answer_instruction})

    messages.append({
        "role": "system",
        "content": system_content,
    })

    # Add exemplar Q&A pairs as few-shot examples
    for ex in exemplars:
        # Load exemplar image
        if 'img_name' in ex:
            img_path = images_root / ex['img_name']
            if img_path.exists():
                image_paths.append(str(img_path))

                # User message with image and question
                question = f"{ex['query_title_en']} {ex.get('query_content_en', '')}".strip()
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(img_path)},
                        {"type": "text", "text": question}
                    ]
                })

                # Assistant message with answer
                answer = ex['responses'][0]['content_en']
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}]
                })

    # Add current test question
    if 'img_name' in test_case:
        img_path = images_root / test_case['img_name']
        if img_path.exists():
            image_paths.append(str(img_path))

            question = f"{test_case['query_title_en']} {test_case.get('query_content_en', '')}".strip()
            user_content = [
                {"type": "image", "image": str(img_path)},
                {"type": "text", "text": question},
            ]
            if answer_instruction:
                user_content.append({"type": "text", "text": f"Instruction: {answer_instruction}"})
            messages.append({
                "role": "user",
                "content": user_content,
            })

    return messages, image_paths


def build_answer_instruction(example: Dict) -> str:
    """Return textual guidance for the expected answer format."""
    metadata = example.get("metadata") or {}
    answer_type = (metadata.get("answer_type") or "").upper()
    content_type = (metadata.get("content_type") or "").lower()

    instructions: List[str] = []
    if answer_type == "CLOSED":
        instructions.append("Answer with 'yes' or 'no'.")
    elif content_type == "modality":
        instructions.append("Respond with only the imaging modality name (e.g., ct, mri, x-ray, ultrasound).")
    elif content_type in {"organ", "position"}:
        instructions.append("Respond with only the anatomical structure or location.")
    elif content_type == "abnormality":
        instructions.append("State only the abnormal finding.")
    else:
        instructions.append("Reply with the exact short phrase that answers the question.")

    return " ".join(instructions)


def normalize_prediction(raw: str, example: Dict) -> str:
    """Normalize model output to improve exact-match accuracy."""
    metadata = example.get("metadata") or {}
    answer_type = (metadata.get("answer_type") or "").upper()

    text = raw.strip().lower()
    if not text:
        return ""

    # Collapse whitespace and remove filler prefixes
    text = re.sub(r"\s+", " ", text)
    for prefix in ["assistant:", "answer:", "final answer:", "the answer is", "response:"]:
        if prefix in text:
            text = text.split(prefix, 1)[-1].strip()

    if answer_type == "CLOSED":
        if "yes" in text:
            return "yes"
        if "no" in text:
            return "no"

    # Keep only the first clause/sentence
    for sep in [".", "!", "?"]:
        if sep in text:
            text = text.split(sep, 1)[0].strip()

    return text.strip(" ,;")


def main():
    parser = argparse.ArgumentParser(description="Generative evaluation with Qwen2-VL-2B + RAG on SLAKE.")
    parser.add_argument("--text-index", default="retriever/slake/text.index", help="Path to text index")
    parser.add_argument("--text-metadata", default="retriever/slake/texts.json", help="Path to text metadata")
    parser.add_argument("--image-index", default="retriever/slake/image.index", help="Path to image index")
    parser.add_argument("--image-metadata", default="retriever/slake/image_texts.json", help="Path to image metadata")
    parser.add_argument("--json-dir", default="dataset/slake/json_files", help="Directory with JSON files")
    parser.add_argument("--images-root", default="dataset/slake/imgs", help="Root directory for images")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output predictions file")
    parser.add_argument("--num-exemplars", type=int, default=2, help="Number of exemplars to retrieve")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test examples")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-2B-Instruct", help="Model ID")
    args = parser.parse_args()

    images_root = Path(args.images_root)

    print(f"[+] Loading Qwen2-VL model: {args.model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    print("[+] Loading retriever...")
    retriever = Retriever(RetrieverConfig(
        text_index=args.text_index,
        text_metadata=args.text_metadata,
        image_index=args.image_index,
        image_metadata=args.image_metadata,
    ))

    print(f"[+] Loading test data from {args.json_dir}/{args.split}.json")
    test_data = load_split(args.json_dir, args.split, limit=args.limit)

    gold_answers = []
    pred_answers = []
    predictions = []

    print(f"[+] Evaluating {len(test_data)} examples with generative LLM...")

    for example in tqdm(test_data, desc="Generating answers", unit="question"):
        encounter_id = example.get("encounter_id", "unknown")
        question = f"{example['query_title_en']} {example.get('query_content_en', '')}".strip()
        gold_answer = example['responses'][0]['content_en'].strip().lower()

        try:
            # Retrieve similar examples
            if 'img_name' in example:
                img_path = images_root / example['img_name']
                if img_path.exists():
                    query_image = Image.open(img_path).convert('RGB')
                    exemplars = retriever.retrieve_hybrid(
                        question,
                        query_image,
                        top_k=args.num_exemplars,
                        alpha=0.5
                    )
                else:
                    # Fallback to text-only if image not found
                    exemplars = retriever.retrieve_by_text(question, top_k=args.num_exemplars)
            else:
                exemplars = retriever.retrieve_by_text(question, top_k=args.num_exemplars)

            answer_instruction = build_answer_instruction(example)
            # Build chat with exemplars
            messages, _ = build_chat_messages(exemplars, example, images_root, answer_instruction)

            # Apply chat template
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process images and text
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            # Generate answer
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    temperature=0.2,
                    top_p=0.8,
                )

            # Trim input tokens and decode
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            raw_prediction = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            prediction = normalize_prediction(raw_prediction, example)

        except Exception as e:
            print(f"\n[ERROR] Failed on {encounter_id}: {e}")
            prediction = ""

        # Record results
        gold_answers.append(gold_answer)
        pred_answers.append(prediction)

        predictions.append({
            "encounter_id": encounter_id,
            "question": question,
            "gold": gold_answer,
            "prediction": prediction,
            "match": prediction == gold_answer,
            "metadata": example.get("metadata", {}),
        })

    # Calculate accuracy
    accuracy = accuracy_score(gold_answers, pred_answers)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_id,
            "split": args.split,
            "num_exemplars": args.num_exemplars,
            "accuracy": accuracy,
            "total_examples": len(test_data),
            "predictions": predictions,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Generative evaluation complete!")
    print(f"[+] Accuracy: {accuracy:.4f}")
    print(f"[+] Evaluated {len(test_data)} examples")
    print(f"[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
