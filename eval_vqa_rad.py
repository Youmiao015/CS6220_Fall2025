#!/usr/bin/env python3
"""
Simple baseline evaluator on the VQA-RAD dataset using nearest-neighbor answers.
This reuses the Retriever (configured with the VQA-RAD indexes) and predicts
each answer by copying the response from the top retrieved example. It reports
overall accuracy and writes per-example predictions for auditing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score  # type: ignore[import]

from retriever import Retriever, RetrieverConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "vqa-rad"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "result" / "vqa_rad_baseline_predictions.json"


def load_split(split: str) -> List[Dict]:
    split_file = DATASET_DIR / "json_files" / f"{split}.json"
    with split_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline RAG on VQA-RAD.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to evaluate (train/test).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write JSON predictions.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of retrieved examples to inspect.")
    args = parser.parse_args()

    data = load_split(args.split)

    retriever = Retriever(
        RetrieverConfig(
            text_index="retriever/vqa_rad/text.index",
            text_metadata="retriever/vqa_rad/texts.json",
            image_index="retriever/vqa_rad/image.index",
            image_metadata="retriever/vqa_rad/image_texts.json",
        )
    )

    gold_answers: List[str] = []
    pred_answers: List[str] = []
    predictions: List[Dict] = []

    for example in data:
        question = f"{example['query_title_en']} {example.get('query_content_en','')}".strip()
        retrieved = retriever.retrieve_by_text(question, top_k=args.top_k)
        candidate = retrieved[0]["responses"][0]["content_en"] if retrieved else ""
        gold = example["responses"][0]["content_en"]

        gold_answers.append(gold.strip().lower())
        pred_answers.append(candidate.strip().lower())
        predictions.append(
            {
                "encounter_id": example["encounter_id"],
                "question": question,
                "gold": gold,
                "prediction": candidate,
                "match": candidate.strip().lower() == gold.strip().lower(),
            }
        )

    accuracy = accuracy_score(gold_answers, pred_answers)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"accuracy": accuracy, "predictions": predictions}, f, ensure_ascii=False, indent=2)

    print(f"[+] Evaluated {len(data)} examples on split '{args.split}' → accuracy {accuracy:.4f}")
    print(f"[+] Detailed predictions saved to {out_path}")


if __name__ == "__main__":
    main()

