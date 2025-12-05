#!/usr/bin/env python3
"""
Evaluate SLAKE dataset performance using the configurable retrieval pipeline that
supports query rewriting (HyDE/decomposition) and reranking (LLM or MiniLM).
Predictions are produced by copying the answer from the top reranked exemplar.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from sklearn.metrics import accuracy_score  # type: ignore[import]
from tqdm import tqdm

from rag import build_pipeline, load_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "result" / "slake_predictions.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "slake.yaml"


def load_split(json_dir: str, split: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load dataset split from JSON file.

    Args:
        json_dir: Directory containing split JSON files
        split: Split name ('train', 'validation', 'test')
        limit: Maximum number of examples to load (for testing)

    Returns:
        List of examples
    """
    split_file = Path(json_dir) / f"{split}.json"
    with split_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Apply limit if specified
    if limit is not None:
        data = data[:limit]

    return data


def main():
    parser = argparse.ArgumentParser(description="Evaluate configurable RAG pipeline on SLAKE.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to pipeline YAML config.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to evaluate (train/validation/test).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write JSON predictions.")
    parser.add_argument("--top-k", type=int, default=3, help="How many reranked exemplars to aggregate.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate (for testing). Use --limit 50 for quick test.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = build_pipeline(cfg)
    data = load_split(cfg.dataset.json_dir, args.split, limit=args.limit)

    # Get images root directory from config
    images_root = Path(cfg.dataset.images_root) if hasattr(cfg.dataset, 'images_root') else Path("dataset/slake/imgs")

    gold_answers: List[str] = []
    pred_answers: List[str] = []
    predictions: List[Dict] = []

    print(f"[+] Evaluating {len(data)} examples from split '{args.split}'...")

    for example in tqdm(data, desc="Evaluating", unit="question"):
        question = f"{example['query_title_en']} {example.get('query_content_en','')}".strip()

        # Load query image if available
        query_image = None
        if 'img_name' in example:
            img_path = images_root / example['img_name']
            if img_path.exists():
                try:
                    query_image = Image.open(img_path).convert('RGB')
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")

        exemplars = pipeline.retrieve(question, query_image=query_image, top_k=args.top_k)
        answers = []
        for ex in exemplars:
            resp = (ex.get("responses") or [{}])[0].get("content_en", "")
            if resp:
                answers.append(resp.strip())
        if answers:
            candidate = Counter(answers).most_common(1)[0][0]
        else:
            candidate = ""
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
                "config": args.config,
                # Optional: include SLAKE metadata for analysis
                "metadata": example.get("metadata", {}),
            }
        )

    accuracy = accuracy_score(gold_answers, pred_answers)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": args.config,
                "split": args.split,
                "top_k": args.top_k,
                "accuracy": accuracy,
                "total_examples": len(data),
                "predictions": predictions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[+] Evaluated {len(data)} examples on split '{args.split}' → accuracy {accuracy:.4f}")
    print(f"[+] Detailed predictions saved to {out_path}")
    if args.limit:
        print(f"[!] Limited to {args.limit} examples for testing")


if __name__ == "__main__":
    main()
