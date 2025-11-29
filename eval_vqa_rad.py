#!/usr/bin/env python3
"""
Evaluate VQA-RAD performance using the configurable retrieval pipeline that
supports query rewriting (HyDE/decomposition) and reranking (LLM or MiniLM).
Predictions are produced by copying the answer from the top reranked exemplar.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score  # type: ignore[import]

from rag import build_pipeline, load_config


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "result" / "vqa_rad_baseline_predictions.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "vqa_rad.yaml"


def load_split(json_dir: str, split: str) -> List[Dict]:
    split_file = Path(json_dir) / f"{split}.json"
    with split_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate configurable RAG pipeline on VQA-RAD.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to pipeline YAML config.")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to evaluate (train/test).")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to write JSON predictions.")
    parser.add_argument("--top-k", type=int, default=3, help="How many reranked exemplars to aggregate.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = build_pipeline(cfg)
    data = load_split(cfg.dataset.json_dir, args.split)

    gold_answers: List[str] = []
    pred_answers: List[str] = []
    predictions: List[Dict] = []

    for example in data:
        question = f"{example['query_title_en']} {example.get('query_content_en','')}".strip()
        exemplars = pipeline.retrieve(question, top_k=args.top_k)
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
                "predictions": predictions,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[+] Evaluated {len(data)} examples on split '{args.split}' → accuracy {accuracy:.4f}")
    print(f"[+] Detailed predictions saved to {out_path}")


if __name__ == "__main__":
    main()

