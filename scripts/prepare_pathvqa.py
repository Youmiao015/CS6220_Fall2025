#!/usr/bin/env python3
"""
Prepare PathVQA in the same JSON format used by our VQA-RAD pipeline.

- Loads images + Q/A from HuggingFace dataset "flaviagiammarino/path-vqa"
- Saves images under: dataset/pathvqa/images/
- Saves MediQA-style JSON under: dataset/pathvqa/json_files/<split>.json
"""

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset  # type: ignore
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=str,
        default="dataset/pathvqa",
        help="Root directory to write images/ and json_files/ under.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="HF splits to convert (e.g. train validation test).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="flaviagiammarino/path-vqa",
        help="HuggingFace dataset repo id.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_root = Path(args.out_root)
    img_dir = out_root / "images"
    json_dir = out_root / "json_files"
    img_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading HF dataset: {args.repo_id}")

    for split in args.splits:
        print(f"[+] Converting split: {split}")
        ds = load_dataset(args.repo_id, split=split)

        records = []
        MAX_QA = 3000
        MAX_IMAGES = 1000
        saved_images = set()

        for i, ex in enumerate(ds):
            if len(records) >= MAX_QA:
                break
            # HF fields: 'image' (PIL), 'question', 'answer'
            img: Image.Image = ex["image"]
            question: str = ex["question"]
            answer: str = ex["answer"]

            img_filename = f"{split}_{i:05d}.jpg"
            img_path = img_dir / img_filename

            if len(saved_images) < MAX_IMAGES:
                img = img.convert("RGB")
                img.save(img_path)
                saved_images.add(img_filename)
            else:
                img_filename = list(saved_images)[0]

            rec = {
                "encounter_id": f"{split}_{i:05d}",
                "image_ids": [img_filename],
                "query_title_en": "",
                "query_content_en": question.strip(),
                "responses": [
                    {
                        "author_id": "gold",
                        "content_en": answer.strip(),
                    }
                ],
            }
            records.append(rec)

        out_json = json_dir / f"{split}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"    -> wrote {len(records)} examples to {out_json}")

    print("[+] Done.")


if __name__ == "__main__":
    main()
