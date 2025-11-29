#!/usr/bin/env python3
"""
Download the flaviagiammarino/vqa-rad dataset from Hugging Face, persist the
images and aligned metadata locally, and emit JSON files shaped like the
existing MediQA wound-care dataset so the current build scripts can reuse
them without major changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "vqa-rad"
IMAGES_DIR = DATASET_DIR / "images"
JSON_DIR = DATASET_DIR / "json_files"


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)


def save_example_image(split: str, idx: int, pil_image) -> str:
    split_dir = IMAGES_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{split}_{idx:05d}.jpg"
    path = split_dir / filename
    pil_image.convert("RGB").save(path, format="JPEG", quality=95)
    return f"{split}/{filename}"


def convert_split(split: str) -> List[Dict]:
    dataset = load_dataset("flaviagiammarino/vqa-rad", split=split)
    converted: List[Dict] = []

    for idx, example in enumerate(dataset):
        image = example.get("image")
        question = example.get("question", "").strip()
        answer = example.get("answer", "").strip()

        if image is None or not question:
            continue

        rel_path = save_example_image(split, idx, image)
        entry = {
            "encounter_id": f"{split}_{idx:05d}",
            "query_title_en": question,
            "query_content_en": "",
            "image_ids": [rel_path],
            "responses": [
                {
                    "author_id": "gold",
                    "content_en": answer,
                }
            ],
            "split": split,
        }
        converted.append(entry)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VQA-RAD dataset locally")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to download and convert.",
    )
    args = parser.parse_args()

    ensure_dirs()

    merged: List[Dict] = []
    for split in args.splits:
        records = convert_split(split)
        merged.extend(records)
        out_file = JSON_DIR / f"{split}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved {len(records)} records → {out_file}")

    merged_file = JSON_DIR / "train-valid.json"
    with merged_file.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved merged file → {merged_file}")


if __name__ == "__main__":
    main()

