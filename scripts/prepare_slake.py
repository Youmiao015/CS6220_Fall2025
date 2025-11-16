#!/usr/bin/env python3
"""
Download the BoKelvin/SLAKE dataset from Hugging Face, persist the
images and aligned metadata locally, and emit JSON files shaped like the
existing MediQA wound-care dataset so the current build scripts can reuse
them without major changes.
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "slake"
IMAGES_DIR = DATASET_DIR / "images"
JSON_DIR = DATASET_DIR / "json_files"
IMGS_ZIP = DATASET_DIR / "imgs.zip"
IMGS_EXTRACTED = DATASET_DIR / "imgs"


def ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    JSON_DIR.mkdir(parents=True, exist_ok=True)


def download_and_extract_images() -> None:
    """Download imgs.zip from HuggingFace and extract images."""
    if IMGS_EXTRACTED.exists() and any(IMGS_EXTRACTED.iterdir()):
        print(f"[+] Images already extracted at {IMGS_EXTRACTED}")
        return

    print("[+] Downloading SLAKE images (imgs.zip)...")
    zip_path = hf_hub_download(
        repo_id="BoKelvin/SLAKE",
        filename="imgs.zip",
        repo_type="dataset",
        local_dir=DATASET_DIR,
    )

    print(f"[+] Extracting images to {IMGS_EXTRACTED}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    print(f"[+] Images extracted successfully")


def get_image_path(img_name: str) -> Optional[Path]:
    """
    Find image file from img_name.

    Args:
        img_name: Image filename from dataset (e.g., "xmlab1/source.jpg")

    Returns:
        Path to image file, or None if not found
    """
    # Try direct path
    img_path = IMGS_EXTRACTED / img_name
    if img_path.exists():
        return img_path

    # Try with different extensions
    for ext in [".jpg", ".png", ".jpeg"]:
        test_path = IMGS_EXTRACTED / img_name.replace(".jpg", ext)
        if test_path.exists():
            return test_path

    return None


def convert_split(split: str, limit: Optional[int] = None) -> List[Dict]:
    """
    Load SLAKE split from HuggingFace and convert to MediQA format.

    Args:
        split: Dataset split ('train', 'validation', 'test')
        limit: Maximum number of examples to process (for testing)

    Returns:
        List of examples in MediQA format
    """
    dataset = load_dataset("BoKelvin/SLAKE", split=split)
    converted: List[Dict] = []

    # Apply limit if specified
    total = len(dataset) if limit is None else min(limit, len(dataset))

    skipped = 0
    for idx, example in enumerate(dataset):
        if idx >= total:
            break

        # Get metadata fields
        img_name = example.get("img_name", "").strip()
        question = example.get("question", "").strip()
        answer = example.get("answer", "").strip()
        qid = example.get("qid", idx)

        # Skip examples without question
        if not question or not img_name:
            skipped += 1
            continue

        # Find and load image from extracted folder
        img_path = get_image_path(img_name)
        if img_path is None:
            print(f"[WARN] Image not found for {img_name}, skipping...")
            skipped += 1
            continue

        # Use original img_name as relative path (keep folder structure)
        # This preserves the original organization
        rel_path = img_name

        # Convert to MediQA format
        entry = {
            "encounter_id": f"slake_{split}_{qid:05d}",
            "query_title_en": question,
            "query_content_en": "",  # SLAKE doesn't have separate content field
            "image_ids": [rel_path],
            "responses": [
                {
                    "author_id": "gold",
                    "content_en": answer,
                }
            ],
            "split": split,
            # Optional: preserve SLAKE-specific metadata for future use
            "metadata": {
                "qid": qid,
                "img_id": example.get("img_id"),
                "img_name": img_name,
                "location": example.get("location"),
                "modality": example.get("modality"),
                "answer_type": example.get("answer_type"),
                "content_type": example.get("content_type"),
                "base_type": example.get("base_type"),
            },
        }
        converted.append(entry)

    if skipped > 0:
        print(f"[!] Skipped {skipped} examples (missing images or questions)")

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SLAKE dataset locally")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        help="Dataset splits to download and convert.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per split (for testing). Use --limit 100 for quick test.",
    )
    args = parser.parse_args()

    ensure_dirs()

    # Download and extract images first
    download_and_extract_images()

    all_records: List[Dict] = []

    for split in args.splits:
        print(f"[+] Processing split '{split}'...")
        records = convert_split(split, limit=args.limit)
        all_records.extend(records)

        # Save per-split JSON file
        out_file = JSON_DIR / f"{split}.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[+] Saved {len(records)} records → {out_file}")

    # Save merged training file (for building indexes)
    # Use 'train' split for index building, or all if 'train' not in splits
    train_records = [r for r in all_records if r["split"] == "train"]
    if not train_records:
        train_records = all_records

    merged_file = JSON_DIR / "train-valid.json"
    with merged_file.open("w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=2)
    print(f"[+] Saved merged training file with {len(train_records)} records → {merged_file}")

    print(f"\n[✓] Total processed: {len(all_records)} examples")
    if args.limit:
        print(f"[!] Limited to {args.limit} examples per split for testing")


if __name__ == "__main__":
    main()
