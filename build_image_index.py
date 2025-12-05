import argparse
import json
import os

# Force CPU mode for M4 compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


DEFAULT_IMG_DIR = "dataset/mediqa-wv/images"
DEFAULT_TRAIN_FILE = "dataset/mediqa-wv/json_files/train-valid.json"
DEFAULT_INDEX_OUT = "retriever/image_index.faiss"
DEFAULT_METADATA_OUT = "retriever/image_texts.json"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def parse_args():
    parser = argparse.ArgumentParser(description="Build image FAISS index using CLIP.")
    parser.add_argument("--img-dir", default=DEFAULT_IMG_DIR, help="Root directory containing dataset images.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="JSON file with metadata entries.")
    parser.add_argument("--index-out", default=DEFAULT_INDEX_OUT, help="Destination FAISS index file.")
    parser.add_argument("--metadata-out", default=DEFAULT_METADATA_OUT, help="Destination JSON metadata file.")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP model name or path.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"[+] Loading CLIP model: {args.clip_model} (CPU mode for M4 compatibility)")
    clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True)
    clip_model = clip_model.to('cpu')  # Force CPU
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    with open(args.train_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Count total images
    total_images = sum(len(ex["image_ids"]) for ex in data)
    print(f"[+] Processing {total_images} images from {len(data)} examples...")

    image_vecs = []
    image_meta = []

    # Create progress bar for images
    pbar = tqdm(total=total_images, desc="Building image embeddings", unit="img")

    for ex in data:
        for fn in ex["image_ids"]:
            path = os.path.join(args.img_dir, fn)
            if not os.path.exists(path):
                pbar.update(1)
                continue

            try:
                image = Image.open(path).convert("RGB")
                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    emb = clip_model.get_image_features(**inputs).cpu().numpy()[0]
                emb /= np.linalg.norm(emb)
                image_vecs.append(emb)
                image_meta.append({"image_id": fn, "source": ex})
                pbar.update(1)
            except Exception as e:
                tqdm.write(f"[WARN] Failed on {fn}: {e}")
                pbar.update(1)

    pbar.close()

    if not image_vecs:
        raise ValueError("No image embeddings generated. Check if your images exist and can be read.")

    dim = image_vecs[0].shape[0]
    index = faiss.IndexFlatIP(dim)

    index.add(np.array(image_vecs).astype("float32"))
    os.makedirs(os.path.dirname(args.index_out), exist_ok=True)
    faiss.write_index(index, args.index_out)

    with open(args.metadata_out, "w", encoding="utf-8") as f:
        json.dump(image_meta, f, ensure_ascii=False, indent=2)

    print(f"[+] Saved image FAISS index and metadata to {args.index_out}")


if __name__ == "__main__":
    main()
