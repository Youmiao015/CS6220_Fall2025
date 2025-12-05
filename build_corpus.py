import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_TRAIN_FILE = "dataset/mediqa-wv/json_files/train-valid.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_FILE = "retriever/index.faiss"
DEFAULT_TEXTS_FILE = "retriever/texts.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index for text corpus.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE, help="Path to training JSON file.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="SentenceTransformer ID.")
    parser.add_argument("--index-file", default=DEFAULT_INDEX_FILE, help="Destination FAISS index file.")
    parser.add_argument("--texts-file", default=DEFAULT_TEXTS_FILE, help="Destination JSON metadata file.")
    parser.add_argument("--normalize", action="store_true", help="Force embedding normalization (default already true).")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.train_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[+] Loading embedding model: {args.embedding_model}")
    model = SentenceTransformer(args.embedding_model)
    texts = []
    vectors = []

    print(f"[+] Encoding {len(data)} text examples...")
    for ex in tqdm(data, desc="Building text embeddings", unit="doc"):
        query = f"{ex['query_title_en']}  {ex['query_content_en']}"
        texts.append(ex)
        emb = model.encode(query, normalize_embeddings=True)
        vectors.append(emb)

    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    os.makedirs(os.path.dirname(args.index_file), exist_ok=True)
    faiss.write_index(index, args.index_file)

    with open(args.texts_file, "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)

    print(f"[+] Saved FAISS index and metadata with {len(texts)} examples.")

if __name__ == "__main__":
    main()
