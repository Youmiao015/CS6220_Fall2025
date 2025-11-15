import json
from dataclasses import dataclass

import faiss
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor


@dataclass
class RetrieverConfig:
    text_index: str = "retriever/index.faiss"
    text_metadata: str = "retriever/texts.json"
    image_index: str = "retriever/image_index.faiss"
    image_metadata: str = "retriever/image_texts.json"
    text_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model_id: str = "openai/clip-vit-base-patch32"


class Retriever:
    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()

        # text-based retriever
        self.text_model = SentenceTransformer(self.config.text_model_id)
        self.text_index = faiss.read_index(self.config.text_index)
        with open(self.config.text_metadata, "r", encoding="utf-8") as f:
            self.text_data = json.load(f)

        # image-based retriever
        self.image_index = faiss.read_index(self.config.image_index)
        with open(self.config.image_metadata, "r", encoding="utf-8") as f:
            self.image_data = json.load(f)

        self.clip_model = CLIPModel.from_pretrained(self.config.clip_model_id)
        self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model_id)

    def retrieve_by_text(self, query_text: str, top_k: int = 2):
        vec = self.text_model.encode([query_text], normalize_embeddings=True).astype("float32")
        D, I = self.text_index.search(vec, top_k)
        return [self.text_data[i] for i in I[0]]

    def retrieve_by_image(self, image: Image.Image, top_k: int = 2):
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            vec = self.clip_model.get_image_features(**inputs).cpu().numpy()
        vec /= np.linalg.norm(vec)
        D, I = self.image_index.search(vec.astype("float32"), top_k)
        return [self.image_data[i]["source"] for i in I[0]]

    def retrieve_hybrid(self, query_text: str, query_image: Image.Image, top_k: int = 2, alpha: float = 0.5):
        # Encode text
        text_vec = self.text_model.encode([query_text], normalize_embeddings=True).astype("float32")
        D_text, I_text = self.text_index.search(text_vec, top_k * 2)

        # Encode image
        inputs = self.clip_processor(images=query_image, return_tensors="pt")
        with torch.no_grad():
            image_vec = self.clip_model.get_image_features(**inputs).cpu().numpy()
        image_vec /= np.linalg.norm(image_vec)
        D_img, I_img = self.image_index.search(image_vec.astype("float32"), top_k * 2)

        # Score map: encounter_id → combined score
        scores = {}

        for d, i in zip(D_text[0], I_text[0]):
            eid = self.text_data[i]["encounter_id"]
            scores[eid] = alpha * (1 - d)  # normalize to similarity

        for d, i in zip(D_img[0], I_img[0]):
            eid = self.image_data[i]["source"]["encounter_id"]
            scores[eid] = scores.get(eid, 0) + (1 - alpha) * (1 - d)

        # Sort and select
        sorted_eids = sorted(scores.items(), key=lambda x: -x[1])
        top_eids = {eid for eid, _ in sorted_eids[:top_k]}

        # Get full source entries
        return [entry["source"] for entry in self.image_data if entry["source"]["encounter_id"] in top_eids]

