# CS6220_Fall2025

This repository adapts the [MasonNLP](https://github.com/AHMRezaul/MEDIQA-WV-2025) RAG framework to build a multimodal, query-rewriting Retrieval-Augmented Generation pipeline for medical visual question answering (Stage 2 milestone complete).

## Highlights
- **Dataset**: VQA-RAD prepared via `scripts/prepare_vqa_rad.py`.
- **Indexes**: `build_corpus.py` + `build_image_index.py` create FAISS text/image stores under `retriever/vqa_rad/`.
- **Configurable pipeline**: `rag/` provides pluggable query rewriting (HyDE + decomposition), rerankers (MiniLM/LLM), and LLM helpers driven by YAML configs (see `configs/vqa_rad.yaml`).
- **Evaluation**: `eval_vqa_rad.py` runs the pipeline end-to-end; results logged in `experiments/vqa_rad.md`.

## Quick Start (New Dataset)
1. **Prepare data**  
   ```
   python scripts/prepare_<dataset>.py --splits train test
   ```
   Produce MediQA-style JSON + image folders under `dataset/<name>/`.

2. **Build indexes**  
   ```
   python build_corpus.py --train-file dataset/<name>/json_files/train.json \
       --index-file retriever/<name>/text.index \
       --texts-file retriever/<name>/texts.json
   python build_image_index.py --img-dir dataset/<name>/images \
       --train-file dataset/<name>/json_files/train.json \
       --index-out retriever/<name>/image.index \
       --metadata-out retriever/<name>/image_texts.json
   ```

3. **Add a config** (`configs/<name>.yaml`) pointing to the new dataset + desired rewrite/rerank modules.

4. **Evaluate**  
   ```
   python eval_vqa_rad.py --config configs/<name>.yaml \
       --split test --top-k 3 --output result/<name>_predictions.json
   ```

Results and observations should be logged in `experiments/<name>.md`.
