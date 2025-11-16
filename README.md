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

## SLAKE Dataset (Alternative)

The SLAKE dataset is a larger alternative to VQA-RAD with 14,028 samples across medical imaging modalities.

- **Source**: [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE)
- **Splits**: train (9,840), validation (2,100), test (2,090)
- **Features**: Multi-modal medical VQA with rich metadata (anatomical location, modality, question types)

### Quick Start with SLAKE

1. **Prepare data** (test with small subset first)
   ```
   python scripts/prepare_slake.py --splits train validation test --limit 100
   ```
   This downloads 100 samples per split for quick testing. Omit `--limit` for full dataset.

2. **Build indexes**
   ```
   python build_corpus.py --train-file dataset/slake/json_files/train.json \
       --index-file retriever/slake/text.index \
       --texts-file retriever/slake/texts.json
   python build_image_index.py --img-dir dataset/slake/images \
       --train-file dataset/slake/json_files/train.json \
       --index-out retriever/slake/image.index \
       --metadata-out retriever/slake/image_texts.json
   ```

3. **Evaluate**
   ```
   python eval_slake.py --config configs/slake.yaml \
       --split test --top-k 3 --limit 50 --output result/slake_test_predictions.json
   ```
   Use `--limit 50` for quick testing, omit for full evaluation.

### Notes
- SLAKE is 6.25× larger than VQA-RAD (14,028 vs 2,244 samples)
- Building full indexes may take 3-4 hours on CPU; consider using GPU (Colab) for faster processing
- The dataset includes additional metadata (location, modality, question type) stored in the `metadata` field
