# VQA-RAD Baseline Evaluation

## Dataset
- Source: [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad)  
- Splits: train (1,793 QA pairs), test (451 QA pairs)  
- Preparation: `scripts/prepare_vqa_rad.py` downloads images and emits MediQA-style JSON.

## Retrieval Assets
- Text index: `retriever/vqa_rad/text.index` (SentenceTransformer `all-MiniLM-L6-v2`)
- Image index: `retriever/vqa_rad/image.index` (CLIP `openai/clip-vit-base-patch32`)

## Baseline Configuration
- Evaluator: `eval_vqa_rad.py`
- Strategy: text-only nearest neighbor, prediction = top retrieved exemplar answer.
- Command:
  ```
  source .venv/bin/activate
  python eval_vqa_rad.py --split test --output result/vqa_rad_baseline_predictions.json
  ```

## Results (Test Split)
| Metric   | Value |
|----------|-------|
| Accuracy | 0.3503 |

Notes: This baseline simply copies answers, so it shows retrieval strength without LLM reasoning. Future runs (HyDE, decomposition, reranking) will be appended below.

---

## Stage 2 – Query Rewriting + Reranking
- Config: `configs/vqa_rad.yaml`
- Modules:
  - Query rewriting: HyDE + decomposition via `google/flan-t5-large`
  - Reranker: MiniLM cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Command (latest run):
  ```
  source .venv/bin/activate
  python eval_vqa_rad.py --config configs/vqa_rad.yaml --split test \
      --top-k 3 --output result/vqa_rad_stage2_hybrid_predictions.json
  ```

### Results (Test Split)
| Strategy | Reranker | Accuracy |
|----------|----------|----------|
| HyDE (1 hypo) | MiniLM cross-encoder | 0.3415 |
| HyDE (2 hypo) + decomposition + voting | MiniLM cross-encoder | **0.3525** |

Upgrading the rewrite LLM, chaining decomposition, retrieving a larger candidate pool, and majority-voting over the top 3 reranked exemplars nudged accuracy above the baseline (0.3503 → 0.3525). Further gains should come from stronger LLMs (hosted inference), LLM reranking, or hybrid (text+image) retrieval in Stage 3.

