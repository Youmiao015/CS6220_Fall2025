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

