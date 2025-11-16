# Running SLAKE Pipeline on Google Colab

This guide shows how to run the full SLAKE pipeline (with image indexes) on Google Colab with free GPU access.

## Why Colab?

- **Free T4 GPU** → 20-50× faster CLIP embedding vs local CPU
- **Time**: Full dataset (~14k samples) takes 3-4 hours on Colab vs 25-30 hours locally
- **No laptop blocking** → Your MacBook stays free for other work

---

## Setup Steps

### 1. Upload Your Project to Google Drive

**Option A: Zip and upload (Recommended)**
```bash
# On your Mac (in project directory)
cd /Users/oliverxt/Play/Georgia-Tech/2025Fall/6220-Big-Data-System-and-Analysis
zip -r final-project.zip final-project/ -x "final-project/.venv/*" "final-project/dataset/*" "final-project/retriever/*"
```

**Upload the zip file:**
- Go to [drive.google.com](https://drive.google.com)
- Upload `final-project.zip` to **My Drive** (root folder)
- Or upload to a specific folder and adjust the path in Step 3

**Option B: Use Google Drive desktop app**
- Copy the project folder directly to Google Drive
- Exclude `.venv/`, `dataset/`, `retriever/` folders (they'll be regenerated)
- No unzip needed if using this method

---

### 2. Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → New Notebook
3. Runtime → Change runtime type → Select **T4 GPU** (or any GPU)

---

### 3. Mount Google Drive & Extract Project

```python
from google.colab import drive
drive.mount('/content/drive')
```

Run the cell, authenticate with your Google account.

```python
# Unzip project to Colab workspace
!unzip -q /content/drive/MyDrive/final-project.zip -d /content/
!ls /content/final-project
```

Expected output: You should see `scripts/`, `configs/`, `rag/`, etc.

---

### 4. Navigate to Project

```python
import os
os.chdir('/content/final-project')
!pwd
```

Expected output: `/content/final-project`

---

### 5. Install Dependencies

```python
# Install required packages
!pip install datasets sentence-transformers transformers torch pillow scikit-learn pyyaml huggingface-hub

# Install faiss-gpu separately (conda-based, already available on Colab)
# Colab comes with faiss-gpu pre-installed, but if needed:
try:
    import faiss
    print("✓ faiss already installed")
except:
    !pip install faiss-cpu  # Fallback to CPU version if GPU not available

# Verify installation
import datasets, sentence_transformers, faiss, transformers
print("✓ All packages installed!")
print(f"FAISS version: {faiss.__version__}")
```

**Note:** Google Colab typically has `faiss` pre-installed with GPU support. If not, it will install CPU version as fallback.

---

## Running the Full Pipeline

### Option A: Full Dataset (14,028 samples, ~3-4 hours)

```python
# Step 1: Prepare dataset (download images + create JSON)
!python scripts/prepare_slake.py --splits train validation test

# Step 2: Build text index
!python build_corpus.py \
  --train-file dataset/slake/json_files/train.json \
  --index-file retriever/slake/text.index \
  --texts-file retriever/slake/texts.json

# Step 3: Build image index (GPU-accelerated!)
!python build_image_index.py \
  --img-dir dataset/slake/imgs \
  --train-file dataset/slake/json_files/train.json \
  --index-out retriever/slake/image.index \
  --metadata-out retriever/slake/image_texts.json

# Step 4: Run evaluation
!python eval_slake.py \
  --config configs/slake.yaml \
  --split test \
  --top-k 3 \
  --output result/slake_full_predictions.json
```

---

### Option B: Large Subset (3,000 samples, ~1 hour)

For faster testing:

```python
# Step 1: Prepare subset
!python scripts/prepare_slake.py --splits train validation test --limit 3000

# Steps 2-4: Same as above (will process 3000 samples instead of full dataset)
```

---

### Option C: Quick Test (100 samples, ~10 minutes)

Same as you ran locally, but with GPU for image index:

```python
!python scripts/prepare_slake.py --splits train validation test --limit 100
!python build_corpus.py --train-file dataset/slake/json_files/train.json \
  --index-file retriever/slake/text.index --texts-file retriever/slake/texts.json
!python build_image_index.py --img-dir dataset/slake/imgs \
  --train-file dataset/slake/json_files/train.json \
  --index-out retriever/slake/image.index \
  --metadata-out retriever/slake/image_texts.json
!python eval_slake.py --config configs/slake.yaml --split test --top-k 3 \
  --limit 50 --output result/slake_colab_test.json
```

---

## Downloading Results

### Method 1: Direct download from Colab

```python
from google.colab import files

# Download predictions
files.download('result/slake_full_predictions.json')

# Optionally download indexes for later use locally
!zip -r indexes.zip retriever/slake/
files.download('indexes.zip')
```

### Method 2: Copy to Google Drive (recommended for large files)

```python
# Copy results to Drive
!cp result/slake_full_predictions.json /content/drive/MyDrive/slake_results/
!cp -r retriever/slake /content/drive/MyDrive/slake_indexes/

print("✓ Results saved to Google Drive")
```

Then download from Drive on your Mac.

---

## Monitoring Progress

### Check GPU utilization:
```python
!nvidia-smi
```

### Monitor running process:
```python
# If running in background, tail logs:
!tail -f /tmp/pipeline.log  # If you redirect output to log file
```

### Estimate remaining time:
- Dataset prep: ~5-10 min (downloads images)
- Text index: ~5-10 min (CPU-bound)
- Image index: ~1-3 hours (GPU-accelerated, depends on dataset size)
- Evaluation: ~10-20 min

---

## Troubleshooting

### Error: "No matching distribution found for faiss-gpu"

**Solution:** Colab has `faiss` pre-installed. Just import it:
```python
import faiss
print(f"FAISS version: {faiss.__version__}")
```

If truly missing, install CPU version:
```python
!pip install faiss-cpu
```

### Runtime disconnects
- Free Colab has 12-hour runtime limit
- Full dataset should finish in 3-4 hours (safe)
- If disconnected, results are saved to Drive if you copied them

### Out of memory
```python
# Use smaller batch size for CLIP (edit build_image_index.py)
# Or process subset with --limit
```

### Slow download from HuggingFace
```python
# Set HF mirror (optional, for faster downloads in some regions)
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

---

## After Running on Colab

1. **Download results** (`slake_full_predictions.json`)
2. **Optional**: Download indexes (`retriever/slake/*.index`) if you want to run eval locally later
3. **Analyze results**:
   ```python
   import json
   with open('result/slake_full_predictions.json') as f:
       data = json.load(f)
   print(f"Accuracy: {data['accuracy']:.4f}")
   print(f"Total examples: {data['total_examples']}")
   ```

---

## Next Steps

- Compare SLAKE results vs VQA-RAD baseline
- Try hybrid retrieval (text + image) by modifying pipeline
- Experiment with different reranker strategies
- Document findings in `experiments/slake.md`

---

## Complete Colab Notebook Template

Save this as a `.ipynb` or run cells sequentially:

```python
# === Cell 1: Setup ===
from google.colab import drive
drive.mount('/content/drive')

# Unzip project
!unzip -q /content/drive/MyDrive/final-project.zip -d /content/

import os
os.chdir('/content/final-project')
!pwd

!pip install -q datasets sentence-transformers transformers torch pillow scikit-learn pyyaml huggingface-hub

# Verify faiss (usually pre-installed on Colab)
import faiss
print(f"FAISS version: {faiss.__version__}")

# === Cell 2: Prepare Data ===
!python scripts/prepare_slake.py --splits train validation test --limit 3000

# === Cell 3: Build Indexes ===
!python build_corpus.py --train-file dataset/slake/json_files/train.json \
  --index-file retriever/slake/text.index --texts-file retriever/slake/texts.json

!python build_image_index.py --img-dir dataset/slake/imgs \
  --train-file dataset/slake/json_files/train.json \
  --index-out retriever/slake/image.index \
  --metadata-out retriever/slake/image_texts.json

# === Cell 4: Evaluate ===
!python eval_slake.py --config configs/slake.yaml --split test --top-k 3 \
  --output result/slake_colab_predictions.json

# === Cell 5: Save Results ===
!cp result/slake_colab_predictions.json /content/drive/MyDrive/
print("✓ Results saved to Google Drive")

# === Cell 6: Show Results ===
import json
with open('result/slake_colab_predictions.json') as f:
    data = json.load(f)
print(f"Accuracy: {data['accuracy']:.4f}")
print(f"Config: {data['config']}")
print(f"Total examples: {data['total_examples']}")
```

Good luck! 🚀
