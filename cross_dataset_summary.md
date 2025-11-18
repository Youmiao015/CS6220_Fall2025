# Cross-Dataset Results Summary

This document summarizes our Medical VQA experiments across **three datasets**:

- **VQA-RAD** – radiology images with short, often binary or factual questions  
- **PathVQA** – pathology images with more open-ended, noisy language  
- **SLAKE** – semantically labeled, ontology-style medical VQA

For each dataset we compare:

- **Baseline**: vanilla VQA model **without** retrieval, query rewriting, reranking, or majority voting  
- **Stage 2 Hybrid (RAG)**: our config-driven pipeline with
  - query rewriting (HyDE + question decomposition)  
  - MiniLM cross-encoder reranker  
  - majority voting over top-k exemplars

The goal of this file is to make our **group results readable at a glance** and highlight where the hybrid design helps and where it hurts.

---

## 1. High-Level Metrics

### 1.1 Accuracy Overview

| Dataset   | Metric Type          | Baseline | Stage 2 Hybrid | Δ (Hybrid – Base) |
|----------|----------------------|----------|----------------|-------------------|
| VQA-RAD  | Strict accuracy      | 0.3503   | 0.3525         | +0.0022           |
| VQA-RAD  | Relaxed (semantic)\* | 0.3681   | 0.3703         | +0.0022           |
| PathVQA  | Exact accuracy       | 0.267    | 0.324          | +0.057            |
| SLAKE    | Exact accuracy       | 0.4852   | 0.4355         | −0.0497           |

\*Relaxed = counts clearly equivalent answers (e.g., *“axial”* vs *“axial plane”*, *“chest x-ray”* vs *“x-ray”*).

**Takeaway:**  
- Hybrid RAG gives **small gains on VQA-RAD**, **larger gains on PathVQA**, and **a noticeable drop on SLAKE**.  
- This suggests RAG is **dataset-dependent** rather than universally better.

---

## 2. VQA-RAD Results

### 2.1 Setup

- Small radiology VQA dataset (451 test samples in our split)  
- Short questions focused on:
  - presence/absence (*“is the heart enlarged?”*)  
  - localization (*“left or right side?”*)  
  - anatomy, modality, and simple measurements  
- Evaluated both with **strict exact match** and with a **relaxed semantic view** that merges near-equivalent answers.

### 2.2 Metrics

- **Strict accuracy**
  - Baseline: **0.3503**
  - Stage 2 Hybrid: **0.3525**
- **Relaxed (semantic) accuracy**
  - Baseline: **0.3681**
  - Stage 2 Hybrid: **0.3703**

So the hybrid model improves accuracy by **~0.22 percentage points** under both views.

### 2.3 Error Distribution (Baseline vs Stage 2)

On a manually analyzed subset, we categorized baseline errors:

- Binary (yes/no) questions – **36%**
- Pathology identification – **24%**
- Localization / laterality – **18%**
- Anatomical structure – **12%**
- Modality / sequence – **6%**
- Quantification – **4%**

With Stage 2 Hybrid, the share of errors shifts roughly to:

- Binary – **32%** (−4)
- Pathology – **20%** (−4)
- Localization – **16%** (−2)
- Anatomy – **14%** (+2)
- Modality – **10%** (+4)

### 2.4 Interpretation

- Hybrid RAG **helped exactly where we cared most**:
  - fewer mistakes on **yes/no** and **pathology** questions  
  - fewer localization errors (left vs right, etc.)
- At the same time, it caused **slightly more** confusion for anatomy labels and imaging sequences.

Even though the net accuracy improvement is small, the qualitative fixes (e.g., correcting *“is the heart enlarged?”* from “yes” to “no”) are clinically meaningful and show that the pipeline is nudging the model in the right direction.

---

## 3. PathVQA Results

### 3.1 Setup

- Pathology-focused VQA dataset with:
  - 5k+ images, ~33k Q–A pairs  
  - Longer, noisier questions and answers than VQA-RAD  
  - Many open-ended or descriptive answers

This environment is a better stress test for retrieval-augmented methods because there is more lexical variety and more room for exemplars to help.

### 3.2 Metrics

- **Exact accuracy (test, n ≈ 3000)**  
  - Baseline (no RAG): **0.267**  
  - Stage 2 Hybrid: **0.324**

That’s a **+5.7 percentage-point** improvement, much larger than on VQA-RAD.

### 3.3 Where Hybrid Helps

Based on breakdowns from our teammate’s PathVQA report:

- Strongest gains on:
  - **yes/no questions** (e.g., “Is there necrosis?”)  
  - **modality / stain type**  
  - **spatial location** within pathology slides  
  - **abnormality existence** (lesion present / absent)
- The combination of:
  - extra query variants from **HyDE + decomposition**, and  
  - the **MiniLM reranker**  
  helps surface training examples that mention similar tissue types, stains, or lesion patterns.

### 3.4 Interpretation

- PathVQA is **noisier and more diverse** than VQA-RAD.  
- In this setting, **more retrieval and more voting** tends to help, because:
  - there are many different ways to phrase the same concept;  
  - exemplars often contain useful patterns to copy or adapt.
- PathVQA is the clearest evidence that our hybrid design is capable of **meaningful real gains**, not just tiny numerical bumps.

---

## 4. SLAKE Results

### 4.1 Setup

- SLAKE is a **semantically labeled** Med-VQA dataset:
  - bilingual questions  
  - ontology-style answer space  
  - explicit entity and relation labels
- Questions are more structured and often tightly tied to the ontology.

### 4.2 Metrics

- **Exact accuracy (test)**  
  - Baseline: **0.4852**  
  - Stage 2 Hybrid: **0.4355**  
  - Net change: **−0.0497** (about a 5-point drop)

### 4.3 Interpretation

- Unlike VQA-RAD and PathVQA, SLAKE already **matches neatly to a small, clean label space**.
- Our aggressive RAG configuration (HyDE + decomposition + many exemplars + voting) seems to:
  - introduce **noise** into the retrieval set, and  
  - push the model toward **overly generic** or irrelevant exemplars.
- Majority voting can then **amplify the wrong cluster** of labels, lowering accuracy.

SLAKE is an important negative result: it shows that **RAG is not automatically beneficial** and that our pipeline must be **tuned per dataset** (or even turned off).

---

## 5. What the Cross-Dataset Comparison Tells Us

### 5.1 When Hybrid RAG Helps

Hybrid RAG is most promising when:

- The dataset has **noisy, free-form language** (VQA-RAD, PathVQA).  
- Many answers are **semantically similar but lexically different**.  
- There are enough candidate exemplars that retrieving multiple neighbors and voting can smooth out noise.

In those conditions:

- Query rewriting (HyDE + decomposition) increases the chance of hitting good neighbors.  
- Cross-encoder reranking filters out the worst matches.  
- Majority voting stabilizes predictions.

### 5.2 When Hybrid RAG Hurts

On structured datasets like SLAKE:

- A simple baseline that directly predicts from the image + question can be **more reliable**.  
- Extra rewrites and big candidate pools bring in **distractors**, and majority voting may follow the crowd rather than the gold label.

This suggests we need **dataset-aware RAG**:

- Possibly:
  - **full RAG** for noisy datasets,  
  - **light or no RAG** for clean ontology-driven datasets.

---

## 6. Next Steps

From this cross-dataset view, the most natural future directions are:

1. **Adaptive RAG policies**
   - Decide per dataset (or even per question type) whether to use:
     - no RAG,  
     - minimal retrieval, or  
     - full HyDE + decomposition + reranking + voting.

2. **Better evaluation**
   - Move beyond strict exact match by using:
     - ontology-aware matching (e.g., synonyms, parent/child concepts),  
     - small human review samples to check clinical reasonableness.

3. **Stronger components inside the same framework**
   - Swap in:
     - larger or domain-tuned LLMs for rewriting,  
     - better vision encoders for retrieval,  
     - LLM-based rerankers instead of (or in addition to) MiniLM.

---
