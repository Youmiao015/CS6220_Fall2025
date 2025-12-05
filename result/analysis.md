# Prediction Analysis: Baseline vs Stage2

## Prediction Examples Comparison

| Approach | Status | Question ID | Question | Gold Answer | Prediction | Analysis |
|----------|--------|-------------|----------|-------------|------------|----------|
| **Baseline**<br>(48.52%) | ✅ Correct | `slake_test_11934` | "What modality is used to take this image?" | "CT" | "CT" | Simple, direct retrieval worked well for this factual question type. |
| **Baseline**<br>(48.52%) | ❌ Failed | `slake_test_11936` | "What is the main organ in the image?" | "Lung, Spinal Cord" | "Lung, Heart" | Partial match - correctly identified "Lung" but missed "Spinal Cord" and incorrectly included "Heart". Simple similarity matching failed to capture complete multi-organ answer. |
| **Stage2**<br>(43.55%) | ✅ Correct | `slake_test_11938` | "Does the picture contain liver?" | "No" | "No" | HyDE query rewriting and cross-encoder reranking helped retrieve relevant exemplars for this binary classification task. |
| **Stage2**<br>(43.55%) | ❌ Failed | `slake_test_11934` | "What modality is used to take this image?" | "CT" | "MRI" | Same question that baseline got correct. HyDE rewriting may have generated hypothetical answers leading to MRI examples, causing reranker to select wrong modality. Query rewriting can introduce noise for straightforward factual questions. |

---

## Key Observations

1. **Baseline performs better overall** (48.52% vs 43.55%), suggesting that simpler retrieval can be more effective for this dataset.

2. **Query rewriting can backfire**: The same question (11934) that baseline answered correctly was answered incorrectly by Stage2, indicating that HyDE rewriting may introduce retrieval errors for direct factual questions.

3. **Multi-part answers are challenging**: Both approaches struggle with questions requiring multiple correct answers (e.g., "Lung, Spinal Cord").

4. **Closed-ended questions**: Both approaches handle yes/no questions reasonably well, though Stage2 shows some inconsistency (correct on 11938, wrong on 11939).

5. **Trade-off**: Stage2's more complex pipeline (rewriting + reranking) adds computational overhead but doesn't consistently improve accuracy, suggesting the baseline's direct similarity matching is well-suited for this medical VQA task.
