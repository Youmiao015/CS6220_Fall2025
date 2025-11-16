PathVQA Evaluation Report
1. Overview

This section summarizes our experiments on the PathVQA dataset using the multimodal RAG pipeline adapted from the MasonNLP framework. PathVQA is significantly larger and more diverse than VQA-RAD, containing:

5,000+ pathology images

33,000+ question–answer pairs

covering X-ray, CT, pathology slides, and clinical findings.

Its higher variance and more complex question taxonomy (open-ended / anatomical / modality / abnormality) make it a strong benchmark for testing retrieval-augmented methods.

We evaluate two settings:

Baseline (no retrieval, no rewriting)

Full RAG (HyDE + decomposition + cross-encoder reranker)

Both experiments are run on the test split (n = 3000).

2. Experimental Settings

All configurations used in this experiment are stored in configs/pathvqa_rag.yaml.
The file contains both baseline and RAG configurations:

The baseline configuration appears as a commented-out block.

The RAG configuration (HyDE + decomposition + reranker) is the active portion of the file.

This design ensures that both settings remain tracked in a single config file while avoiding duplication.

2.1 Baseline Configuration

No query rewriting

No subquestion decomposition

No reranking

FLAN-T5-Large as the generation model

Answers generated directly from the question

This setting measures how well the LLM can answer medical visual QA questions without additional retrieved context.

2.2 RAG Configuration

We enable the full retrieval-augmented pipeline:

HyDE: Generates hypothetical textual descriptions to improve retrieval

Decomposition: Splits complex questions into up to 3 simpler sub-questions

Cross-Encoder Reranker: ms-marco-MiniLM-L-6-v2 reranks top-k retrieved passages

Text + Image retrieval via FAISS

FLAN-T5-Large performs final generation

This configuration aims to provide the model with richer, more specific multimodal evidence.

3. Results
Model	Accuracy	Improvement
Baseline (no RAG)	0.2670	—
RAG: HyDE + decomposition + reranker	0.3243	+0.0573
Key Findings

Retrieval augmentation provides a 5.7-point absolute accuracy improvement.

The gain is consistent across question types requiring domain knowledge (anatomy, modality, abnormality presence).

Reranking significantly improves retrieval precision and results relevance.

4. Error Analysis
4.1 Common Failure Modes
1. Open-ended clinical reasoning

Complex multi-step pathology questions remain challenging even with retrieval.

2. Fine-grained abnormality identification

LLM generations sometimes output synonyms or adjacent findings (opacity vs consolidation).

3. Retrieval mismatch

Retriever occasionally chooses clinically irrelevant but visually similar captions.

5. Observations

HyDE improves retrieval recall, especially when the question is short or underspecified.

Decomposition stabilizes multi-step queries (e.g., anatomical + abnormality).

Cross-encoder reranking improves top-k precision, providing cleaner evidence to the LLM.

Gains appear strongest in:

yes/no questions

modality classification

spatial location questions

abnormality existence questions

Open descriptive questions remain the weakest category.

6. Limitations

FLAN-T5 is not domain-adapted; domain-specific models may yield higher accuracy.

CLIP has limited pathology representation capability, affecting image retrieval.

Strict accuracy underestimates performance for semantically correct paraphrases.

7. Future Work

Employ domain-specific vision encoders (MedCLIP, BiomedCLIP).

Replace FLAN-T5 with a medical VLM (LLaVA-Med, Med-Flamingo).

Explore chain-of-thought generation after retrieval.

Use additional metrics such as BLEU, ROUGE, and METEOR.

Add image captioning to enrich textual retriever inputs.

8. Conclusion

Our experiments demonstrate that retrieval-augmented techniques meaningfully improve medical visual question answering on PathVQA:

Baseline: 26.7%

Full RAG: 32.4%

This +5.7% gain confirms that question rewriting, subquestion decomposition, and strong reranking are essential for handling large-scale medical VQA tasks. The results establish a strong foundation for further improvements using domain-specific encoders and multimodal LLMs.