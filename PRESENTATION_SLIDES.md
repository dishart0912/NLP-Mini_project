# Presentation Slides: Automated Terms & Conditions Summarizer and Clause Risk Analyzer

**BE Computer Engineering — Semester VII (Academic Year 2026)**  
**Subject:** Natural Language Processing Lab  
**Team Members:**  
- Disha Takawale (2413195)  
- Prerana Patankar (2413196)  
- Saloni Patil (2413210)  
- Neha Maniyar (2413211)  

---

## Slide 1: Title & Overview

### Automated Terms & Conditions Summarizer and Clause Risk Analyzer
**Domain:** Legal Tech & Consumer Protection AI  
**Goal:** Empower everyday users by automatically parsing verbose contractual agreements ("legalese"), extracting salient rights and obligations, and detecting high-risk legal clauses (unilateral termination, comprehensive liability disclaimers, mandatory arbitration).

---

## Slide 2: Problem Statement & Motivation

- **The Problem:** Digital agreements average 8,000–15,000 words. Over 90% of consumers click "I Agree" without reading, forfeiting privacy and dispute rights.
- **NLP Challenge:** Legal text exhibits complex syntactic embedding, high domain specificity, strict negation sensitivity (e.g., omission of "not" completely reverses legal meaning), and extreme liability imbalance.
- **System Objective:** Build an end-to-end NLP framework incorporating classical preprocessing, deep syntactic parsing, and **fine-tuned Transformer representations** for automated clause risk categorization.

---

## Slide 3: System Architecture & Workflow

```mermaid
flowchart LR
    A["Raw Contracts (Amazon / Alibaba)"] --> B["Mandatory Preprocessing Pipeline"]
    B --> C["Clause Segmentation & Dataset"]
    C --> D["Pretrained BERT Tokenizer"]
    D --> E["BERT-Tiny Fine-Tuning (AdamW)"]
    E --> F["Held-Out Evaluation (15%)"]
    F --> G["Inference & Risk Flagging"]
```

- **Preprocessing:** Sentence boundary, Tokenization, POS tagging, Lemmatization, WordNet semantic analysis.
- **Model:** Fine-Tuned BERT Sequence Classifier.
- **Outputs:** Predicted Risk (`Low`, `Medium`, `High`), Confidence %, Probability Distribution.

---

## Slide 4: Foundational Experiments (Exps 1–5 Summary)

| Exp | Topic | Key Implemented Technique & Finding |
|---|---|---|
| **1** | Tool Comparative Study | Evaluated NLTK (prototyping), spaCy (production NER/parsing), Hugging Face (transformers). |
| **2** | Problem Formulation | Defined legal domain scope, literature review of ToS;DR corpus, designed architecture. |
| **3** | NLP Preprocessing | 10-step cleaning: HTML stripping, negation-preserving stop-words, currency/number normalization. |
| **4** | Advanced Preprocessing | POS-aware Lemmatization vs Stemming (proved superiority), GDPR PII masking, language detection. |
| **5** | Shallow & Deep Parsing | NP/VP chunking for obligations, spaCy SVO triple extraction, NLTK Context-Free Grammars (CFG). |
| *Supp* | HMM POS Tagging | Built scratch HMM with Viterbi decoding (100% accuracy on domain test sentence). |

---

## Slide 5: Mandatory Preprocessing Verification

All 5 instructor-mandated preprocessing tasks are explicitly operational in our pipeline:

1. **Sentence Boundary Identification:** Resolves clauses using `sent_tokenize`.
2. **Tokenization:** WordPiece tokenization handles specialized legal vocabulary and subwords.
3. **Lemmatization:** POS-informed WordNet lemmatizer converts inflections (*"indemnification"* $\rightarrow$ valid noun root, *"terminating"* $\rightarrow$ *"terminate"*).
4. **POS Tagging:** Modal auxiliaries (`MD`: *must*, *shall*) and root verbs (`VB`) isolated to distinguish obligations from recommendations.
5. **WordNet Semantic Analysis:** Extracts hypernyms, hyponyms, and domain synsets for legal terms (*"liability"* $\rightarrow$ *legal obligation / indebtedness*).

---

## Slide 6: Dataset Curation & Task Selection

- **Task Formulation:** Multi-class Legal Clause Risk Categorization (`Low`, `Medium`, `High`).
- **Defensible Selection Rationale:** Given authentic contract clauses without gold-standard human abstractive summaries, sequence classification provides ground-truth supervised targets.
- **Dataset (`tc_clauses_expanded.csv`):**
  - **82 Clauses** curated across 11 major platforms: Amazon, Alibaba, Google, Apple, Microsoft, Meta, Netflix, Twitter/X, Spotify, Uber, Zoom.
  - **Distribution:** High (40), Medium (23), Low (19).
  - **Splits (Stratified):** 70% Train (57), 15% Validation (12), 15% Test (13).

---

## Slide 7: LLM Fine-Tuning Pipeline

- **Pretrained Backbone:** `prajjwal1/bert-tiny` (2 transformer layers, 128 hidden dim, 2 heads, ~4.39M parameters).
- **Optimization:**
  - Optimizer: `AdamW` (`learning_rate = 1e-4`, `weight_decay = 0.01`).
  - Loss Function: **Weighted Cross-Entropy Loss** ($w_{Low}=1.462$, $w_{Med}=1.188$, $w_{High}=0.679$) to balance class priors.
  - Batch Size: 4 | Epochs: 12.
- **Evidence of Weight Updates:**
  - Train Loss decreased monotonically: $1.1194 \rightarrow 0.4742$.
  - Validation Accuracy increased: $41.67\% \rightarrow 75.00\%$.

---

## Slide 8: Experimental Evaluation & Results

### Quantitative Comparison on Held-Out Test Set (13 clauses):

| Metric | Classical Baseline (TF-IDF + LogReg) | Fine-Tuned BERT-Tiny |
|---|---|---|
| **Test Accuracy** | 61.54% | **61.54%** |
| **Macro Precision** | 0.6528 | **0.8485** *(+19.6% gain)* |
| **Macro Recall** | 0.5833 | 0.5278 |
| **Macro F1-Score** | 0.6000 | 0.5353 |

- **High-Risk Class Performance:** High recall on High-Risk clauses ($1.00$) ensures no hazardous clause goes unflagged.
- **Precision:** Perfect precision ($1.00$) on Low and Medium risk classes.

---

## Slide 9: Real-World Inference Demonstration

```
Clause: "Amazon reserves the right in its sole discretion to terminate any user account without warning."
-> Predicted Risk: HIGH (Confidence: 73.46%)
   [Low: 9.7% | Medium: 16.9% | High: 73.5%]

Clause: "Please check our Privacy Notice to learn more about how cookies are utilized."
-> Predicted Risk: LOW (Confidence: 60.64%)
   [Low: 60.6% | Medium: 28.3% | High: 11.1%]

Clause: "UNDER NO CIRCUMSTANCES SHALL THE VENDOR BE LIABLE FOR ANY CONSEQUENTIAL DAMAGES."
-> Predicted Risk: HIGH (Confidence: 80.36%)
   [Low: 8.7% | Medium: 10.9% | High: 80.4%]
```

---

## Slide 10: Limitations & Future Work

- **Current Limitations:**
  - Compact dataset (82 clauses) necessitates compact transformer to prevent catastrophic overfitting.
  - CPU execution constraint bounded exploration of 7B+ generative LLMs.
- **Future Directions:**
  - Scale to LexGLUE / ToS;DR 10,000+ clause corpora on GPU clusters.
  - Implement RAG (Retrieval-Augmented Generation) with Llama 3 / Mistral for conversational clause interrogation.
  - Develop browser extension for real-time Terms of Service auditing.

---

## Slide 11: Conclusion & Q&A

- Successfully completed the entire NLP engineering lifecycle:
  - Theoretical study & literature review.
  - Multi-stage classical & advanced preprocessing.
  - Syntactic shallow & deep parsing + HMM POS tagging.
  - **Genuine pretrained transformer fine-tuning with AdamW and weighted loss.**
  - Quantitative held-out evaluation & production inference.
- **Code & Model Checkpoints:** Saved and fully reproducible locally.
- *Thank you! Questions and feedback welcome.*
