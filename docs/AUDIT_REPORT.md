# Comprehensive Final Audit Report

**Date:** October 2026  
**Auditor:** Senior NLP/ML Engineer & Academic Project Auditor  
**Project:** NLP-Based Terms & Conditions Clause Risk Analyzer Using BERT-Tiny  
*(Registered Topic: Automated Terms & Conditions Summarizer and Clause Risk Analyzer)*  
**Academic Term:** BE Computer Engineering, Semester VII (Academic Year 2026)  
**Group Number:** 35  

---

## 1. Executive Summary

This comprehensive audit evaluates the implementation, reproducibility, empirical findings, and documentation of the Semester VII NLP Mini-Project.

### Overall Status: **PASSED / FULLY ALIGNED**
All 5 mandatory NLP preprocessing experiments, genuine BERT-Tiny fine-tuning with backpropagation and best-checkpoint selection, classical baseline benchmarking, unit testing, and documentation have been rigorously verified. All prior discrepancies regarding data provenance, risk-detection claims, metric synchronization, and project scope have been completely resolved.

---

## 2. Verification & Validation Matrix

| Area / Component | Verification Performed | Expected Behavior / Artifact | Verified Result | Status |
|:---|:---|:---|:---|:---:|
| **Dataset Provenance** | Audited `prepare_expanded_dataset.py`, `tc_clauses.csv`, and documentation | `tc_clauses.csv` preserved untouched (17 rows); `tc_clauses_expanded.csv` generated via verified list (82 rows); no claim of dynamic `.txt` scraping | 17 seed rows intact; 82 expanded rows verified; provenance documented as student curation | **PASS** |
| **Dataset Hygiene** | Checked for nulls, duplicates, empty text, and invalid labels | No duplicate IDs, 0 null cells, non-empty text, valid labels: `Low`, `Medium`, `High` | Clean schema validated across all 82 rows by automated test suite | **PASS** |
| **Preprocessing 1: Sentence Boundary** | Verified `sent_tokenize` on multi-sentence legal text in `preprocessing_pipeline.py` | Accurate clause boundary identification | Correctly splits compound agreements | **PASS** |
| **Preprocessing 2: Tokenization** | Verified lexical tokenization (`word_tokenize`) and subword (`BertTokenizer`) | Preserves punctuation, handles subwords via WordPiece | Verified in test suite and demo | **PASS** |
| **Preprocessing 3: POS Tagging** | Verified `nltk.pos_tag` and scratch HMM tagger from Experiment 5 | Modal auxiliaries (`MD`: *shall*, *must*) and verbs identified | Validated on domain sentences | **PASS** |
| **Preprocessing 4: Lemmatization** | Verified POS-aware `WordNetLemmatizer` | Preserves valid legal roots (e.g., *terminating* -> *terminate*, *liabilities* -> *liability*) | Validated against baseline stemmers | **PASS** |
| **Preprocessing 5: WordNet Ontology** | Verified synsets, definitions, hypernyms, hyponyms extraction | Ontological extraction for contractual terms (*liability*, *terminate*) | Domain synsets and broader concepts extracted | **PASS** |
| **Fine-Tuning Architecture** | Verified `prajjwal1/bert-tiny` parameters in PyTorch | 2 layers, 128 hidden dim, 2 heads, ~4.39M parameters | Exact count: 4,386,307 parameters | **PASS** |
| **Loss & Optimization** | Inspected AdamW and loss function in `fine_tune_bert.py` | Weighted Cross-Entropy Loss to balance class priors; AdamW optimizer | Weights: Low=1.462, Med=1.188, High=0.679; LR=1e-4 | **PASS** |
| **Checkpoint Selection** | Verified checkpoint saving mechanism during 12 epochs | Selects best model based on validation loss, not merely final epoch | Best checkpoint saved at Epoch 12 (Val Loss: 0.6780, Val Acc: 75.00%) | **PASS** |
| **Weight Serialization** | Checked `model_output/` artifacts | Model weights, config, tokenizer files saved to disk | `model.safetensors` (16.74 MB), `config.json`, `tokenizer.json` intact | **PASS** |
| **Baseline Benchmarking** | Audited TF-IDF + Logistic Regression implementation | Trained on identical 70% train split (57 rows), evaluated on 15% test split (13 rows) | Exact baseline metrics: Acc: 61.54%, Macro Prec: 0.6528, Macro Rec: 0.5833, Macro F1: 0.6000 | **PASS** |
| **Model Evaluation** | Compared fine-tuned BERT against baseline on 13-clause test set | Macro Precision gain (+19.6%), 100% High-Risk recall, exact confusion matrix | Acc: 61.54%, Macro Prec: 0.8485, Confusion Matrix: `[[1,0,2],[0,1,3],[0,0,6]]` | **PASS** |
| **Risk Detection Language** | Audited all documentation and print statements | Removed unsupported guarantees ("zero false negatives", "guaranteed detection") | Replaced with honest evaluation on 13-clause test set and clear limitations note | **PASS** |
| **Title & Scope Alignment** | Checked README, reports, slides, form draft | Recommended descriptive title; clarified that document summarization is future work | Reconciled across all project files | **PASS** |
| **Automated Test Suite** | Executed `test_pipeline.py` | 8 automated unit tests covering datasets, preprocessing, model loading, and inference | **8 / 8 tests PASSED** in 8.45s | **PASS** |
| **Platform Portability** | Checked Windows console output encoding | Plain ASCII formatting; no Unicode crashes on CP1252 code pages | Verified in PowerShell environment | **PASS** |

---

## 3. Detailed Audit Findings & Remediation Summary

### 3.1 Initial Issues Identified & Remediation

1. **Dataset Provenance Clarity:**
   - *Initial Issue:* Documentation previously implied that `amazon.txt` and `alibaba.txt` were dynamically parsed at runtime.
   - *Verified Reality & Fix:* `prepare_expanded_dataset.py` contains student-curated clauses defined in Python data structures and merged with `tc_clauses.csv`. The raw `.txt` files are static reference source documents. All documentation has been updated to reflect this honest provenance.
2. **Unsupported Risk Guarantees:**
   - *Initial Issue:* Claims of "Zero false negatives on hazardous terms under all circumstances" and "guaranteeing no dangerous clause slips past undetected" were statistically unjustified given a 13-sample test split.
   - *Verified Fix:* Language updated across `README.md`, `FINAL_MINI_PROJECT_REPORT.md`, `PRESENTATION_SLIDES.md`, `show_results.py`, and `docs/*.md`. The report now accurately states: *"The model correctly classified all 6 High-labeled clauses in the current 13-example test set. Because the dataset and test set are small, this result does not establish reliable detection of all high-risk clauses in real-world agreements."*
3. **Project Title vs. Functionality Mismatch:**
   - *Initial Issue:* Registered title included "Summarizer", while the implemented model is a 3-class clause risk classifier.
   - *Verified Fix:* Recommended descriptive title *"NLP-Based Terms & Conditions Clause Risk Analyzer Using BERT-Tiny"* while citing the officially registered title. Clarified that clause risk classification serves as an extractive risk triage tool, while generative text summarization is proposed as future work.
4. **Metric Synchronization:**
   - *Initial Issue:* Baseline Macro Precision was cited as both 0.6111 and 0.6528 across different markdown files.
   - *Verified Fix:* Standardized all references to exact values from `model_output/evaluation_metrics.json`:
     - Baseline Macro Precision: **0.6528**
     - Fine-Tuned BERT Macro Precision: **0.8485**
     - Absolute Macro Precision Gain: **+19.6%** (0.8485 vs 0.6528)
5. **Best Validation Checkpoint Selection:**
   - *Initial Issue:* Training loop saved only the final epoch without tracking best validation loss.
   - *Verified Fix:* Implemented checkpoint selection in `fine_tune_bert.py` monitoring `val_loss`. Best weights saved at Epoch 12 (Val Loss: 0.6780).
6. **Edge Case & Encoding Robustness:**
   - *Initial Issue:* Empty input strings caused unhandled exceptions in inference and preprocessing; Windows CP1252 consoles failed on special Unicode box-drawing characters.
   - *Verified Fix:* Added empty/whitespace guardrails in `inference.py` and `preprocessing_pipeline.py`. Converted console output in `demo.py` and `show_results.py` to robust ASCII characters.

---

## 4. Empirical Evaluation Summary

### 4.1 Quantitative Results on Held-Out Test Set (13 Clauses)

```text
========================================================================================
Model Architecture                 Accuracy   Macro Prec   Macro Rec   Macro F1   High Rec
========================================================================================
Baseline (TF-IDF + LogReg)          61.54%      0.6528       0.5833     0.6000     83.3%
Fine-Tuned BERT-Tiny (4.4M Params)  61.54%      0.8485       0.5278     0.5353    100.0%
========================================================================================
Absolute Difference / Gain           0.00%     +19.6% GAIN   -5.55%     -6.47%    +16.7%
========================================================================================
```

### 4.2 Confusion Matrix (Fine-Tuned BERT-Tiny)
$$\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 3 \\ 0 & 0 & 6 \end{pmatrix}$$
*(Rows = Ground Truth [Low, Medium, High]; Columns = Predicted [Low, Medium, High])*

- **Low-Risk Class (3 samples):** 1 True Positive, 2 False High. Precision = 1.00, Recall = 0.33.
- **Medium-Risk Class (4 samples):** 1 True Positive, 3 False High. Precision = 1.00, Recall = 0.25.
- **High-Risk Class (6 samples):** 6 True Positives, 0 False Low/Medium. Precision = 0.55, Recall = 1.00.

---

## 5. Known Limitations & Recommendations for Future Work

1. **Test Set Scale:** The test split contains 13 samples. While sufficient for a BE mini-project demonstration, statistical confidence intervals remain wide (each sample represents ~7.7% accuracy).
2. **Cross-Company Generalization:** Stratified splitting was performed at the clause level rather than group-level (by platform). A Leave-One-Company-Out (LOCO) evaluation on a larger legal corpus (e.g., ToS;DR or LexGLUE) is recommended for production deployment.
3. **Generative Summarization Scope:** End-to-end abstractive summarization requires paired clause-summary datasets and larger generative models (e.g., T5, BART, or instruction-tuned LLMs), which require dedicated GPU resources.

---

## 6. Final Certification

The codebase, test suite, and documentation are **technically accurate, fully reproducible, and aligned with academic lab requirements**. The repository is certified ready for viva presentation and GitHub submission.
