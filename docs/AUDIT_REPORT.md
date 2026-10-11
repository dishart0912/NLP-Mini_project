# Initial Repository Audit Report

**Date:** October 2026  
**Auditor:** Senior NLP/ML Engineer & Academic Project Auditor  
**Project:** Automated Terms & Conditions Summarizer and Clause Risk Analyzer  

---

## 1. Executive Summary

This audit assesses the code, artifacts, datasets, and documentation in this repository against the October 2026 academic guidelines.

### Overall Finding:
The core implementation of BERT fine-tuning, dataset stratification, preprocessing demonstrations, and interactive inference is genuine and functional. However, the audit identified specific implementation bugs, documentation discrepancies, and methodological limitations that must be addressed to ensure the project is technically sound, reproducible, and ready for viva examination.

---

## 2. Classification of Audit Findings

### Category 1: Confirmed Implementation Bugs
1. **Checkpoint Selection Bug in `fine_tune_bert.py`:**
   - *Issue:* The training loop ran for 12 epochs and saved the final epoch model directly to `model_output/` without selecting the best validation checkpoint.
   - *Impact:* While epoch 12 achieved the best validation loss in this particular run, best-checkpoint selection (monitoring validation loss) is required by machine learning best practices to prevent overfitting.
2. **Missing `requirements.txt`:**
   - *Issue:* No dependency manifest file exists. While setup commands were provided in Markdown files, automated reproducibility requires an explicit `requirements.txt`.
3. **Empty Input Handling in `inference.py`:**
   - *Issue:* `inference.py` assumes a non-empty string input and does not gracefully handle empty strings or strings consisting solely of whitespace or special characters before tokenization.
4. **Encoding Fragility on Windows Consoles:**
   - *Issue:* `demo.py` and `show_results.py` previously contained Unicode symbols that threw `UnicodeEncodeError` under Windows default CP1252 code pages. Although patched, this needs formal unit/smoke testing.

### Category 2: Documentation Inaccuracies & Discrepancies
1. **Metric Discrepancy Between JSON and Documentation:**
   - *Baseline Precision:* `evaluation_metrics.json` records baseline macro precision as **0.6528**, while several markdown files reported **0.6111**.
   - *Precision Gain Claim:* The markdown files claimed a **+23.7%** precision gain based on 0.6111, whereas the actual gain over 0.6528 is **+19.6%** (0.8485 vs 0.6528).
2. **Data Provenance & Source File Attribution:**
   - *Issue:* Documentation occasionally implied that `amazon.txt` and `alibaba.txt` are parsed at runtime by `prepare_expanded_dataset.py`.
   - *Reality:* `prepare_expanded_dataset.py` contains hardcoded Python dictionary lists representing authentic clauses manually extracted from those agreements and other tech services. The raw `.txt` files are source reference documents, not dynamically parsed inputs.
3. **Project Title vs Functionality Mismatch:**
   - *Issue:* The project title contains **"Summarizer"**, but the fine-tuned LLM is exclusively a 3-class **Clause Risk Classifier**. No text-to-text abstractive or extractive summarizer model was trained.
   - *Resolution Required:* The documentation must explicitly clarify that while clause extraction and risk analysis serve as an extractive filtering mechanism for long contracts, the fine-tuned LLM is specifically a sequence classification model for clause risk categorization.
4. **Separation of NLTK Preprocessing and BERT Tokenizer:**
   - *Issue:* Some guides could be misinterpreted as implying that NLTK lemmatization and WordNet outputs feed directly into BERT.
   - *Reality:* BERT uses its own subword WordPiece tokenizer (`BertTokenizer`). NLTK preprocessing is an independent linguistic demonstration module. This architectural distinction must be prominently stated.

### Category 3: Evaluation & Methodology Limitations
1. **Small Dataset & Test Set Limitations:**
   - Total dataset size is **82 clauses**. The 15% stratified held-out test set contains only **13 clauses** (6 High, 4 Medium, 3 Low).
   - Because the test set has only 13 samples, each misclassification changes accuracy by ~7.7%.
   - The model achieved 1.00 recall on High-Risk test samples (6/6), but claiming "zero false negatives under all circumstances" is statistically invalid given the small test sample size.
2. **Generalization Across Companies:**
   - A random stratified split distributes clauses from the same platform across train and test sets. It does not establish domain transfer or generalization to entirely unseen companies. Group-based (Leave-One-Company-Out) evaluation limitations must be clearly discussed.
3. **Confidence Scores as Model Logits vs Calibrated Probabilities:**
   - Softmax outputs represent uncalibrated model confidence over 3 classes, not legal certainty.

### Category 4: Optional Improvements
1. Automated smoke test suite (`test_pipeline.py`) validating dataset integrity, preprocessing pipeline, model loading, and inference.
2. Dynamic clause counting in `prepare_expanded_dataset.py` rather than hardcoded prints.
3. Explicit `requirements.txt` with tested version bounds.

---

## 3. Remediation Plan

1. **Dataset (`prepare_expanded_dataset.py`):** Add schema validation, duplicate checks, dynamic row counting, and clear provenance comments.
2. **Preprocessing (`preprocessing_pipeline.py`):** Add input validation, graceful missing-corpus handling, and documentation on WordNet's legal terminology limitations.
3. **Training & Checkpointing (`fine_tune_bert.py`):** Implement best-checkpoint saving (monitoring validation loss), record training configuration, seed, and save the best model weights.
4. **Testing & Inference (`inference.py`, `demo.py`, `tests/`):** Add smoke tests verifying all components; ensure empty/invalid inputs are handled cleanly.
5. **Documentation Alignment:** Synchronize every metric across `evaluation_metrics.json`, `README.md`, `FINAL_MINI_PROJECT_REPORT.md`, `PRESENTATION_SLIDES.md`, `form_answers.md`, and all `docs/*.md` files.
