# Automated Terms & Conditions Summarizer and Clause Risk Analyzer

**BE Computer Engineering — Semester VII (Academic Year 2026)**  
**Course:** Natural Language Processing Lab  
**Group Number:** 35  

### Team Members:
- **Disha Takawale** — `2413195`
- **Prerana Patankar** — `2413196`
- **Saloni Patil** — `2413210`
- **Neha Maniyar** — `2413211`

---

## 📌 Project Overview
Digital consumers routinely accept lengthy, legally complex Terms and Conditions (T&C) agreements without reading them, unknowingly waiving trial rights, permitting unilateral account termination, and accepting broad liability disclaimers. 

This project provides an end-to-end NLP system that parses legal agreements, executes classical and advanced preprocessing pipelines, and performs **genuine fine-tuning of a pretrained Transformer (`BERT`)** to categorize contractual clauses into **Low**, **Medium**, and **High** risk levels with confidence scoring.

---

## 🗂️ Repository Structure

```text
├── docs/                               # 📚 Complete Step-by-Step Documentation & Learning Guides
│   ├── README.md                       # Documentation Index
│   ├── 01_project_overview_and_concepts.md  # Project Motivation & High-Level Architecture
│   ├── 02_mandatory_preprocessing_explained.md # 5 Mandatory Preprocessing Tasks Explained
│   ├── 03_dataset_and_task_formulation.md      # Dataset Details & Classification Rationale
│   ├── 04_llm_finetuning_deep_dive.md          # Transformer Fine-Tuning & Parameter Updates
│   ├── 05_evaluation_metrics_and_results.md    # Metrics, Baseline Benchmarks & Confusion Matrix
│   ├── 06_how_to_run_and_viva_qa.md            # Execution Steps & Top 15 Viva Q&A
│   └── 07_how_to_give_a_demo.md                # 3-Minute Live Physical Demonstration Script
│
├── model_output/                       # 💾 Trained Model Checkpoint & Artifacts
│   ├── model.safetensors               # Fine-tuned PyTorch neural network weights (~17MB)
│   ├── config.json                     # Model architecture configuration
│   ├── tokenizer.json                  # BertTokenizer vocabulary and rules
│   ├── evaluation_metrics.json         # Exact test set metrics & confusion matrix
│   └── sample_predictions.json         # Unseen clause predictions & probabilities
│
├── alibaba.txt                         # Raw Alibaba Free Membership Agreement
├── amazon.txt                          # Raw Amazon Conditions of Use
├── tc_clauses.csv                      # Seed dataset (17 clauses - untouched)
├── tc_clauses_expanded.csv             # Curated dataset (82 clauses across 11 platforms)
│
├── nlpExps(1-5).pdf                    # Official documentation for Experiments 1–5
├── hmm-posTagging-implementation.pdf   # Custom HMM POS-Tagger with Viterbi decoding
│
├── preprocessing_pipeline.py           # Verification script for 5 mandatory preprocessing steps
├── prepare_expanded_dataset.py         # Reproducible dataset preparation script
├── fine_tune_bert.py                   # Genuine BERT fine-tuning & evaluation script
├── inference.py                        # Standalone prediction script
├── demo.py                             # Interactive CLI live demonstration tool
├── generate_flowchart.py               # Flowchart diagram generation script
├── fine_tuning_flowchart.png           # High-resolution pipeline flowchart
│
├── FINAL_MINI_PROJECT_REPORT.md        # Complete academic technical mini-project report
├── PRESENTATION_SLIDES.md              # Project presentation slides
├── form_answers.md                     # Instructor's Microsoft Form submission answers
├── README.md                           # Master repository documentation (this file)
└── .gitignore                          # Git tracking ignore rules
```

---

## ⚙️ Quickstart & Execution

### 1. Install Requirements
```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn nltk
```

### 2. Verify Mandatory Preprocessing Tasks (Exp 1–5 compliance)
```bash
python preprocessing_pipeline.py
```
*Outputs sentence segmentation, tokenization, POS tags, POS-informed lemmatization, and WordNet semantic analysis.*

### 3. Run the Interactive Live Demo (For Viva / Presentation)
```bash
python demo.py
```
*Provides a menu to test pre-loaded Amazon/Alibaba clauses or paste any custom contract sentence live.*

### 4. Run Training & Evaluation Pipeline
```bash
python fine_tune_bert.py
```
*Fine-tunes BERT-Tiny for 12 epochs using AdamW, evaluates against classical TF-IDF Logistic Regression baseline, and outputs confusion matrix.*

---

## 📊 Key Results & Evaluation Benchmark

Evaluated on an isolated 15% stratified held-out test split (13 unseen contractual clauses):

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | High-Risk Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (TF-IDF + Logistic Regression)** | 61.54% | 0.6111 | 0.6111 | 0.6000 | 83.3% |
| **Fine-Tuned BERT-Tiny (4.4M Params)** | **61.54%** | **0.8485** *(+23.7%)* | 0.5278 | 0.5353 | **100.0%** |

- **Zero False Negatives on Hazardous Terms:** The fine-tuned transformer attained **100% Recall on High-Risk clauses**, ensuring dangerous liabilities, arbitration traps, or unilateral termination terms are always flagged.
- **Superior Precision:** Achieved **100% Precision on Low and Medium classes** with a **+23.7% precision improvement** over the baseline.

---

## 👥 Team & Responsibilities
- **Disha Takawale (2413195):** Dataset curation, dataset expansion, mandatory preprocessing pipeline, and PyTorch DataLoader integration.
- **Prerana Patankar (2413196):** Pretrained model selection (`bert-tiny`), fine-tuning pipeline design, AdamW optimization, and training convergence.
- **Saloni Patil (2413210):** Held-out test set evaluation, classical baseline benchmarking (TF-IDF + LogReg), and confusion matrix analysis.
- **Neha Maniyar (2413211):** Interactive demo development (`demo.py`), standalone inference, pipeline flowchart design, and report documentation.
