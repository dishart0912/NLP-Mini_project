# Team Learning & Project Documentation Index

Welcome! This folder contains clean, structured documentation covering everything implemented in our NLP Mini-Project.

**Project Title:** Automated Terms & Conditions Summarizer and Clause Risk Analyzer  
*(Descriptive Academic Scope: NLP-Based Terms & Conditions Clause Risk Analyzer Using BERT-Tiny)*  

**Group 35 Members:**
- **Disha Takawale** (2413195)
- **Prerana Patankar** (2413196)
- **Saloni Patil** (2413210)
- **Neha Maniyar** (2413211)

---

## 📚 Essential Documentation Roadmap

To avoid confusion and redundancy, the documentation is organized into clear, focused guides:

### 1. 🌟 Master Individual Study & Viva Guide (Start Here!)
- **[`DEEP_CONTRIBUTION_AND_VIVA_GUIDE.md`](DEEP_CONTRIBUTION_AND_VIVA_GUIDE.md)**  
  *The primary study guide for all 4 members. Contains in-depth technical breakdowns of each member's contributions, slide speaking roles, line-by-line implementation details, mathematical formulas, and tough cross-examination questions for viva defense.*

---

### 2. Topic-Specific Technical Guides
1. **[`01_project_overview_and_concepts.md`](01_project_overview_and_concepts.md)**  
   *Problem statement, legal NLP challenges, system architecture, and summary of foundational Experiments 1–5.*

2. **[`02_mandatory_preprocessing_explained.md`](02_mandatory_preprocessing_explained.md)**  
   *The 5 mandatory preprocessing tasks explained step-by-step with code examples: Sentence boundary, Tokenization, POS Tagging, Lemmatization, and WordNet ontology.*

3. **[`03_dataset_and_task_formulation.md`](03_dataset_and_task_formulation.md)**  
   *Dataset provenance (82 clauses across 11 platforms), stratified 70/15/15 splits, and why sequence classification was selected over summarization.*

4. **[`04_llm_finetuning_deep_dive.md`](04_llm_finetuning_deep_dive.md)**  
   *Transformer fine-tuning mechanics: BERT-Tiny architecture (4.39M parameters), AdamW optimizer, Weighted Cross-Entropy Loss, and best-checkpoint selection.*

5. **[`05_evaluation_metrics_and_results.md`](05_evaluation_metrics_and_results.md)**  
   *Quantitative evaluation: Classical baseline (TF-IDF + Logistic Regression) vs. BERT-Tiny (+19.6% precision gain), 100% high-risk test recall, and confusion matrix interpretation.*

6. **[`06_how_to_run_and_viva_qa.md`](06_how_to_run_and_viva_qa.md)**  
   *Environment execution commands, live CLI demonstration walkthrough, and the top 15 general viva questions & answers.*

---

### 3. Formal Academic Reports
- **[`AUDIT_REPORT.md`](AUDIT_REPORT.md)**: Official evidence-based verification report validating all 9 unit tests, dataset schemas, and model weights.

---

## ⚡ Quick Reference Commands

| Operation | Script | Command |
|---|---|---|
| **Mandatory Preprocessing (5 steps)** | `preprocessing_pipeline.py` | `python preprocessing_pipeline.py` |
| **Interactive Live Demo** | `demo.py` | `python demo.py` |
| **Model Metrics & Loss Table** | `show_results.py` | `python show_results.py` |
| **Automated Unit Tests** | `test_pipeline.py` | `python -m unittest test_pipeline.py -v` |
| **Fine-Tuning & Evaluation Engine** | `fine_tune_bert.py` | `python fine_tune_bert.py` |
| **Single Clause Inference** | `inference.py` | `python inference.py` |
