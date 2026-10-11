# Team Learning & Project Documentation Index

Welcome! This folder contains beginner-friendly, structured, and in-depth documentation covering everything implemented in our NLP Mini-Project:

**Project Title:** Automated Terms & Conditions Summarizer and Clause Risk Analyzer  
**Group Members:**
- Disha Takawale (2413195)
- Prerana Patankar (2413196)
- Saloni Patil (2413210)
- Neha Maniyar (2413211)

---

## Documentation Roadmap

We have divided the documentation into 6 bite-sized, structured guides so that every team member can understand the concepts, code, and viva questions clearly:

1. [01_project_overview_and_concepts.md](01_project_overview_and_concepts.md)  
   *What is this project about? Why legal NLP? What was already done in Experiments 1–5, and what was added now?*

2. [02_mandatory_preprocessing_explained.md](02_mandatory_preprocessing_explained.md)  
   *The 5 mandatory preprocessing tasks explained step-by-step with examples: Sentences, Tokens, POS Tagging, Lemmatization, and WordNet ontology.*

3. [03_dataset_and_task_formulation.md](03_dataset_and_task_formulation.md)  
   *Where did the data come from? Why did we choose Clause Risk Classification instead of Summarization? How was the dataset curated and split?*

4. [04_llm_finetuning_deep_dive.md](04_llm_finetuning_deep_dive.md)  
   *How does fine-tuning actually work? What is BERT-Tiny? What are weights, loss functions, learning rates, epochs, and AdamW?*

5. [05_evaluation_metrics_and_results.md](05_evaluation_metrics_and_results.md)  
   *How did our model perform? What do Accuracy, Precision, Recall, Macro F1, and the Confusion Matrix mean? Why is our baseline comparison important?*

6. [06_how_to_run_and_viva_qa.md](06_how_to_run_and_viva_qa.md)  
   *How to run the scripts on your computer, plus the top 15 viva questions your professor or external examiner might ask (with exact answers!)*

---

## Quick Reference Summary

| Script / Asset | Purpose | Command |
|---|---|---|
| `preprocessing_pipeline.py` | Runs the 5 mandatory preprocessing tasks | `python preprocessing_pipeline.py` |
| `prepare_expanded_dataset.py` | Builds the 82-clause dataset | `python prepare_expanded_dataset.py` |
| `fine_tune_bert.py` | Trains BERT, benchmarks baseline, evaluates | `python fine_tune_bert.py` |
| `inference.py` | Predicts risk on new contract sentences | `python inference.py` |
| `generate_flowchart.py` | Recreates the pipeline diagram | `python generate_flowchart.py` |
