# Official Microsoft Form Submission Draft

**Form:** NLP Mini-Project Guidelines – October 2026  
**User:** DISHA  
**Project Title:** Automated Terms & Conditions Summarizer and Clause Risk Analyzer  

---

### 1. Select Group Number
**Answer:** `35`

---

### 2. Select the NLP Task
* [x] **Text Classification / Sentiment Analysis**  
*(Note: Our task is Domain-Specific Legal Clause Risk Classification into Low, Medium, and High risk).*

---

### 3. Group Members' Names and Roll Numbers
**Answer:**
```text
1. Disha Takawale - 2413195
2. Prerana Patankar - 2413196
3. Saloni Patil - 2413210
4. Neha Maniyar - 2413211
```

---

### 4. The primary purpose of WordNet analysis in LLM fine-tuning is to:
* [ ] A) Tokenize sentences
* [x] **B) Capture semantic relationships among words**
* [ ] C) Measure model accuracy
* [ ] D) Reduce words to base form

---

### 5. During fine-tuning of an LLM for a specific NLP task, what is typically modified?
* [ ] A) Model architecture layers
* [x] **B) Pre-trained model weights using domain-specific data**
* [ ] C) Tokenization algorithm
* [ ] D) WordNet database

---

### 6. Why is a domain-specific dataset used during fine-tuning?
* [ ] A) To train a model from scratch
* [x] **B) To adapt the pre-trained model to the target domain and improve task relevance**
* [ ] C) To reduce model parameters
* [ ] D) To remove the need for evaluation metrics

---

### 7. After fine-tuning an LLM, which step is essential to verify performance improvement?
* [ ] A) Expanding the training dataset
* [x] **B) Evaluating the model using metrics like BERTScore or F1-score**
* [ ] C) Retokenizing the dataset
* [ ] D) Changing the optimizer

---

### 8. Which Libraries and APIs Are Used?
Select the following checkboxes:
* [x] **Hugging Face Transformers**
* [x] **PyTorch**
* [x] **NLTK**
* [x] **WordNet**
* [x] **scikit-learn**
* [x] **spaCy** *(used in Experiments 1, 4, 5)*

---

### 9. Specify the Fine-Tuning Configuration. Pretrained model, dataset, learning rate, epochs, batch size and training environment.
**Answer:**
```text
- Pretrained Model: prajjwal1/bert-tiny (BertForSequenceClassification, 2 transformer layers, 128 hidden dim, 2 attention heads, ~4.39M parameters)
- Dataset: tc_clauses_expanded.csv (82 domain-specific contractual clauses across 11 platforms; stratified split: 70% Train [57 clauses], 15% Validation [12 clauses], 15% Held-Out Test [13 clauses])
- Optimizer: AdamW (weight decay = 0.01)
- Learning Rate: 1e-4 (0.0001)
- Epochs: 12 epochs (training loss dropped monotonically from 1.1194 to 0.4742; validation accuracy reached 75.00%)
- Batch Size: 4
- Loss Function: Weighted Cross-Entropy Loss (weights: Low = 1.462, Medium = 1.188, High = 0.679 to handle class imbalance)
- Max Sequence Length: 128 tokens (WordPiece tokenization with padding and truncation)
- Training Environment: Local CPU execution environment running Python 3.14, PyTorch 2.14.1, and Transformers 5.19.0
```

---

### 10. Specify Each Group Member's Contribution and Demonstration Responsibility.
**Answer:**
```text
1. Disha Takawale (2413195): Curated and expanded the legal dataset (tc_clauses_expanded.csv), implemented the 5 mandatory preprocessing requirements (sentence boundary, tokenization, POS tagging, lemmatization, and WordNet semantic analysis), and built the PyTorch DataLoader pipeline. Responsible for demonstrating the preprocessing pipeline and dataset curation.

2. Prerana Patankar (2413196): Selected the pretrained BERT-Tiny model architecture, designed the fine-tuning training loop with AdamW optimization, configured weighted cross-entropy loss, and tracked loss convergence over 12 epochs. Responsible for demonstrating model fine-tuning and parameter weight update verification.

3. Saloni Patil (2413210): Implemented the classical TF-IDF Logistic Regression baseline model, evaluated the fine-tuned model on the isolated held-out test split, generated the confusion matrix, and analyzed precision/recall/F1 metrics. Responsible for demonstrating evaluation metrics, baseline benchmarking, and safety recall analysis.

4. Neha Maniyar (2413211): Developed the interactive CLI demo (demo.py) and standalone inference pipeline (inference.py), generated the pipeline flowchart (fine_tuning_flowchart.png), and prepared technical documentation and presentation slides. Responsible for demonstrating live clause inference and the architecture flowchart.
```

---

### 11. Enter Title
**Answer:**
```text
Automated Terms & Conditions Summarizer and Clause Risk Analyzer
(Descriptive Academic Scope: NLP-Based Terms & Conditions Clause Risk Analyzer Using BERT-Tiny)
```

---

### 12. Sample Input
**Answer:**
```text
Amazon reserves the right in its sole discretion to terminate accounts, terminate your rights to use Amazon Services, or cancel orders without notice.
```

---

### 13. Sample Output
**Answer:**
```text
Predicted Risk Level: HIGH (Confidence: 73.46%)
Assessment: WARNING! Unilateral account termination and restriction of consumer rights.
Probability Distribution: Low = 9.7% | Medium = 16.9% | High = 73.5%
```

---

### 14. State Preprocessing Technique Used
**Answer:**
```text
1. Sentence Boundary Identification (nltk.sent_tokenize)
2. Tokenization (nltk.word_tokenize and WordPiece BertTokenizer)
3. Part-of-Speech (POS) Tagging (NLTK Perceptron Tagger and scratch HMM Viterbi tagger isolating modal auxiliaries)
4. POS-Informed Lemmatization (WordNetLemmatizer mapping legal morphology to valid base lemmas)
5. WordNet Semantic Analysis (Hypernym, hyponym, and synset extraction for contractual terms like 'liability' and 'terminate')
6. HTML stripping, contraction expansion, number/currency normalization, and GDPR PII anonymization (from foundational Exps 3-4)
```

---

### 15. State LLM Model used in project
**Answer:**
```text
prajjwal1/bert-tiny (Pretrained BERT Sequence Classification Transformer, ~4.39M parameters)
```

---

### 16. Upload Report/PPt
**File to Upload:**
- Option A: `FINAL_MINI_PROJECT_REPORT.md` (or export to PDF)
- Option B: `PRESENTATION_SLIDES.md` (or export to PPT/PDF)

---

### 17. Upload Your Code (python file)
**File to Upload:**
- Primary Code File: `fine_tune_bert.py`  
*(Or zip containing `fine_tune_bert.py`, `preprocessing_pipeline.py`, and `demo.py`)*

---

### 18. Upload your Dataset (CSV file)
**File to Upload:**
- File: `tc_clauses_expanded.csv` *(82 rows across 11 platforms, Size: ~17 KB)*  
*(Or original seed file: `tc_clauses.csv`)*
