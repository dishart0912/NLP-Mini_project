# 6. How to Run and Viva Q&A Guide

This guide tells you exactly how to run each piece of code, plus the **Top 15 Viva Questions and Answers** you and your teammates need to know.

---

## 1. How to Run Everything on Your Machine

Open your terminal / command prompt in the project directory (`e:\tsec\sem7\lab\nlp`):

### 1. Run Mandatory Preprocessing Verification
```bash
python preprocessing_pipeline.py
```
*Prints sentence boundaries, tokens, POS tags, lemmatized forms, and WordNet hypernyms/hyponyms.*

### 2. Generate the Expanded Dataset
```bash
python prepare_expanded_dataset.py
```
*Curates `tc_clauses_expanded.csv` (82 rows) while preserving `tc_clauses.csv`.*

### 3. Run Fine-Tuning and Evaluation
```bash
python fine_tune_bert.py
```
*Trains BERT-Tiny for 12 epochs using AdamW, logs loss convergence, evaluates on test data, compares against the TF-IDF baseline, and saves model checkpoints to `model_output/`.*

### 4. Run Instant Clause Inference
```bash
python inference.py
```
*Loads the saved model and classifies unseen contract sentences into Low, Medium, and High risk with confidence scores.*

### 5. Generate the Flowchart Diagram
```bash
python generate_flowchart.py
```
*Renders `fine_tuning_flowchart.png`.*

---

## 2. Top 15 Viva Questions & Answers for Team Members

### Q1: What is the problem statement of your mini-project?
**Answer:** Digital users routinely accept complex, multi-page Terms and Conditions without reading them due to verbose legal jargon ("legalese"), unknowingly waiving dispute rights and agreeing to unilateral account termination. Our project builds an NLP system that preprocesses legal agreements, extracts obligations, and automatically classifies clauses into Low, Medium, and High risk categories using a fine-tuned Transformer model.

---

### Q2: What were Experiments 1 through 5 about?
**Answer:**
- **Exp 1:** Comparative study of NLP tools (NLTK, spaCy, Hugging Face, Google Cloud NLP, AWS Comprehend).
- **Exp 2:** Problem formulation and legal literature review (ToS;DR dataset).
- **Exp 3:** Basic text preprocessing (HTML removal, contraction expansion, number/currency tokenization, negation preservation).
- **Exp 4:** Advanced preprocessing (Lemmatization vs Stemming, GDPR PII anonymization, language detection).
- **Exp 5:** Shallow parsing (NP/VP chunking) and Deep dependency parsing (SVO triples) + CFG trees.
- *(Plus our HMM implementation for Viterbi POS-tagging).*

---

### Q3: What are the 5 mandatory preprocessing requirements, and how did you implement them?
**Answer:**
1. **Sentence Boundary Identification:** Used NLTK `sent_tokenize` to isolate discrete contractual obligations.
2. **Tokenization:** Used `nltk.word_tokenize` and subword `BertTokenizer`.
3. **POS Tagging:** Used NLTK Perceptron Tagger and our scratch HMM tagger.
4. **Lemmatization:** Used POS-informed WordNet Lemmatizer to reduce words to valid dictionary bases (`terminating` $\rightarrow$ `terminate`).
5. **WordNet Semantic Analysis:** Used NLTK `wordnet` synsets to extract hypernyms, hyponyms, and definitions for legal terms like "liability" and "terminate".

---

### Q4: Why is WordNet analysis useful in legal NLP?
**Answer:** WordNet captures semantic ontological relationships (synonyms, hypernyms/broader terms, hyponyms/narrower terms) rather than just exact word strings. For example, it connects "liability" to "indebtedness" and "obligation", enabling the system to understand legal accountability.

---

### Q5: What is the difference between Stemming and Lemmatization?
**Answer:**
- **Stemming** chops off word suffixes heuristically and often produces non-words (e.g., Porter Stemmer turns *indemnification* into `indemnif`).
- **Lemmatization** uses morphological analysis and dictionary vocabulary to return valid base words (e.g., *indemnification* remains `indemnification`, and *obligations* becomes `obligation`). For legal text, lemmatization is essential to maintain dictionary validity.

---

### Q6: Why did you choose Legal Clause Risk Classification instead of Summarization for fine-tuning?
**Answer:** Fine-tuning an abstractive summarization model (like T5) requires thousands of pairs of legal clauses and gold-standard human-written summaries. Our actual dataset contains authentic clauses with verified risk labels (`Low`, `Medium`, `High`), but no human summary targets. Selecting risk classification is the scientifically sound and defensible choice because it relies on genuine ground-truth data.

---

### Q7: What model did you fine-tune, and what is its architecture?
**Answer:** We fine-tuned `prajjwal1/bert-tiny` (`BertForSequenceClassification`), an efficient 2-layer Bidirectional Encoder Representations from Transformers (BERT) model with 128 hidden embedding dimensions, 2 attention heads, and ~4.39 million trainable parameters.

---

### Q8: Why use BERT-Tiny instead of a massive 7-Billion parameter LLM?
**Answer:**
1. A small domain dataset (82 clauses) would cause massive models to overfit severely (memorizing training text rather than learning general patterns).
2. BERT-Tiny uses the exact same self-attention transformer mechanics as larger models while being trainable and runnable in seconds on standard CPU hardware without GPU costs.

---

### Q9: What actually changes during fine-tuning?
**Answer:** During fine-tuning, the internal neural network weights (the Query, Key, Value attention weight matrices, feed-forward layers, and the linear classification head) are updated through gradient backpropagation using task-specific loss gradients, rather than remaining static.

---

### Q10: What optimizer and loss function did you use?
**Answer:**
- **Optimizer:** `AdamW` (Adam with Decoupled Weight Decay) with a learning rate of $1 \times 10^{-4}$ and weight decay of $0.01$.
- **Loss Function:** **Weighted Cross-Entropy Loss**, with higher loss penalties for under-represented classes ($w_{\text{Low}} = 1.462, w_{\text{Med}} = 1.188, w_{\text{High}} = 0.679$) to overcome class imbalance.

---

### Q11: How do you prove that the model actually learned during training?
**Answer:**
1. The training loss decreased monotonically from **1.1194** down to **0.4742** (a 57.6% reduction).
2. The validation accuracy increased from **41.67%** up to **75.00%**.
3. The resulting model weights were saved to disk in `model_output/model.safetensors`.

---

### Q12: What baseline model did you use, and how did BERT compare?
**Answer:** We trained a classical **TF-IDF + Logistic Regression** baseline on the exact same training split.
While both models achieved 61.54% overall accuracy on the held-out test split, **Fine-Tuned BERT achieved a Macro Precision of 0.8485 compared to the baseline's 0.6528 (an absolute gain of +19.6%)**, and achieved **1.00 (100%) precision** on Low and Medium risk classes.

---

### Q13: What was the recall on High-Risk clauses, and why is it important?
**Answer:** The fine-tuned BERT model achieved **1.00 (100%) Recall on High-Risk clauses** (catching all 6 out of 6 in the test set). In consumer legal AI, missing a hazardous clause (a False Negative) could cause severe harm to the user. Achieving 100% High-Risk recall ensures no dangerous clause goes unflagged.

---

### Q14: What datasets did you use?
**Answer:**
- Real-world Terms of Service from **Amazon** (`amazon.txt`) and **Alibaba** (`alibaba.txt`).
- The seed dataset `tc_clauses.csv` (17 rows).
- The expanded dataset `tc_clauses_expanded.csv` (82 clauses across 11 platforms including Google, Apple, Microsoft, Meta, Netflix, Twitter/X, Spotify, Uber, and Zoom).

---

### Q15: What are the limitations and future scope of this project?
**Answer:**
- **Limitations:** The dataset size is modest (82 clauses) due to manual legal annotation requirements, and CPU execution limits training large multi-billion parameter models.
- **Future Scope:** Scaling up the corpus to 10,000+ clauses using the LexGLUE / ToS;DR benchmarks on GPU hardware, implementing a browser extension for live contract checking, and combining Retrieval-Augmented Generation (RAG) for interactive contract question-answering.
