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
*Creates `tc_clauses_expanded.csv` (82 rows) while keeping the original `tc_clauses.csv` untouched.*

### 3. Run Fine-Tuning and Evaluation
```bash
python fine_tune_bert.py
```
*Trains BERT-Tiny for 12 epochs using AdamW, prints loss at each epoch, evaluates on the test set, compares against the TF-IDF baseline, and saves model files to `model_output/`.*

### 4. Run Instant Clause Inference
```bash
python inference.py
```
*Loads the saved model and classifies new contract sentences into Low, Medium, and High risk with confidence scores.*

### 5. Generate the Flowchart Diagram
```bash
python generate_flowchart.py
```
*Renders `fine_tuning_flowchart.png`.*

---

## 2. Top 15 Viva Questions & Answers for Team Members

### Q1: What is the problem statement of your mini-project?
**Answer:** Most people blindly click "I Agree" on Terms and Conditions because they are too long and full of confusing legal language ("legalese"). Without reading them, users unknowingly give up their right to sue, agree to account deletion without warning, and accept clauses that heavily favor the company. Our project builds an NLP system that reads, cleans, and automatically classifies each legal clause into Low, Medium, or High risk — using a fine-tuned Transformer model — so users can quickly see what they're really agreeing to.

---

### Q2: What were Experiments 1 through 5 about?
**Answer:**
- **Exp 1:** Compared NLP tools — NLTK, spaCy, Hugging Face, Google Cloud NLP, AWS Comprehend.
- **Exp 2:** Defined our problem and reviewed existing legal NLP work (like the ToS;DR dataset — a public project rating T&C fairness).
- **Exp 3:** Basic text cleaning — removing HTML, expanding contractions, handling numbers/currencies, keeping negation words like "not".
- **Exp 4:** Advanced cleaning — Lemmatization vs Stemming, hiding personal data (GDPR PII anonymization), language detection.
- **Exp 5:** Shallow parsing (finding Noun/Verb phrases), deep parsing (Subject-Verb-Object triples), and building CFG (Context-Free Grammar) parse trees.
- *(Plus our HMM (Hidden Markov Model) implementation for POS tagging using the Viterbi algorithm — a method to find the most likely sequence of tags.)*

---

### Q3: What are the 5 mandatory preprocessing requirements, and how did you implement them?
**Answer:**
1. **Sentence Boundary Identification:** Used NLTK `sent_tokenize` — a smart full-stop detector that handles abbreviations like "LLC." correctly.
2. **Tokenization:** Used `nltk.word_tokenize` for word-level tokens and `BertTokenizer` for BERT's subword tokens.
3. **POS Tagging:** Used NLTK Perceptron Tagger and our from-scratch HMM tagger.
4. **Lemmatization:** Used POS-informed WordNet Lemmatizer to reduce words to their real dictionary base (`terminating` $\rightarrow$ `terminate`).
5. **WordNet Semantic Analysis:** Used NLTK `wordnet` synsets to find hypernyms (broader/parent terms), hyponyms (narrower/child terms), and definitions for legal words like "liability" and "terminate".

---

### Q4: Why is WordNet analysis useful in legal NLP?
**Answer:** WordNet understands meaning, not just spelling. It connects related words — for example, "liability" is connected to "indebtedness" and "legal obligation". This helps the system understand legal clauses that use different words to describe the same dangerous concept — instead of just matching exact keywords.

---

### Q5: What is the difference between Stemming and Lemmatization?
**Answer:**
- **Stemming** roughly chops off word endings using simple rules — and often produces non-real words. For example, the Porter Stemmer turns *indemnification* into `indemnif`, which doesn't exist in any dictionary.
- **Lemmatization** uses a vocabulary and grammar understanding to return real dictionary words. For example, *obligations* becomes `obligation` and *terminating* becomes `terminate`.
- Analogy: Stemming is like cutting a plant stem with scissors — you might cut too much. Lemmatization is like pruning it properly — you know exactly where to stop.
- For legal text, lemmatization is essential because the words must remain meaningful.

---

### Q6: Why did you choose Legal Clause Risk Classification instead of Summarization for fine-tuning?
**Answer:** Training a summarization model (like T5 or BART) requires thousands of pairs of `(Legal Clause, Human-Written Summary)`. Our dataset has real legal clauses with verified risk labels (`Low`, `Medium`, `High`), but no human summaries. Making up summaries would be dishonest and unscientific. Clause Risk Classification uses our real data, has clear correct answers, and is the most honest engineering choice.

---

### Q7: What model did you fine-tune, and what is its architecture?
**Answer:** We fine-tuned `prajjwal1/bert-tiny` — a tiny version of BERT (Bidirectional Encoder Representations from Transformers, made by Google). It is set up as `BertForSequenceClassification` with:
- 2 Transformer layers
- 128 hidden embedding dimensions (each word is a 128-number vector)
- 2 attention heads (the model looks at the sentence from 2 angles at once)
- ~4.39 million trainable parameters (numbers that get updated during training)

---

### Q8: Why use BERT-Tiny instead of a massive 7-Billion parameter LLM?
**Answer:**
1. A tiny dataset (82 clauses) would cause a huge model to **overfit** — meaning it would just memorize the training sentences like a student cramming without understanding, and fail on new examples.
2. BERT-Tiny runs in seconds on a standard CPU. No GPU needed. Affordable and practical.
3. BERT-Tiny uses the same core Transformer mechanics (Multi-Head Self-Attention, Layer Normalization, Feed-Forward layers) as big models — just scaled down.

---

### Q9: What actually changes during fine-tuning?
**Answer:** The internal numerical weights of the neural network change. These include the Query, Key, and Value weight matrices inside the self-attention layers (which control how the model focuses on different words), the feed-forward layer weights, and the final linear classification head. All of these are updated through **gradient backpropagation** — the model measures its mistake (loss), and then adjusts each weight in the direction that reduces that mistake.

---

### Q10: What optimizer and loss function did you use?
**Answer:**
- **Optimizer:** `AdamW` — Adam with Decoupled Weight Decay. It's the standard optimizer for transformer models. It adjusts step sizes smartly for each parameter and prevents overfitting via L2 regularization (a penalty on large weights). Learning rate = $1 \times 10^{-4}$, weight decay = $0.01$.
- **Loss Function:** **Weighted Cross-Entropy Loss** — rarer classes (Low, Medium) get a higher penalty when misclassified, so the model doesn't just always predict the most common class (High).
  - Weights: $w_{\text{Low}} = 1.462, w_{\text{Med}} = 1.188, w_{\text{High}} = 0.679$.

---

### Q11: How do you prove that the model actually learned during training?
**Answer:**
1. Training loss dropped from **1.1194** to **0.4742** — a 57.6% reduction. The model's mistakes shrank with every epoch.
2. Validation accuracy rose from **41.67%** to **75.00%**. The model got much better at predicting on data it hadn't trained on.
3. The best model checkpoint (lowest validation loss at Epoch 12) was saved to `model_output/model.safetensors` — proof that real parameters were updated and stored.

---

### Q12: What baseline model did you use, and how did BERT compare?
**Answer:** We trained a classical **TF-IDF + Logistic Regression** baseline on the exact same training data for a fair comparison.
- Both models achieved **61.54% overall accuracy** on the held-out test set.
- But **Fine-Tuned BERT got a Macro Precision of 0.8485**, while the baseline only got **0.6528** — an absolute gain of **+19.6%**.
- BERT also achieved **1.00 (100%) precision** on Low and Medium risk classes — whenever it predicted Low or Medium, it was always correct.

---

### Q13: What was the recall on High-Risk clauses, and why is it important?
**Answer:** The fine-tuned BERT model got **1.00 (100%) Recall on High-Risk clauses** — it correctly found all 6 High-Risk clauses in the 13-example test set, missing none.
- This matters because in consumer protection AI, missing a dangerous clause (a False Negative) is the worst kind of mistake — it could leave a user unaware of a serious risk.
- Important caveat: With only 13 test samples, this shows strong sensitivity on our academic benchmark. We'd need more data to guarantee the same in real-world contracts.

---

### Q14: What datasets did you use?
**Answer:**
- Reference Terms of Service raw text from **Amazon** (`amazon.txt`) and **Alibaba** (`alibaba.txt`).
- The seed dataset `tc_clauses.csv` — 17 rows (original, kept untouched).
- The expanded dataset `tc_clauses_expanded.csv` — 82 clauses from 11 platforms, manually curated by our team (not scraped automatically, so labels are clean and verified).

---

### Q15: What are the limitations and future scope of this project?
**Answer:**
- **Task vs Title:** The project is registered as *"Automated Terms & Conditions Summarizer and Clause Risk Analyzer"*, but the actual model is a 3-class risk classifier — not a full summarizer. Full abstractive summarization is listed as future work.
- **Limitations:**
  - Small dataset — only 82 clauses and 13 test samples.
  - Splitting was done at clause level, not by company — so we haven't fully tested cross-company generalization.
- **Future Scope:**
  - Scale up using public legal datasets like LexGLUE or ToS;DR on GPU hardware.
  - Add abstractive summarization using encoder-decoder or instruction-tuned models.
  - Build an interactive browser extension that flags risky clauses automatically when a user visits a signup page.
