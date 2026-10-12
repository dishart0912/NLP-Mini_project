# Deep-Dive Contribution & Viva Defense Guide

**Project Title:** NLP-Based Terms & Conditions Clause Risk Analyzer Using BERT-Tiny  
*(Registered Topic: Automated Terms & Conditions Summarizer and Clause Risk Analyzer)*  
**BE Computer Engineering — Semester VII (Academic Year 2026)**  
**Course:** Natural Language Processing Lab  
**Group Number:** 35  

---

## 🎯 Purpose of This Document

This is **NOT a slide presentation script**. This is a **detailed technical document to help you defend your project**.

Examiners often skip the slides and ask sharp technical questions about:
- Why you chose specific algorithms, libraries, or formulas.
- The exact math, parameters, and loss functions used.
- Why you rejected other methods (like Porter stemmer, BERT-Base, or T5).
- How the code works, line by line.

This document gives each of the 4 team members a clear, deep breakdown of **their own area of work** so they can answer any question with confidence.

---

## 🧭 Member Assignment Overview

| Member Name | Roll Number | Technical Domain & Implementation Area | Primary Code & Artifacts |
|:---|:---:|:---|:---|
| **Disha Takawale** | 2413195 | **Linguistic Preprocessing, Syntax & WordNet Ontology** | `preprocessing_pipeline.py`, `prepare_expanded_dataset.py`, `tc_clauses.csv` |
| **Prerana Patankar** | 2413196 | **Transformer Architecture, PyTorch Backprop & Optimization** | `fine_tune_bert.py`, `model_output/model.safetensors`, AdamW loop |
| **Saloni Patil** | 2413210 | **Statistical Benchmarking, Baseline & Evaluation Metrics** | `fine_tune_bert.py` (Steps 1 & 3), `show_results.py`, Confusion Matrix |
| **Neha Maniyar** | 2413211 | **Inference Architecture, Extractive Risk Triage & Productization** | `inference.py`, `demo.py`, `fine_tuning_flowchart.png`, `docs/08_*` |

---

# 📚 SECTION 1: DISHA TAKAWALE (2413195)
### Domain: Linguistic Preprocessing, Morphological Normalization, Syntactic Tagging & WordNet Ontology

---

### 1.1 Core Technical Responsibilities
1. Built and checked the starting dataset (`tc_clauses.csv`) and wrote the script to expand it (`prepare_expanded_dataset.py`).
2. Coded the 5 required NLP lab preprocessing steps in `preprocessing_pipeline.py`.
3. Wrote the function (`get_wordnet_pos`) that converts POS tag formats so lemmatization works correctly.
4. Explored the WordNet word database to find related legal terms (broader and narrower word meanings).

---

### 1.2 Deep Technical Concepts Disha Must Know

#### A. Sentence Boundary Identification:
- **Library/Function:** `nltk.sent_tokenize()`.
- **How it works inside:** It uses an algorithm called **Punkt** — an unsupervised sentence boundary detector. Think of it like a smart reader who has learned that "Inc." or "e.g." are abbreviations, NOT the end of a sentence. So it won't incorrectly split *"Amazon.com, Inc."* into two sentences.
- **Why it matters in Legal NLP:** Legal contracts are packed with long, dense paragraphs. Splitting them into individual sentences helps us treat each legal duty or right on its own.

#### B. Subword vs. Lexical Tokenization:
- **Lexical Tokenization:** `nltk.word_tokenize()` — this just splits text into regular words and punctuation using pattern-matching rules. Like cutting a sentence at every space and comma.
- **Subword Tokenization (BERT):** BERT uses **WordPiece**. If a legal word is rare (like *"indemnification"*), WordPiece breaks it into smaller known pieces: `['in', '##dem', '##ni', '##fication']`. This way, the model never sees an unknown word — it's like reading an unfamiliar word syllable by syllable.

#### C. Lemmatization vs. Stemming (Very Common Viva Question!):
- **Why Porter/Lancaster Stemming Fails:** Stemmers are blunt — they chop off word endings using fixed rules, without understanding grammar. *"terminating"* becomes *"terminat"*; *"liabilities"* becomes *"liabil"*. These are not real words. In legal text, that distorts meaning.
  > Think of stemming like using scissors to cut a word — you might cut too much.
- **Why WordNet Lemmatization Succeeds:** It looks up the real dictionary base form using `WordNetLemmatizer.lemmatize()`.
  > Lemmatization is like looking up the word in a dictionary to find its root — it's smarter and more accurate.
- **The Crucial POS Mapping Trap:** If you pass *"terminating"* to a lemmatizer without telling it the word is a **verb**, it assumes it's a **noun** and does nothing! Disha's code maps detailed Treebank tags (`VBG`, `VBD`, `VBZ`) to `wordnet.VERB`, so the lemmatizer correctly turns *"terminating"* into *"terminate"*.

#### D. Syntactic Analysis (POS Tagging):
- **Implementation:** `nltk.pos_tag()` (Averaged Perceptron Tagger) + a Custom HMM Viterbi Tagger from Experiment 5.
- **Why POS Tagging matters in law:**
  - Finds **Modal Auxiliaries (`MD`)**: Words like *shall*, *must*, *will* mean the rule is **compulsory**. Words like *may* or *should* mean it's **optional**. This distinction is huge in legal text.
  - Finds **Root Verbs (`VB`)**: Core actions like *terminate*, *indemnify*, *arbitrate* — these are the power words in a contract.

#### E. WordNet Semantic Ontology:
- **What WordNet Is:** A big English word database. Words are grouped into meaning-sets called **Synsets** — like a dictionary + thesaurus combined.
- **Hypernyms vs. Hyponyms:**
  - **Hypernym:** A broader/parent concept (*Y is a hypernym of X if X is a type of Y*). Example: *"indebtedness"* is a hypernym of *"liability"* — liability is a type of indebtedness.
  - **Hyponym:** A narrower/child concept (*X is a hyponym of Y*). Example: *"abort", "break", "finalize"* are all hyponyms of *"terminate"* — they are specific types of termination.
- **Domain Limitation to Acknowledge:** WordNet covers general English. It doesn't know old Latin legal phrases like *"force majeure"* or *"contra proferentem"*. So we focused on common operational verbs and liability nouns.

---

### 1.3 Tough Viva Cross-Questions & Exact Answers for Disha

> **Examiner Q1: Does your fine-tuned BERT model use the lemmatized words produced by NLTK?**  
> **Disha's Answer:** *"No, Sir — and that's a deliberate design decision. BERT is a pretrained model that works directly on raw text using its own WordPiece tokenizer. If we fed it lemmatized words (like turning 'terminating' into 'terminate' and stripping endings), we would destroy the tense and grammar structure that BERT relies on to understand context. Our NLTK pipeline is a separate linguistic analysis module — it runs independently and is not fed into BERT."*

> **Examiner Q2: How did you prepare the expanded dataset, and where did the extra clauses come from?**  
> **Disha's Answer:** *"The original seed file `tc_clauses.csv` had only 17 clauses — way too few for a proper train-validation-test split. We wrote `prepare_expanded_dataset.py`, which kept all 17 original clauses untouched and added 65 new clauses. These 65 were manually written and labelled by our team, based on real Terms of Service from 11 major platforms like Amazon, Google, Apple, Meta, and Microsoft. The reference texts were `amazon.txt` and `alibaba.txt`. The final dataset has 82 clauses: 40 High, 23 Medium, and 19 Low risk."*

> **Examiner Q3: How does your Penn Treebank to WordNet POS mapping work?**  
> **Disha's Answer:** *"Penn Treebank uses 36 fine-grained tags like `VBD`, `VBG`, `NN`, `NNS`, `JJ`. But WordNet only understands 4 broad categories: NOUN, VERB, ADJ, and ADV. My function `get_wordnet_pos()` in `preprocessing_pipeline.py` just checks the first letter of the Treebank tag: 'V' maps to `wn.VERB`, 'N' maps to `wn.NOUN`, 'J' maps to `wn.ADJ`, and 'R' maps to `wn.ADV`. Without this mapping, verbs get treated as nouns by the lemmatizer and don't get reduced properly."*

---

# 📚 SECTION 2: PRERANA PATANKAR (2413196)
### Domain: Pretrained Transformer Architecture, PyTorch Backpropagation, Optimization & Checkpoint Selection

---

### 2.1 Core Technical Responsibilities
1. Chose the transformer model (`prajjwal1/bert-tiny`) and explained why it fits our project size.
2. Built the PyTorch `Dataset` and `DataLoader` — the system that feeds text into the model in batches.
3. Wrote the training loop that runs for multiple epochs and updates model weights using backpropagation.
4. Set up the **Weighted Cross-Entropy Loss** function to handle the fact that some risk classes have fewer examples.
5. Added logic to save the **best model checkpoint** (the version of the model that performs best on validation data).

---

### 2.2 Deep Technical Concepts Prerana Must Know

#### A. BERT Architecture & Self-Attention:
- **Architecture of `prajjwal1/bert-tiny`:**
  - **Transformer Layers (Encoder blocks):** 2
  - **Hidden Embedding Dimension ($d_{model}$):** 128 — each word becomes a list of 128 numbers.
  - **Self-Attention Heads ($h$):** 2 (each head handles a dimension of $128 / 2 = 64$)
  - **Intermediate Feedforward Dimension:** 512
  - **Total Parameters:** **4,386,307** (~4.39 Million) — these are the learnable numbers in the model.
- **Bidirectional Self-Attention Formula:**
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  > **Plain English:** Every word looks at every other word in the sentence (both before and after it) and decides how much attention to pay to each. Q = what am I looking for? K = what do I offer? V = what information do I pass along? The $\sqrt{d_k}$ stops the numbers from getting too large before the softmax.

#### B. Why BERT-Tiny Instead of BERT-Base or a 7B LLM?
- **Parameter-to-Data Ratio:** Our dataset has 82 clauses. `bert-base-uncased` has 110 million parameters. Training 110 million parameters on just 82 examples is like trying to teach a stadium-full of teachers using a single textbook — the model just memorizes everything and learns nothing useful (this is called **overfitting**).
- **Execution Feasibility:** `bert-tiny` trains cleanly on a regular CPU in about 15 seconds. No expensive GPU needed.

#### C. The AdamW Optimizer (vs. Standard SGD / Adam):
- **Hyperparameters:** $\text{Learning Rate} = 1 \times 10^{-4}$ (0.0001), $\text{Weight Decay} = 0.01$.
- **Why AdamW:** Standard Adam optimizer adds the weight penalty (L2 regularization) directly into the gradient calculation, which confuses the momentum tracking. AdamW separates the weight decay step cleanly:
  $$\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$
  > **Plain English:** After each update, AdamW shrinks every weight slightly ($\lambda \theta_t$) to prevent any single weight from growing too large. Think of it like a gentle reset to stop the model from becoming overconfident.

#### D. Weighted Cross-Entropy Loss Formula:
- Standard Cross-Entropy:
  $$\mathcal{L} = -\sum_{c=1}^C y_c \log(\hat{y}_c)$$
  > **Plain English:** For the correct class, take the log of the predicted probability, make it negative, and that's your loss. The worse the prediction, the higher the loss.
- **Weighted Cross-Entropy (Used in our code):**
  $$\mathcal{L}_{weighted} = -\sum_{c=1}^C w_c \cdot y_c \log(\hat{y}_c)$$
  > **Plain English:** Same as above, but each class's loss is multiplied by a weight. Rare classes get a bigger weight, so mistakes on them hurt more.
- **Why Class Weights Were Computed:**
  We have 40 High-risk and only 19 Low-risk clauses in training. Without weights, the model would just predict "High" for everything and still get decent accuracy. The balanced class weights fix this:
  $$w_c = \frac{N_{total}}{C \times N_c}$$
  Resulting weights: $w_{\text{Low}} = 1.462$, $w_{\text{Medium}} = 1.188$, $w_{\text{High}} = 0.679$. Getting a Low-risk clause wrong costs 2.15× more than getting a High-risk clause wrong — because Low is the rare class.

#### E. Best Checkpoint Selection Logic:
- A simple training loop would just save the model at the end of Epoch 12 — but what if Epoch 8 was actually better?
- Prerana's loop watches the `val_loss` after every epoch. If `avg_val_loss < best_val_loss`, it saves a deep copy of the model weights at that point.
- After all 12 epochs finish, the script automatically loads the best saved version (Epoch 12 turned out to be best: Val Loss = 0.6780, Val Acc = 75.00%) before running on the test set.

---

### 2.3 Tough Viva Cross-Questions & Exact Answers for Prerana

> **Examiner Q1: How do you prove your model was actually fine-tuned and you didn't just use pretrained weights?**  
> **Prerana's Answer:** *"Sir, in `fine_tune_bert.py`, we updated all 4,386,307 weights using PyTorch backpropagation. We tracked the loss across all 12 epochs: training loss dropped steadily from **1.1194 down to 0.4742** — that's a 57.6% drop. At the same time, validation accuracy went from 41.67% up to 75.00%. The updated weights were saved to disk as `model_output/model.safetensors` (16.74 MB). You can reload it and run inference independently — that's proof."*

> **Examiner Q2: Why did you pick a learning rate of 1e-4 instead of the usual 1e-3 or 1e-5?**  
> **Prerana's Answer:** *"When training from scratch, people use $\eta = 10^{-3}$, but that's too high for fine-tuning — it would overwrite the pretrained BERT weights and destroy the knowledge already built in (called catastrophic forgetting). On the other hand, $10^{-5}$ is too slow to converge in just 12 epochs on a CPU. A learning rate of $1 \times 10^{-4}$ with AdamW was the sweet spot — loss dropped steadily every epoch without any blowup."*

> **Examiner Q3: What is the input sequence length, and what happens if a clause is 200 words long?**  
> **Prerana's Answer:** *"We set `max_length = 128` tokens. In `LegalClauseDataset`, if a clause is shorter than 128 tokens, we pad it with zero tokens (`[PAD]`) and set their attention mask to 0 so the model ignores them — like filling blank spaces. If a clause is longer than 128 tokens, we cut it off at 128 subwords. For typical legal sentences, 128 WordPiece tokens is enough to cover the full obligation."*

---

# 📚 SECTION 3: SALONI PATIL (2413210)
### Domain: Statistical Evaluation, Baseline Benchmarking, Metric Mathematics & Sensitivity Analysis

---

### 3.1 Core Technical Responsibilities
1. Built and trained a simple classical model (**TF-IDF + Logistic Regression**) on the same 70% training data, to use as a comparison baseline.
2. Tested both models on the same isolated 15% test set (13 clauses).
3. Calculated evaluation metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, and the Confusion Matrix.
4. Explained what the trade-off between precision and recall means for consumer safety.

---

### 3.2 Deep Technical Concepts Saloni Must Know

#### A. Why a Baseline Model is Mandatory in NLP:
- You can't just say "BERT got 61.54% accuracy" — that number means nothing on its own. You need to compare it against a simpler model to show whether deep learning was actually worth it.
- Saloni built a **TF-IDF + Logistic Regression** pipeline as the baseline:
  - **Feature Extraction:** Word unigrams and bigrams (`ngram_range=(1, 2)`), keeping the top 500 features. Think of it as counting which words (and two-word pairs) appear most in each risk category.
  - **Classifier:** `LogisticRegression(class_weight="balanced", random_state=42)` — a standard, well-understood classifier.

#### B. The Complete Evaluation Results Table:
| Metric | Classical Baseline (TF-IDF + LogReg) | Fine-Tuned BERT-Tiny | Difference / Advantage |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | 61.54% (8/13) | **61.54% (8/13)** | Equal |
| **Macro Precision** | 0.6528 | **0.8485** | **+19.6% Absolute Gain** |
| **Macro Recall** | 0.5833 | 0.5278 | -5.55% |
| **Macro F1-Score** | 0.6000 | 0.5353 | -6.47% |
| **High-Risk Recall** | 83.3% (5/6) | **100.0% (6/6)** | **+16.7% Gain** |

#### C. Metric Formulas (Saloni Must Know These by Heart!):
1. **Accuracy:**
   $$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}} = \frac{8}{13} \approx 61.54\%$$
   > **Plain English:** Out of 13 test clauses, 8 were labelled correctly.

2. **Precision (Quality/Cleanliness):**
   $$\text{Precision}_c = \frac{\text{True Positives}_c}{\text{True Positives}_c + \text{False Positives}_c}$$
   > **Plain English:** Of all the clauses the model *said* were class C, how many *actually* were?
   - When BERT says "Low Risk", it's **100% correct** ($1.00$).
   - When BERT says "Medium Risk", it's **100% correct** ($1.00$).
   - Macro Precision takes the unweighted average across all 3 classes:
     $$\text{Macro Precision} = \frac{1.00 + 1.00 + 0.5455}{3} = \mathbf{0.8485}$$

3. **Recall (Safety/Sensitivity):**
   $$\text{Recall}_c = \frac{\text{True Positives}_c}{\text{True Positives}_c + \text{False Negatives}_c}$$
   > **Plain English:** Of all clauses that *actually* belong to class C, how many did the model *catch*?
   - Out of 6 High-Risk test clauses, BERT caught all 6: $\text{Recall} = \frac{6}{6} = \mathbf{1.00}$.

4. **F1-Score:**
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
   > **Plain English:** The harmonic mean (balanced average) of Precision and Recall. Useful when you care equally about both.

#### D. How to Read the Confusion Matrix:
$$\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 3 \\ 0 & 0 & 6 \end{pmatrix}$$
- **Rows = Ground Truth** (Row 0: Low [3 actual], Row 1: Medium [4 actual], Row 2: High [6 actual]).
- **Columns = Model Predictions** (Col 0: Low, Col 1: Med, Col 2: High).
- **Row 0 (Actual Low):** 1 correctly predicted Low, 0 predicted Med, 2 wrongly predicted High.
- **Row 1 (Actual Med):** 0 predicted Low, 1 correctly predicted Med, 3 wrongly predicted High.
- **Row 2 (Actual High):** 0 predicted Low, 0 predicted Med, **all 6 correctly predicted High!**

> Think of the confusion matrix like a scoreboard: the diagonal (top-left to bottom-right) shows correct guesses; everything off the diagonal is a mistake.

#### E. The Consumer Safety Interpretation:
- In legal protection tools, **False Negatives are dangerous**: if the AI says *"This clause is Low Risk (safe)"* but it was actually a hidden account-deletion trap, the user gets hurt.
- If the AI is over-cautious and flags a Medium clause as High, the user just spends 30 extra seconds reading it — no real harm done.
- **Honest Limitation to State:** Our test set is only 13 clauses. Catching 6 out of 6 High-Risk clauses shows strong sensitivity on our benchmark, but it is **not** a guarantee it will work perfectly on millions of real-world contracts.

---

### 3.3 Tough Viva Cross-Questions & Exact Answers for Saloni

> **Examiner Q1: If Accuracy is identical (61.54%), why is BERT-Tiny better than Logistic Regression?**  
> **Saloni's Answer:** *"Sir, accuracy is misleading here — with only 13 test samples, every single correct or wrong prediction shifts accuracy by 7.7%. The real story is in the other metrics. BERT-Tiny's **Macro Precision is 0.8485 vs. 0.6528 for Logistic Regression — a +19.6% gain**. More importantly, the baseline Logistic Regression missed one High-risk clause (recall = 83.3%), but BERT caught all 6 High-risk clauses (recall = 100%) and made zero false positive errors on Low and Medium predictions."*

> **Examiner Q2: What is the difference between Macro Average, Micro Average, and Weighted Average?**  
> **Saloni's Answer:**  
> - *"**Micro Average:** Pools all True Positives, False Positives, and False Negatives together across all classes and computes one global score. In multi-class single-label problems, Micro F1 equals Accuracy.*  
> - *"**Macro Average:** Calculates the metric separately for each class, then takes a simple arithmetic average. Every class counts equally — no class can hide behind the others.*  
> - *"**Weighted Average:** Same as Macro, but each class is weighted by how many actual examples it has. We use Macro Average because we want Low, Medium, and High to be treated equally for consumer safety — we don't want the large High class to dominate the score."*

> **Examiner Q3: Why is your Macro Recall (0.5278) lower than the baseline (0.5833)?**  
> **Saloni's Answer:** *"Because BERT-Tiny is cautious with borderline clauses. Thanks to the weighted loss function, the model was trained to be very sensitive to High-risk misses. So when it was unsure about 2 Low clauses and 3 Medium clauses, it pushed them all into the High-risk category just to be safe. This brought down the recall for Low (0.33) and Medium (0.25), but made sure it caught every single High-risk clause — zero misses."*

---

# 📚 SECTION 4: NEHA MANIYAR (2413211)
### Domain: Inference Engine, Extractive Risk Triage vs. Summarization, Demo Interface & System Architecture

---

### 4.1 Core Technical Responsibilities
1. Built the standalone inference engine (`inference.py`) and the interactive terminal demo (`demo.py`) that users can run.
2. Handled edge cases — what happens when someone types nothing, types spaces, or uses special characters.
3. Wrote system architecture explanations and created the pipeline flowchart image (`fine_tuning_flowchart.png`).
4. Explained clearly why the system does **extractive risk triage** (flagging dangerous clauses) rather than **abstractive summarization** (generating new summary text), and documented this in `docs/08_*`.

---

### 4.2 Deep Technical Concepts Neha Must Know

#### A. How Real-Time Inference Works (`inference.py`):
1. Loads the saved model weights (`model_output/model.safetensors`) and tokenizer from disk.
2. Switches the model to evaluation mode: `model.eval()` — this disables training-only behaviours like Dropout.
3. Turns off gradient tracking: `with torch.no_grad():` — we don't need to learn anything during inference, so this saves memory and makes it faster.
4. Passes the raw input text through `BertTokenizer(truncation=True, padding="max_length", max_length=128)`.
5. Gets raw output scores (called **logits**) from the model.
6. Converts logits into proper probabilities using the **Softmax function**:
   $$P(\text{class } i) = \frac{e^{z_i}}{\sum_{j=0}^2 e^{z_j}}$$
   > **Plain English:** Take each raw score, raise $e$ to its power, then divide by the sum of all three. This squishes the three scores into numbers that add up to 1.0 — i.e., proper percentages.
7. Picks the highest probability class using $\text{argmax}(P)$ and maps it to the label (`Low`, `Medium`, `High`) with a confidence percentage.

#### B. The "Summarizer" in the Title vs. Implemented Classification (Crucial Viva Defense!):
- **Why Didn't We Train a Text Summarizer?**
  - True abstractive summarization (generating new shorter text) requires an Encoder-Decoder model like T5 or BART — trained on thousands of pairs of `(Long Clause, Human-Written Short Summary)`.
  - Consumer Terms & Conditions **do not come with pre-written human summaries**.
  - Making up fake summaries or calling a paid API would not count as genuine fine-tuning. We needed real, verifiable training.
- **How Our System Acts as an Extractive Risk Filter:**
  - A typical Terms & Conditions document is around 15,000 words. Nobody reads all of it.
  - Our system splits the document into clauses, scores each one, filters out the 90% of ordinary terms (`Low Risk`), and **brings the dangerous 10% (`High Risk`) to the user's attention**.
  - This approach is called **Extractive Risk Triage** in Legal Tech — like a doctor triaging patients to treat the critical cases first.
  - Full generative text-to-text summarization is our documented **future work**.

#### C. Edge Case & Robustness Engineering:
- **Empty / Whitespace Input:** If a user types just spaces or presses Enter accidentally, a naive script crashes with a tensor shape error. Neha's code catches this and returns `"UNKNOWN"` with a status of `"empty_input"` — graceful failure, not a crash.
- **Windows CP1252 Encoding:** Windows Command Prompt can crash when it sees Unicode box-drawing characters like `│`, `┌`, `─`. Neha made sure `demo.py` and `show_results.py` use only standard ASCII characters (`|`, `+`, `-`) so they work cleanly on any Windows machine.

#### D. Verified Sample Inferences (Neha Must Know These):
1. *"Amazon reserves the right in its sole discretion to terminate any user account without warning or compensation."*  
   $\rightarrow$ **HIGH RISK (Confidence: 73.46%)**  
   `[Low: 9.7% | Med: 16.9% | High: 73.5%]`
2. *"Please check our Privacy Notice to learn more about how cookies and personal identifiers are utilized."*  
   $\rightarrow$ **LOW RISK (Confidence: 60.64%)**  
   `[Low: 60.6% | Med: 28.3% | High: 11.1%]`
3. *"UNDER NO CIRCUMSTANCES SHALL THE VENDOR BE LIABLE FOR ANY CONSEQUENTIAL OR PUNITIVE DAMAGES."*  
   $\rightarrow$ **HIGH RISK (Confidence: 80.36%)**  
   `[Low: 8.7% | Med: 10.9% | High: 80.4%]`

---

### 4.3 Tough Viva Cross-Questions & Exact Answers for Neha

> **Examiner Q1: Your registered title says 'Summarizer and Clause Risk Analyzer'. Where is the summary? Why is it just outputting High, Medium, or Low?**  
> **Neha's Answer:** *"Sir/Madam, this is an important distinction. True abstractive summarization means the model reads a long clause and writes a shorter version in its own words. That requires a model like T5 or BART trained on thousands of real examples where each long clause has a matching human-written short summary. Contract datasets like ToS;DR only provide risk labels — not summaries. Instead of fabricating training data, we did genuine PyTorch fine-tuning on sequence classification. In legal tech, this is called **Extractive Risk Triage** — the system automatically filters out the harmless clauses and flags only the high-liability ones for the user to read. Generating actual text summaries is our documented future work."*

> **Examiner Q2: What do your confidence scores actually mean? Are they legal guarantees?**  
> **Neha's Answer:** *"No, Sir. The Softmax confidence scores are just mathematical probability distributions across three classes based on how similar the clause is to patterns the model learned. They show the model's relative confidence — not a legal verdict. We clearly mention this disclaimer in both the CLI demo and the documentation. No AI classifier can replace professional legal advice."*

> **Examiner Q3: How would you deploy this model in a real-world commercial setting?**  
> **Neha's Answer:** *"Because `BERT-Tiny` has only 4.39M parameters and the saved model file is just 16.7 MB, it runs extremely fast on a regular CPU — under 15ms per clause. It can be exported to ONNX format and run inside a browser extension using ONNX Runtime Web, or served through a FastAPI backend. Imagine a browser extension: as you navigate to a signup page, it automatically reads the Terms of Service, splits the text into clauses, runs each one through the model, and highlights dangerous clauses in red — all before you click 'I Agree'."*

---

## ⚡ Unified Emergency Numbers Cheat-Sheet (For All 4 Members)

| Technical Metric / Parameter | Value to State Confidently |
|:---|:---|
| **Pretrained Model Backbone** | `prajjwal1/bert-tiny` (Pretrained BERT Sequence Classifier) |
| **Model Parameters** | **4,386,307** (~4.39 Million parameters, all trainable) |
| **Model Architecture Details** | 2 Transformer Layers, 128 Hidden Dimension, 2 Attention Heads |
| **Total Dataset Size** | **82 clauses** across 11 platforms (Amazon, Alibaba, Google, Apple, etc.) |
| **Dataset Splits** | Stratified: 70% Train (57), 15% Validation (12), 15% Test (13) |
| **Optimizer & Hyperparameters** | **AdamW**, Learning Rate = $1 \times 10^{-4}$ (0.0001), Weight Decay = 0.01 |
| **Batch Size & Epochs** | Batch Size = 4, Total Epochs = 12 |
| **Loss Function** | **Weighted Cross-Entropy Loss** ($w_{Low}=1.462, w_{Med}=1.188, w_{High}=0.679$) |
| **Training Loss Drop** | Decreased monotonically from **1.1194 down to 0.4742** (57.6% drop) |
| **Validation Accuracy** | Climbed from **41.67% to 75.00%** |
| **Held-Out Test Set Size** | **13 clauses** (6 High, 4 Medium, 3 Low) |
| **Baseline Model** | TF-IDF (1-2 grams, 500 features) + Balanced Logistic Regression |
| **Macro Precision Comparison** | **BERT-Tiny: 0.8485** vs. **Baseline: 0.6528** (**+19.6% Absolute Gain**) |
| **High-Risk Class Recall** | **100.0%** (6 out of 6 High-Risk test clauses detected) |
| **Confusion Matrix** | `[[1, 0, 2], [0, 1, 3], [0, 0, 6]]` |
