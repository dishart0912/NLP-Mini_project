# Team Study & Presentation Guide: Slide Division & Viva Prep

**Project Title:** NLP-Based Terms & Conditions Clause Risk Analyzer Using BERT-Tiny  
*(Registered Topic: Automated Terms & Conditions Summarizer and Clause Risk Analyzer)*  
**BE Computer Engineering — Semester VII (Academic Year 2026)**  
**Course:** Natural Language Processing Lab  
**Group Number:** 35  

---

## 👥 Master Slide Division & Responsibility Matrix

| Speaker | Roll Number | Assigned Slides | Primary Topic & Contribution | Code / Demo Responsibility |
|:---|:---:|:---:|:---|:---|
| **Neha Maniyar** | 2413211 | **Slides 1 & 2** | Introduction, Problem Statement & Motivation | Presentation opening, problem context |
| **Disha Takawale** | 2413195 | **Slides 3 & 4** | System Architecture & 5 Preprocessing Steps | Run `python preprocessing_pipeline.py` |
| **Prerana Patankar** | 2413196 | **Slides 5 & 6** | Dataset Curation & BERT-Tiny Fine-Tuning | Run `python show_results.py` |
| **Saloni Patil** | 2413210 | **Slides 7, 8, 9, 10** | Evaluation Metrics, Sample Predictions, Future Work & Wrap-up | Run `python demo.py` |

---

## 🗣️ Member 1: Neha Maniyar (Roll No. 2413211)

### Assigned Slides: Slide 1 & Slide 2

---

### 📄 Slide 1: Title & Overview
- **What is on the slide:** Project title, subtitle (*"Making complex legal language easier to review"*), and team details.
- **Your Speaking Script:**
  > *"Good morning respected professors. Our project is titled **'Terms & Conditions Clause Risk Analyzer Using BERT-Tiny'**, registered under the topic of Automated Terms & Conditions Summarizer and Clause Risk Analyzer. Our team members are Disha Takawale, Prerana Patankar, Saloni Patil, and myself, Neha Maniyar."*

---

### 📄 Slide 2: The Challenge (Why do Terms & Conditions need a risk analyzer?)
- **What is on the slide:**
  - **The Problem:** Digital agreements average 8,000 to 15,000 words. Over 90% of users click "I Agree" without reading, forfeiting consumer rights.
  - **The NLP Challenge:** Legal text contains specialized jargon, nested sentences, and extreme negation sensitivity (a single word like "not" reverses legal obligations).
  - **Our Objective:** Analyze individual clauses and classify them into Low, Medium, or High risk with confidence scores.
- **Your Speaking Script:**
  > *"Digital contracts are intentionally verbose and written in dense legalese. As a result, consumers routinely accept them without review, unknowingly surrendering trial rights or accepting unilateral account termination.*  
  > *From an NLP perspective, legal text presents strict syntactic challenges: nested clauses, domain-specific terminology, and high negation sensitivity where omitting a single word like 'not' reverses the clause's legal impact.*  
  > *Our objective is to build an automated NLP system that parses agreements, analyzes clauses individually, and assigns actionable risk labels—Low, Medium, or High—along with exact probability distributions.*  
  > *Now, Disha will explain our system architecture and the preprocessing pipeline."*

---

### 💡 Viva Questions Neha Should Prepare:
1. **Q: Why is legal text harder for NLP than news articles or social media?**
   - **Ans:** Legal text has high syntactic embedding (sentences with 4-5 nested clauses), domain-specific archaic terminology (e.g., *indemnification, severability*), and extreme negation sensitivity where standard stopword removal can destroy meaning.
2. **Q: Why focus on clause-level analysis rather than the entire 15,000-word document at once?**
   - **Ans:** Transformers have context window limits (typically 512 tokens). Breaking contracts into individual clauses allows granular risk triage so users see exactly which specific terms are hazardous.

---

## 🗣️ Member 2: Disha Takawale (Roll No. 2413195)

### Assigned Slides: Slide 3 & Slide 4

---

### 📄 Slide 3: System Architecture (How the analyzer works)
- **What is on the slide:** 4-stage pipeline:
  1. Clause Input $\rightarrow$ 2. NLP Preprocessing $\rightarrow$ 3. BERT-Tiny Model $\rightarrow$ 4. Risk Output.
  - Note: Additional components: WordNet semantics, shallow/deep parsing, HMM POS tagging.
- **Your Speaking Script:**
  > *"Thank you, Neha. Our system operates as an end-to-end 4-stage pipeline:  
  > First, the raw legal clause is provided as input.  
  > Second, it passes through our linguistic preprocessing pipeline to normalize and structure the text.  
  > Third, the clause is encoded and processed by our fine-tuned BERT-Tiny sequence classification model.  
  > Fourth, the system outputs the predicted risk level—Low, Medium, or High—along with softmax confidence scores across all three classes.  
  > This pipeline is backed by our foundational experiments, including shallow and deep parsing, WordNet ontology, and a custom HMM POS tagger."*

---

### 📄 Slide 4: Preprocessing (Five checks that prepare legal text)
- **What is on the slide:**
  1. Sentence boundary identification (separates text into sentence units).
  2. Tokenization (splits text into words and WordPiece subwords).
  3. Lemmatization (maps inflections to base roots, e.g., *terminating* $\rightarrow$ *terminate*).
  4. POS tagging (identifies modals like *must* and *shall* and verbs).
  5. WordNet semantic analysis (extracts synsets, hypernyms, and hyponyms).
- **Your Speaking Script:**
  > *"To satisfy all mandatory NLP lab guidelines, our preprocessing pipeline implements five essential operations:  
  > 1. **Sentence Boundary Identification:** Using `nltk.sent_tokenize` to isolate discrete contractual conditions from long paragraphs.  
  > 2. **Tokenization:** Lexical word tokenization via NLTK, alongside subword WordPiece tokenization for the transformer.  
  > 3. **Lemmatization:** POS-informed WordNet lemmatization, ensuring legal words like 'terminating' map to 'terminate' without the crude truncation caused by stemmers.  
  > 4. **POS Tagging:** Identifying parts of speech to isolate modal auxiliaries like 'shall' and 'must' (tagged as `MD`), which indicate non-negotiable obligations.  
  > 5. **WordNet Semantic Analysis:** Extracting ontological relationships for legal keywords—for example, connecting 'liability' to its broader hypernyms like legal obligation and debt.  
  > Now, Prerana will walk through our dataset curation and model fine-tuning."*

---

### 💻 Code Demo for Disha:
Run in terminal:
```bash
python preprocessing_pipeline.py
```
**Point to on screen:**
- Sentence split count.
- Modal verbs (`shall`, `must`) tagged as `MD`.
- Lemma: `damages` $\rightarrow$ `damage`.
- WordNet synset and definition for `liable`.

---

### 💡 Viva Questions Disha Should Prepare:
1. **Q: Why POS-informed lemmatization instead of Porter stemming?**
   - **Ans:** Porter stemming chops off suffixes heuristically (*terminating* becomes *terminat*, which is not an English word). POS-informed WordNet lemmatization uses dictionary morphology and part-of-speech context to return the grammatically valid base lemma (*terminate*).
2. **Q: Is the NLTK preprocessed output fed into BERT?**
   - **Ans:** No! This is an important architectural distinction. BERT uses its own subword WordPiece tokenizer (`BertTokenizer`) on raw text so that sentence structure and punctuation are preserved for self-attention. The NLTK preprocessing is an independent analytical verification module.

---

## 🗣️ Member 3: Prerana Patankar (Roll No. 2413196)

### Assigned Slides: Slide 5 & Slide 6

---

### 📄 Slide 5: Dataset (A curated dataset for clause-risk classification)
- **What is on the slide:**
  - 82 curated clauses across 11 platforms (Amazon, Alibaba, Google, Apple, Microsoft, Meta, Netflix, X, Spotify, Uber, Zoom).
  - Risk Distribution: High (40), Medium (23), Low (19).
  - Stratified split: 70% Train (57), 15% Val (12), 15% Test (13).
- **Your Speaking Script:**
  > *"Thank you, Disha. Regarding our dataset: authentic consumer contracts do not come with gold-standard human abstractive summaries. Therefore, formulating our task as sequence risk classification provides verifiable, mathematically defensible ground-truth targets.  
  > We compiled a curated benchmark of **82 contractual clauses** across 11 major tech platforms, including Amazon, Alibaba, Google, Apple, Microsoft, and Netflix.  
  > The dataset includes 40 High-risk, 23 Medium-risk, and 19 Low-risk clauses.  
  > We used a stratified train-val-test split: 70% for training (57 clauses), 15% for validation (12 clauses), and an isolated 15% held-out test set (13 clauses) to guarantee fair evaluation."*

---

### 📄 Slide 6: Model Training (Fine-tuning BERT-Tiny)
- **What is on the slide:**
  - Backbone: `BERT-Tiny` (2 layers, 128 hidden dim, 2 heads, ~4.39M parameters).
  - Training Config: Optimizer AdamW, LR 0.0001, Weight Decay 0.01, Batch 4, Epochs 12.
  - Loss: Weighted Cross-Entropy ($w_{Low}=1.462$, $w_{Med}=1.188$, $w_{High}=0.679$).
  - Result: Training loss decreased from **1.1194 down to 0.4742**; validation accuracy reached 75.00%.
- **Your Speaking Script:**
  > *"For our deep learning component, we selected `prajjwal1/bert-tiny`, an efficient 2-layer transformer with 128 hidden dimensions, 2 attention heads, and approximately 4.39 million trainable parameters. This compact size avoids overfitting on our specialized legal dataset while providing true bidirectional self-attention.  
  > We fine-tuned the model for 12 epochs using the **AdamW optimizer** with a learning rate of $1 \times 10^{-4}$ and batch size of 4.  
  > To address class imbalance, we implemented **Weighted Cross-Entropy Loss** with weights computed strictly on the training set.  
  > Across the 12 epochs, training loss dropped from **1.1194 down to 0.4742**—a 57.6% relative loss reduction—while validation accuracy reached 75.00%. We monitored validation loss at every epoch and reloaded the best-performing checkpoint from Epoch 12 for testing.  
  > Now, Saloni will explain our evaluation metrics and demonstration."*

---

### 💻 Code Demo for Prerana:
Run in terminal:
```bash
python show_results.py
```
**Point to on screen:**
- Serialized weights: `model_output/model.safetensors` (~16.7 MB).
- 4,386,307 trainable parameters.
- Epoch-by-epoch loss table showing loss decreasing from 1.1194 down to 0.4742.

---

### 💡 Viva Questions Prerana Should Prepare:
1. **Q: Why BERT-Tiny instead of BERT-Base (110M params) or a 7B LLM?**
   - **Ans:** BERT-Base has 110 million parameters; on an 82-clause dataset, it would memorize the training data and severely overfit. BERT-Tiny has 4.4M parameters, which balances expressive transformer self-attention with generalization on smaller domain datasets while running efficiently on CPU.
2. **Q: How does AdamW differ from standard Adam?**
   - **Ans:** Standard Adam applies L2 regularization to gradients, which interacts poorly with momentum updates. AdamW decouples weight decay directly from gradient updates, leading to better regularization and generalization.
3. **Q: What is the best checkpoint selection?**
   - **Ans:** Instead of just saving the final 12th epoch, the training loop records validation loss at every epoch. The checkpoint with the lowest validation loss is automatically reloaded before test evaluation to prevent overfitting.

---

## 🗣️ Member 4: Saloni Patil (Roll No. 2413210)

### Assigned Slides: Slides 7, 8, 9 & 10

---

### 📄 Slide 7: Evaluation (Comparing the baseline with BERT-Tiny)
- **What is on the slide:**
  - Metrics table: Test Accuracy: 61.54% (both) | Macro Precision: 0.6528 vs **0.8485 (+19.6% gain)** | Macro Recall: 0.5833 vs 0.5278 | Macro F1: 0.6000 vs 0.5353.
  - High-risk recall: 1.00 (6 of 6 high-risk test clauses caught). Note: Small test set limits generalization.
- **Your Speaking Script:**
  > *"Thank you, Prerana. To scientifically verify our fine-tuned transformer, we compared it against a classical baseline: **TF-IDF (1-2 grams) with balanced Logistic Regression**, trained on the exact same 70% split.  
  > On the held-out test set of 13 unseen clauses:  
  > - Both models achieved an accuracy of **61.54%** (8 of 13 correct).  
  > - However, fine-tuned BERT-Tiny achieved a **Macro Precision of 0.8485**, delivering an absolute gain of **+19.6%** over the baseline's 0.6528, with perfect 1.00 precision on Low and Medium classes.  
  > - Crucially, BERT-Tiny achieved **1.00 Recall on High-Risk clauses**—catching all 6 high-risk clauses in the test set. Because the test set is 13 samples, this demonstrates high sensitivity on our benchmark rather than an absolute commercial guarantee."*

---

### 📄 Slide 8: Model Demonstration (Example clauses and predicted risk)
- **What is on the slide:**
  - Example 1: *"Terminate any user account without warning"* $\rightarrow$ **HIGH (73.46%)**
  - Example 2: *"Check our Privacy Notice to learn how cookies are used"* $\rightarrow$ **LOW (60.64%)**
  - Example 3: *"Vendor shall not be liable for consequential damages"* $\rightarrow$ **HIGH (80.36%)**
- **Your Speaking Script:**
  > *"Here are real-world predictions produced by our model:  
  > - For unilateral termination: 'Terminate any user account without warning', the model predicts **HIGH RISK with 73.46% confidence**.  
  > - For benign privacy notices: 'Check our Privacy Notice to learn how cookies are used', it predicts **LOW RISK with 60.64% confidence**.  
  > - For complete liability disclaimers: 'Vendor shall not be liable for consequential damages', it outputs **HIGH RISK with 80.36% confidence**."*

---

### 📄 Slide 9: Future Work (A prototype today. A stronger assistant tomorrow.)
- **What is on the slide:**
  - Small dataset (82 clauses) $\rightarrow$ Future: Expand with LexGLUE / ToS;DR.
  - Limited scope (Classification) $\rightarrow$ Future: Full-document abstractive summarization.
  - Roadmap: Grounded explanations with RAG; browser extension for real-time contract auditing.
- **Your Speaking Script:**
  > *"As honest engineering limitations:  
  > First, our current dataset contains 82 clauses; future work involves evaluating larger corpora like LexGLUE and ToS;DR on GPU hardware.  
  > Second, while our current model performs extractive risk triage, full document text summarization is our next phase using generative encoder-decoder models. We also aim to incorporate Retrieval-Augmented Generation (RAG) and package the model into a browser extension."*

---

### 📄 Slide 10: Conclusion & Wrap-Up
- **What is on the slide:** Takeaway: *"Make the fine print more visible."* Preprocess, Classify, Evaluate.
- **Your Speaking Script:**
  > *"In conclusion, we developed an end-to-end NLP system integrating text preprocessing, linguistic analysis, baseline benchmarking, and genuine BERT-Tiny fine-tuning to make legal fine print more transparent.  
  > Thank you, professors! We are now ready for your questions."*

---

### 💻 Code Demo for Saloni:
Run in terminal:
```bash
python demo.py
```
**Interactive Demo Steps:**
- Press `1`: Shows Amazon unilateral termination $\rightarrow$ HIGH RISK (73.5%).
- Press `4`: Shows standard Privacy Notice $\rightarrow$ LOW RISK (60.6%).
- Press `6`: Type or paste any custom clause the professor asks for!

---

### 💡 Viva Questions Saloni Should Prepare:
1. **Q: Why is accuracy identical (61.54%) between baseline and BERT?**
   - **Ans:** In a small held-out test split of 13 clauses, each sample represents ~7.7% of accuracy. Both models got 8 out of 13 correct. But accuracy treats all mistakes equally. BERT-Tiny achieved a **+19.6% higher Macro Precision (0.8485 vs 0.6528)** and achieved 100% precision on Low and Medium classes, meaning it makes far fewer false accusations on safe clauses.
2. **Q: Explain the confusion matrix:**
   $$\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 3 \\ 0 & 0 & 6 \end{pmatrix}$$
   - **Ans:** Rows are Actual (Low, Med, High) and Columns are Predicted (Low, Med, High). Row 3 shows `[0, 0, 6]`, meaning all 6 actual High-Risk test clauses were predicted as High (0 false negatives).
3. **Q: Why doesn't the model generate text summaries if the title says Summarizer?**
   - **Ans:** Authentic contracts do not have gold-standard human abstractive summaries to train on. We formulated the task as 3-class clause risk classification, which acts as **extractive risk triage**—highlighting the dangerous clauses for the user. Text-to-text generative summarization is documented as future work.

---

## ⚡ Emergency Cheat-Sheet: Quick Numbers Every Member Must Know

| Parameter | Value |
|:---|:---|
| **Model Name** | `prajjwal1/bert-tiny` |
| **Total Parameters** | 4,386,307 (~4.39 Million) |
| **Dataset Size** | 82 clauses across 11 platforms |
| **Dataset Split** | 70% Train (57), 15% Val (12), 15% Test (13) |
| **Number of Epochs** | 12 |
| **Optimizer & Learning Rate** | AdamW, $\eta = 1 \times 10^{-4}$ (0.0001) |
| **Training Loss Convergence** | 1.1194 $\rightarrow$ 0.4742 (57.6% drop) |
| **Validation Accuracy** | 41.67% $\rightarrow$ 75.00% |
| **Test Accuracy** | 61.54% (8 / 13) |
| **Macro Precision** | 0.8485 (BERT) vs. 0.6528 (Baseline) $\rightarrow$ **+19.6% Gain** |
| **High-Risk Test Recall** | 100.0% (6 / 6 detected in test set) |
