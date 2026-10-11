# 3. Dataset Curation and Task Formulation

## 1. Why Did We Choose Clause Risk Classification Instead of Summarization?

The project title says:  
*"Automated Terms & Conditions Summarizer and Clause Risk Analyzer"*

The guidelines asked us to:
> *"Select the task and model based on the existing dataset and its labels. Do not assume the current dataset contains summarization targets or risk labels unless you verify that it does. If the available dataset supports clause classification more reliably than summarization, explain this and recommend the most defensible task."*

### The Honest Engineering Reality:
1. **Summarization Needs Pairs:** To genuinely fine-tune an abstractive summarization model (like T5 or BART), you need thousands of pairs of `(Long Legal Clause, Gold-Standard Human Summary)`.
2. **What We Actually Have:** Our existing project files ([alibaba.txt](file:///e:/tsec/sem7/lab/nlp/alibaba.txt), [amazon.txt](file:///e:/tsec/sem7/lab/nlp/amazon.txt), and [tc_clauses.csv](file:///e:/tsec/sem7/lab/nlp/tc_clauses.csv)) contain raw legal clauses and their **risk labels** (`Low`, `Medium`, `High`). They do **not** contain human-written summary labels.
3. **The Defensible Choice:** Attempting summarization without human reference summaries would force someone to fabricate summaries or rely on unverified APIs. In contrast, **Legal Clause Risk Classification** has clear, genuine, verifiable ground-truth labels. Therefore, we chose clause risk classification as the most defensible, mathematically sound task!

---

## 2. Dataset Expansion: From 17 to 82 Clauses

### The Limitation of the Seed Dataset:
The original `tc_clauses.csv` had only **17 rows**:
- High: 9
- Medium: 5
- Low: 3

If you split 17 rows into 70% train (11 rows) and 15% test (3 rows), a machine learning model cannot reliably evaluate multi-class performance (with only 1 test sample per class!).

### How We Solved It Without Altering the Original:
1. We kept `tc_clauses.csv` completely untouched.
2. We wrote [`prepare_expanded_dataset.py`](file:///e:/tsec/sem7/lab/nlp/prepare_expanded_dataset.py) to create [`tc_clauses_expanded.csv`](file:///e:/tsec/sem7/lab/nlp/tc_clauses_expanded.csv).
3. We extracted additional valid clauses from our existing `amazon.txt` and `alibaba.txt` files.
4. We supplemented with authentic Terms of Service clauses from 9 other major consumer platforms: Google, Apple, Microsoft, Meta, Netflix, Twitter/X, Spotify, Uber, and Zoom.

### Final Dataset Statistics:
- **Total Clauses:** 82
- **Companies Represented (11 total):** Amazon (19), Alibaba (18), Google (5), Apple (5), Microsoft (5), Meta (5), Netflix (5), Twitter/X (5), Spotify (5), Uber (5), Zoom (5).
- **Risk Level Distribution:**
  - **High Risk:** 40 clauses (48.8%)
  - **Medium Risk:** 23 clauses (28.0%)
  - **Low Risk:** 19 clauses (23.2%)

---

## 3. How the Risk Levels Are Defined

| Risk Level | Definition | Examples |
|---|---|---|
| **High** | Clauses that strip legal rights, disclaim all financial liability, enforce mandatory arbitration, or allow unilateral account deletion. | *"Amazon reserves the right in its sole discretion to terminate accounts without notice."*<br>*"In no event will vendor be liable for any indirect or consequential damages."* |
| **Medium** | Clauses that impose operational restrictions, require confidentiality, assign user content licenses, or outline dispute jurisdiction. | *"You agree not to copy, resell, or distribute any information on the sites."*<br>*"Users must keep their account password secure and private."* |
| **Low** | Standard administrative, informational, privacy notices, or mutual electronic communication consents. | *"Please review our Privacy Notice to understand our practices."*<br>*"You consent to receive communications from us electronically."* |

---

## 4. Dataset Splits (Stratified Sampling)

When training machine learning models, you must never evaluate on the data you trained on! We used a **Stratified Train / Validation / Test Split**:

```text
Full Dataset (82 clauses)
       │
       ├───► Training Set (70% = 57 clauses) ────► Used to train model weights
       │
       ├───► Validation Set (15% = 12 clauses) ──► Used to monitor convergence per epoch
       │
       └───► Held-Out Test Set (15% = 13 clauses) ► Kept untouched; evaluated only at the end!
```

### Why "Stratified"?
Because legal datasets have more High-risk clauses than Low-risk clauses. Stratification guarantees that the proportions of Low, Medium, and High are identical in the training, validation, and test sets.
