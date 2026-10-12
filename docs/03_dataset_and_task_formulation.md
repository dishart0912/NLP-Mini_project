# 3. Dataset Curation and Task Formulation

## 1. Why Did We Choose Clause Risk Classification Instead of Summarization?

The project title says:  
*"Automated Terms & Conditions Summarizer and Clause Risk Analyzer"*

The guidelines asked us to:
> *"Select the task and model based on the existing dataset and its labels. Do not assume the current dataset contains summarization targets or risk labels unless you verify that it does. If the available dataset supports clause classification more reliably than summarization, explain this and recommend the most defensible task."*

### The Honest Engineering Reality:
1. **Summarization Needs Pairs:** To train a summarization model (like T5 or BART — encoder-decoder models that generate new text), you need thousands of pairs of `(Long Legal Clause, Gold-Standard Human Summary)`. A "gold-standard summary" means a summary that a human expert actually wrote — not auto-generated.
2. **What We Actually Have:** Our existing project files ([alibaba.txt](../alibaba.txt), [amazon.txt](../amazon.txt), and [tc_clauses.csv](../tc_clauses.csv)) contain raw legal clauses with **risk labels** (`Low`, `Medium`, `High`). They do **not** have any human-written summaries.
3. **The Defensible Choice:** If we tried summarization, we'd have to make up summaries or use unverified tools — that's not honest engineering. **Legal Clause Risk Classification** uses our real, verified labels, so it is the most honest and scientifically sound choice.

---

## 2. Dataset Expansion: From 17 to 82 Clauses

### The Limitation of the Seed Dataset:
The original `tc_clauses.csv` had only **17 rows**:
- High: 9
- Medium: 5
- Low: 3

If you split 17 rows into 70% train (11 rows) and 15% test (3 rows), the model can't reliably learn or be evaluated — you'd have only 1 test sample per class! That's too small to trust any results.

### How We Solved It Without Altering the Original:
1. We kept `tc_clauses.csv` completely untouched (original file, original data).
2. We wrote [`prepare_expanded_dataset.py`](../prepare_expanded_dataset.py) to build a new, bigger file: [`tc_clauses_expanded.csv`](../tc_clauses_expanded.csv).
3. We manually picked and labeled extra clauses from our existing `amazon.txt` and `alibaba.txt` source files.
4. We added authentic Terms of Service clauses from 9 other major consumer platforms: Google, Apple, Microsoft, Meta, Netflix, Twitter/X, Spotify, Uber, and Zoom.
*(Note: The raw `amazon.txt` and `alibaba.txt` files are our background reference texts. Clauses were checked and structured carefully in Python to make sure the labels are clean and correct — no messy scraping.)*

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
| **High** | Clauses that take away your legal rights, remove financial responsibility from the company, force you into private arbitration, or let the company delete your account without warning. | *"Amazon reserves the right in its sole discretion to terminate accounts without notice."*<br>*"In no event will vendor be liable for any indirect or consequential damages."* |
| **Medium** | Clauses that put restrictions on what you can do, require you to keep things confidential, give the company rights over your content, or set rules about where disputes are handled. | *"You agree not to copy, resell, or distribute any information on the sites."*<br>*"Users must keep their account password secure and private."* |
| **Low** | Standard informational notices, privacy statements, or basic electronic communication agreements — mostly harmless. | *"Please review our Privacy Notice to understand our practices."*<br>*"You consent to receive communications from us electronically."* |

---

## 4. Dataset Splits (Stratified Sampling)

A key rule in machine learning: **never test your model on data it has already seen during training!** That would be like giving students the exact exam questions during practice — the results wouldn't be trustworthy.

We used a **Stratified Train / Validation / Test Split** — "stratified" means the proportion of Low/Medium/High risk clauses is kept the same in every split:

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
Legal datasets naturally have more High-risk clauses than Low-risk ones. Without stratification, your test set might accidentally have no Low-risk clauses at all!  
Stratification guarantees that the ratio of Low : Medium : High is the same across training, validation, and test sets — making results fair and trustworthy.
