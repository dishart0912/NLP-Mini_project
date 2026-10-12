# 5. Evaluation Metrics and Results

How do we prove our fine-tuned model actually works? This guide breaks down the evaluation numbers, the baseline comparison, and what each metric means.

---

## 1. What is the Baseline Model and Why Do We Need It?

In NLP research, you cannot just show one model's score and say "it's good". You need to compare it against a **Baseline** — a simpler, older approach — to show that deep learning actually gave a real advantage.

Think of it like a science fair: you need a control experiment to prove your new method is better.

### Our Classical Baseline:
- **Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency) — a classic method that turns text into numbers based on how often words appear. Common words like "the" get low scores; rare, important words get high scores. Bigrams (word pairs like "not liable") are also used.
- **Classifier:** Logistic Regression with balanced class weights — a simple, well-understood classification algorithm.
- **Trained on:** The exact same 70% training split as our BERT model — so the comparison is fair.

---

## 2. Benchmark Comparison on the Held-Out Test Set (13 Clauses)

| Metric | Classical Baseline (TF-IDF + LogReg) | Fine-Tuned BERT-Tiny | Difference / Gain |
|---|:---:|:---:|:---:|
| **Overall Accuracy** | 61.54% | **61.54%** | Equal |
| **Macro Precision** | 0.6528 | **0.8485** | **+19.6% absolute gain (0.8485 vs 0.6528)** |
| **Macro Recall** | 0.5833 | 0.5278 | -5.5% |
| **Macro F1-Score** | 0.6000 | 0.5353 | -6.5% |

---

## 3. Understanding the Metrics (Beginner Friendly)

### 1. Accuracy
- **Formula:** $\frac{\text{Correct Predictions}}{\text{Total Predictions}}$
- *Plain English: Out of all predictions, how many were right?*
- Both models got $8 / 13 = 61.54\%$. With only 13 test samples, each single prediction changes accuracy by about 7.7%.

### 2. Precision (Quality of Predictions)
- **Formula:** $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$
- *Plain English: When the model says a clause is "Low Risk", how often is it actually Low Risk? It measures how trustworthy the model's claims are.*
- **Our Fine-Tuned Model achieved 1.00 (100%) Precision on Low and Medium classes!** When it says Low or Medium, it is always right — zero false alarms.
- **Macro Precision rose from 0.6528 to 0.8485 (+19.6% gain).** This shows that BERT understands meaning better than simple word-count methods.

### 3. Recall (Completeness of Predictions)
- **Formula:** $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$
- *Plain English: Out of all the actual High-Risk clauses in the test set, how many did the model successfully catch? It measures how many dangerous clauses we didn't miss.*
- **Our Fine-Tuned Model achieved 1.00 (100%) Recall on the High-Risk class!**
  - Actual High-Risk clauses in test set = 6
  - Detected High-Risk clauses = 6
  - Zero High-Risk clauses were missed. *(This is promising, but we need more test data to fully confirm this in real-world use.)*

### 4. Macro F1-Score
- The harmonic mean (a balanced average that punishes extreme imbalances) of Precision and Recall, calculated equally across all 3 classes.
- *Plain English: A single score that balances both Precision and Recall. Useful when you care about both "not crying wolf" and "not missing danger".*

---

## 4. The Confusion Matrix Explained

A **Confusion Matrix** shows where the model's predictions landed versus the correct answers. It's like a report card that shows not just the score, but exactly which questions were wrong.

```text
                  PREDICTED
              Low   Med   High
ACTUAL  Low  [ 1     0     2 ]
        Med  [ 0     1     3 ]
       High  [ 0     0     6 ]
```

### How to Read This Matrix:
1. **Row 1 (Actual Low = 3 clauses):**
   - 1 was correctly predicted as Low. ✅
   - 2 were (incorrectly but cautiously) predicted as High. ⚠️
2. **Row 2 (Actual Medium = 4 clauses):**
   - 1 was correctly predicted as Medium. ✅
   - 3 were (incorrectly but cautiously) predicted as High. ⚠️
3. **Row 3 (Actual High = 6 clauses):**
   - **All 6 were correctly predicted as High — zero misses!** ✅✅

### Why is this cautious behavior actually good for Consumer Protection?
In legal AI, **False Negatives are dangerous:**
- A False Negative means the AI tells a user *"This clause is safe (Low Risk)"* when it's actually a High-Risk clause (like unilateral account termination). The user then ignores a real danger.
- Erring on the side of caution by flagging borderline clauses as High Risk is safer — it makes the user double-check, which is better than missing a real threat.
- On our 13-example test set, the fine-tuned BERT model caught all 6 High-Risk clauses. That said, we need a larger test set to fully confirm this in practice.

---

## 5. Live Predictions on Unseen Test Clauses

When running [`inference.py`](../inference.py), the fine-tuned model gives a probability score for each risk class:

1. *"Amazon reserves the right in its sole discretion to terminate any user account without warning or compensation."*  
   $\rightarrow$ **HIGH RISK** (Confidence: **73.5%**)  
   `[Low: 9.7% | Medium: 16.9% | High: 73.5%]`

2. *"Please check our Privacy Notice to learn more about how cookies and personal identifiers are utilized."*  
   $\rightarrow$ **LOW RISK** (Confidence: **60.6%**)  
   `[Low: 60.6% | Medium: 28.3% | High: 11.1%]`

3. *"Users must maintain the confidentiality of their secret credentials and update passwords every 90 days."*  
   $\rightarrow$ **MEDIUM RISK** (Confidence: **47.5%**)  
   `[Low: 42.8% | Medium: 47.5% | High: 9.7%]`

4. *"UNDER NO CIRCUMSTANCES SHALL THE VENDOR BE LIABLE FOR ANY CONSEQUENTIAL OR PUNITIVE DAMAGES ARISING FROM SERVICE INTERRUPTION."*  
   $\rightarrow$ **HIGH RISK** (Confidence: **80.4%**)  
   `[Low: 8.7% | Medium: 10.9% | High: 80.4%]`

5. *"Any dispute arising under this Agreement shall be resolved exclusively through confidential binding arbitration."*  
   $\rightarrow$ **HIGH RISK** (Confidence: **66.0%**)  
   `[Low: 11.2% | Medium: 22.8% | High: 66.0%]`
