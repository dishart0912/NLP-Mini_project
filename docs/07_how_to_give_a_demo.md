# How to Give a Live Demo of This Project

This guide provides a step-by-step walkthrough for demonstrating the **Automated Terms & Conditions Summarizer and Clause Risk Analyzer** to your instructor, external examiner, or class.

---

## 3-Minute Demo Plan

| Step | Time | What to Say / Show | Command / Action |
|---|---|---|---|
| **1. Hook & Problem** | 30s | Explain why Terms & Conditions are dangerous for digital users. | Show [01_project_overview_and_concepts.md](01_project_overview_and_concepts.md) |
| **2. Mandatory Preprocessing** | 45s | Demonstrate tokenization, lemmatization, and WordNet legal semantics. | `python preprocessing_pipeline.py` |
| **3. Live Model Demonstration** | 90s | Run the interactive demo and let the examiner test any clause! | `python demo.py` |
| **4. Architecture & Metrics** | 15s | Show the flowchart and test set gains (+19.6% precision, 100% High-Risk recall). | Open `fine_tuning_flowchart.png` |

---

## Step 1: Open Your Terminal & Start the Demo

Open PowerShell or Command Prompt in the project directory:
```bash
python demo.py
```

You will see the interactive terminal menu:
```text
===========================================================================
  AUTOMATED TERMS & CONDITIONS RISK ANALYZER - LIVE DEMO MENU
===========================================================================
Choose an option:
  [1] Test Sample: Amazon Unilateral Termination
  [2] Test Sample: Alibaba Liability Disclaimer
  [3] Test Sample: Microsoft Mandatory Arbitration
  [4] Test Sample: Standard Privacy Notice
  [5] Test Sample: Account Password Security
  [6] Type or paste your OWN custom legal clause
  [0] Exit Demo
===========================================================================
```

---

## Step 2: Walk Through 3 Sample Demonstrations

### Demo Action 1: Show a High-Risk Clause (Press `1` or `2`)
- **Clause:** *"Amazon reserves the right in its sole discretion to terminate accounts or cancel orders without notice."*
- **What happens:** The model predicts **HIGH RISK (73.5% confidence)** and shows a visual probability bar.
- **What to tell the examiner:**
  > *"Notice how the model caught the words 'sole discretion' and 'terminate without notice'. It flagged this as High Risk because it strips user rights."*

### Demo Action 2: Show a Safe / Low-Risk Clause (Press `4`)
- **Clause:** *"Please review our Privacy Notice to understand how we collect and protect personal information."*
- **What happens:** The model predicts **LOW RISK (60.6% confidence)**.
- **What to tell the examiner:**
  > *"Here, the clause is purely informational, so the model correctly identifies it as Low Risk."*

### Demo Action 3: The "Magic Moment" — Let the Examiner Test a Custom Clause! (Press `6`)
Ask your professor:
> *"Sir / Ma'am, do you have any legal clause you want to paste or test live?"*

If they give you a sentence, paste it in! If they ask you to type one, use:
> *"The company shall not be liable for any loss of data or financial damages."*

The model will instantly analyze it and output:
```text
PREDICTED RISK LEVEL : HIGH
PROBABILITY BREAKDOWN:
  🟢 Low Risk    :   8.7%  |██░░░░░░░░░░░░░░░░░░░░░░░|
  🟡 Medium Risk :  10.9%  |███░░░░░░░░░░░░░░░░░░░░░░|
  🔴 High Risk   :  80.4%  |████████████████████░░░░░|
```

---

## Step 3: Show the Preprocessing Pipeline
If the instructor asks: *"Did you implement POS tagging, Lemmatization, and WordNet?"*

Run:
```bash
python preprocessing_pipeline.py
```
**Point to the screen:**
- Show **WordNet Analysis**: Point out how `"liability"` maps to its legal definition (*"the state of being legally obliged and responsible"*).
- Show **Lemmatization**: Point out how `"damages"` was reduced to the base root `"damage"`.

---

## Step 4: Show the Flowchart & Metrics
- Open [`fine_tuning_flowchart.png`](../fine_tuning_flowchart.png) on your screen.
- State the key finding:
  > *"Our fine-tuned BERT achieved 100% recall on the 6 high-risk clauses in our held-out test split, along with a +19.6% gain in macro precision over the classical baseline."*
