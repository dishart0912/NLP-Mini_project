# Quick Revision Guide: Experiments 1 to 5 & HMM POS Tagger

**BE Computer Engineering — Semester VII**  
**Course:** Natural Language Processing Lab (Academic Year 2026)  
**Group Number:** 35  
**Project Title:** Automated Terms & Conditions Summarizer and Clause Risk Analyzer  

---

## 🎯 Purpose of This Guide
During your project viva, examiners often test your fundamentals by asking:
> *"Before fine-tuning BERT, what experiments did you perform in the lab?"*  
> *"What were the objectives and findings of Experiments 1 to 5 and your HMM tagger?"*

This quick-revision sheet summarizes **each experiment's objective, tools, results, and likely viva questions** in simple language so any team member can answer immediately.

---

## 🧭 Quick Summary Table

| Exp # | Topic | Core Focus | Key Tools | Key Takeaway / Result |
|:---:|:---|:---|:---|:---|
| **1** | **NLP Tools & Applications** | Comparing Open-Source vs Cloud NLP | NLTK, spaCy, Hugging Face, Google Cloud NLP, AWS Comprehend | Open-source gives full parameter control & privacy; Cloud APIs charge per call and don't allow weight fine-tuning. |
| **2** | **Problem Formulation** | Identifying the Legal NLP Problem | Python, pandas, ToS;DR dataset | Defined 3-class risk classification (`Low`, `Medium`, `High`) based on consumer contract clauses. |
| **3** | **Basic Text Preprocessing** | Cleaning raw text without destroying meaning | BeautifulSoup4, contractions, regex, NLTK | Preserved negation words (`not`, `never`) and currency symbols (`$100`) critical for legal liability. |
| **4** | **Advanced Preprocessing** | Lemmatization, Stemming, PII masking | WordNet Lemmatizer, Porter/Lancaster Stemmer, Presidio/Regex | Stemming destroys legal terms (`terminating` → `terminat`); Lemmatization preserves valid words. Masked GDPR emails/names. |
| **5** | **Shallow & Deep Parsing** | Finding grammar structure & actors | NLTK RegexpParser, spaCy dependency parser | NP/VP chunking finds **Who** (Actor: Customer) does **What** (Action: must maintain credentials). Extracted SVO triples. |
| **HMM** | **HMM POS Tagging from Scratch** | Statistical POS tagging via Viterbi | Python math/numpy, NLTK (baseline) | Calculated Transition & Emission probabilities; solved most likely tag sequence via dynamic programming. |

---

# 📖 Experiment 1: Study of NLP Tools and Applications

### 1. Objective:
To evaluate and compare foundational open-source NLP libraries with cloud-based proprietary NLP platforms.

### 2. Tools Compared:
* **Category 1 (Open-Source):**
  * **NLTK:** Best for education and linguistic research (tokenization, POS tagging, WordNet).
  * **spaCy:** Industrial-strength, high-speed C-optimized pipeline (Named Entity Recognition, Dependency Parsing).
  * **Hugging Face Transformers:** State-of-the-art pretrained transformer backbones (BERT, RoBERTa, T5).
* **Category 2 (Cloud NLP APIs):**
  * **Google Cloud Natural Language API** & **AWS Comprehend:** Pre-built cloud APIs for sentiment, entity extraction, and syntax.

### 3. Key Finding for Viva:
* **Why not use Cloud APIs for this project?**  
  Cloud APIs are "black boxes". You cannot inspect internal attention weights, you cannot fine-tune on local custom datasets without enterprise costs, and sending confidential legal contracts to external cloud servers violates data privacy.

---

# 📖 Experiment 2: Problem Formulation & Mini-Project Setup

### 1. Objective:
To explore NLP application domains (Sentiment, Summarization, NER, Classification) and formalize our mini-project problem statement.

### 2. Problem Statement Formulated:
> Consumers routinely accept multi-page Terms and Conditions without reading them. Companies hide unilateral account termination, liability disclaimers, and mandatory arbitration clauses. We formulated an NLP system to analyze legal contracts and classify clauses into **Low**, **Medium**, and **High** risk levels.

### 3. Literature Surveyed:
* **ToS;DR (Terms of Service; Didn't Read):** Community project labeling unfair contract terms.
* **LexGLUE:** Academic legal benchmark for legal NLP tasks.

---

# 📖 Experiment 3: Basic Text Preprocessing

### 1. Objective:
To build a preprocessing pipeline that cleans dirty web text while **preserving critical legal meaning**.

### 2. Operations Implemented:
1. **HTML & Tag Stripping:** Removed `<div>`, `<p>`, `&nbsp;` using BeautifulSoup.
2. **Contraction Expansion:** Expanded `"won't"` → `"will not"`, `"can't"` → `"cannot"`.
3. **Number & Currency Normalization:** Kept monetary caps (e.g., `"$1,000"`, `"100%"`) because they define liability limits in law.
4. **Negation Preservation:** **Did NOT remove stopwords like `"not"`, `"no"`, `"never"`!**  
   * *Why?* If you remove `"not"`, the clause *"Vendor will not be liable"* becomes *"Vendor will be liable"* — completely reversing the legal meaning!

---

# 📖 Experiment 4: Advanced Preprocessing & Morphology

### 1. Objective:
To compare stemming vs. lemmatization on legal words and implement privacy masking (GDPR/PII).

### 2. Key Findings:
* **Stemming vs Lemmatization (The Golden Viva Question!):**
  * **Porter Stemmer:** Uses heuristic chopping rules.
    * `"terminating"` → `"terminat"` (non-word)
    * `"liabilities"` → `"liabil"` (non-word)
    * `"indemnification"` → `"indemnif"` (broken)
  * **WordNet Lemmatizer:** Uses morphological dictionary lookup with POS tags.
    * `"terminating"` (with `pos='v'`) → `"terminate"` (valid dictionary root)
    * `"liabilities"` (with `pos='n'`) → `"liability"`
* **PII Anonymization (GDPR compliance):**
  * Masked email addresses (`user@domain.com` → `[EMAIL]`) and phone numbers to ensure consumer privacy.

---

# 📖 Experiment 5: Shallow and Deep Parsing

### 1. Objective:
To extract grammatical structures, contractual actors, and obligations using shallow (chunking) and deep (dependency) parsing.

### 2. Operations & Results:
1. **Shallow Parsing (NP/VP Chunking via NLTK RegexpParser):**
   * Rule: `NP: {<DT>?<JJ>*<NN.*>+}` and `VP: {<MD>?<VB.*>+}`
   * *Example:*
     * `[NP: The customer]` $\rightarrow$ **Contractual Actor (Subject)**
     * `[VP: must maintain]` $\rightarrow$ **Mandatory Legal Obligation**
     * `[NP: confidential credentials]` $\rightarrow$ **Object**
2. **Deep Parsing (Subject-Verb-Object Triples):**
   * Extracted relational triples: `(Amazon, reserves, right)`, `(Customer, waives, trial)`.
3. **Context-Free Grammar (CFG):**
   * Defined CFG production rules (`S -> NP VP`, `VP -> V NP`) to parse formal contractual syntax trees.

---

# 📖 HMM Implementation: Viterbi POS Tagger from Scratch

### 1. Objective:
To build a **Hidden Markov Model (HMM)** with the **Viterbi Algorithm** from mathematical scratch (without relying on pre-built libraries like NLTK or spaCy) for POS tagging.

### 2. Mathematical Components:
1. **Hidden States ($S$):** The grammatical POS tags (e.g., `NNP`, `VBZ`, `JJ`, `NNS`, `MD`).
2. **Observations ($O$):** The actual words in the sentence (e.g., *"Amazon"*, *"provides"*, *"services"*).
3. **Transition Probability Matrix ($A$):**
   $$P(\text{tag}_j \mid \text{tag}_i) = \frac{\text{Count}(\text{tag}_i \rightarrow \text{tag}_j)}{\text{Count}(\text{tag}_i)}$$
   *The probability of tag $j$ occurring immediately after tag $i$.*
4. **Emission Probability Matrix ($B$):**
   $$P(\text{word}_k \mid \text{tag}_i) = \frac{\text{Count}(\text{tag}_i \text{ emits } \text{word}_k)}{\text{Count}(\text{tag}_i)}$$
   *The probability that a given POS tag generates that specific word.*

### 3. Viterbi Algorithm (Dynamic Programming):
* **Why Viterbi?** Checking all possible tag sequences for a sentence has exponential complexity $\mathcal{O}(T^N)$. Viterbi uses dynamic programming to find the single most probable tag path in $\mathcal{O}(N \times T^2)$ time.
* **Viterbi Trellis Formula:**
  $$V_t(j) = \max_i \left( V_{t-1}(i) \times P(\text{tag}_j \mid \text{tag}_i) \right) \times P(\text{word}_t \mid \text{tag}_j)$$
  *Plain English:* At each word, find the previous tag that gives the highest total probability when combined with the transition and emission probabilities.
* **Backpointer Array:** Keeps track of the best path so the algorithm can backtrack from the final word to output the sequence of tags.

---

## ⚡ Top 5 Viva Questions on Experiments 1–5 & HMM

> **Q1: Why did you keep negation words during text cleaning in Experiment 3?**  
> **Answer:** *"In general sentiment analysis, standard stopwords like 'not' and 'never' are often removed. But in legal contracts, removing 'not' reverses the legal meaning — turning 'shall not be liable' into 'shall be liable', which inverts the contractual obligation."*

> **Q2: Why did Porter Stemming fail on legal texts in Experiment 4?**  
> **Answer:** *"Porter Stemming cuts word endings using rough heuristic rules without checking a dictionary. It produced truncated non-words like 'terminat' and 'liabil'. In contrast, WordNet Lemmatization uses morphological dictionary lookup and POS tagging to produce valid dictionary roots like 'terminate' and 'liability'."*

> **Q3: What was the main benefit of Shallow Chunking in Experiment 5?**  
> **Answer:** *"It separated noun phrases (NP) from verb phrases (VP). This allows the system to clearly identify the legal actor (e.g., 'The customer') and the contractual duty (e.g., 'must maintain'), which is essential for obligation extraction."*

> **Q4: What is the difference between Transition and Emission probabilities in an HMM?**  
> **Answer:** *"Transition probability is the chance of moving from one POS tag to another (e.g., how often does a Noun follow an Adjective). Emission probability is the chance that a specific POS tag produces a specific word (e.g., how often does the tag 'VERB' produce the word 'terminate')."*

> **Q5: What is the Out-of-Vocabulary (OOV) problem in HMMs, and how does BERT solve it?**  
> **Answer:** *"In an HMM, if an unseen word was never in the training data, its emission probability is 0, which collapses the entire Viterbi path. BERT solves this completely using WordPiece subword tokenization — breaking unseen words into known syllables (e.g., 'indemnification' into ['in', '##dem', '##ni', '##fication']). No word is ever unknown."*
