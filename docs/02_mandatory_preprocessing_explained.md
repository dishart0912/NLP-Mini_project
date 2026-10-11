# 2. Mandatory Preprocessing Explained

The instructor's guidelines explicitly require **5 specific preprocessing steps**. Here is what each step does, why it matters for legal text, and how our code implements it.

All of these are runnable in: [`preprocessing_pipeline.py`](file:///e:/tsec/sem7/lab/nlp/preprocessing_pipeline.py).

---

## 1. Sentence Boundary Identification
### What is it?
Splitting a long, unbroken block of paragraph text into separate individual sentences.

### Why is it tricky in Legal Text?
Legal contracts have titles, abbreviations (like *e.g.*, *i.e.*, *LLC.*, *Sec.*), and bullet points. If you just split on periods (`.`), you might split *Amazon.com LLC.* right in the middle!

### In Our Code:
We use NLTK's `sent_tokenize`:
```python
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(raw_text)
```
**Example:**
> **Input:** *"Amazon may terminate accounts. You agree to arbitrate."*  
> **Output:**  
> `Clause 1:` *"Amazon may terminate accounts."*  
> `Clause 2:` *"You agree to arbitrate."*

---

## 2. Tokenization
### What is it?
Breaking a sentence into individual units called **tokens** (words, numbers, and punctuation).

### Why does it matter?
Computers cannot process raw sentences directly; they need isolated words or subwords. In legal documents, preserving punctuation and symbols (like `$1,000` or `100%`) is critical because monetary caps determine liability limits!

### In Our Code:
We use both word-level tokenization (`nltk.word_tokenize`) and subword WordPiece tokenization (`BertTokenizer`):
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(sentence)
```
**Example:**
> *"Amazon will not be liable."* $\rightarrow$ `['Amazon', 'will', 'not', 'be', 'liable', '.']`

---

## 3. Part-of-Speech (POS) Tagging
### What is it?
Labeling each token with its grammatical category (e.g., Noun, Verb, Adjective, Modal Auxilary).

### Why does it matter in Contracts?
In legal contracts, the most dangerous or important words are **Modal Verbs** (POS tag `MD`) such as:
- **`shall` / `must`** $\rightarrow$ Represents a non-negotiable legal **obligation** on the user.
- **`may`** $\rightarrow$ Represents an optional **right** or discretionary power claimed by the company.

### In Our Code:
We use NLTK's Perceptron tagger (and we also implemented a scratch Hidden Markov Model in our HMM PDF):
```python
from nltk import pos_tag
tagged_tokens = pos_tag(tokens)
```
**Sample Output:**
```python
[('Amazon', 'NNP'), ('will', 'MD'), ('not', 'RB'), ('be', 'VB'), ('liable', 'JJ')]
```
*(Notice: `will` is tagged as `MD` (Modal), `not` is `RB` (Adverb), `liable` is `JJ` (Adjective)).*

---

## 4. Lemmatization (POS-Informed)
### What is it?
Reducing different grammatical forms of a word back to its dictionary root (lemma).
- For example: *terminating*, *terminates*, and *terminated* all reduce to the lemma **`terminate`**.

### Stemming vs. Lemmatization (Crucial Viva Point!):
- **Stemming** chops off the ends of words using rough rules (e.g., Porter Stemmer turns *indemnification* into `indemnif`, which is not even a real word!).
- **Lemmatization** uses vocabulary and morphological analysis to return real dictionary words (e.g., turns *indemnification* into `indemnification` and *obligations* into `obligation`).

### In Our Code:
We map the POS tag so the lemmatizer knows whether the word is a noun, verb, or adjective:
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemma = lemmatizer.lemmatize("terminating", pos='v')  # Returns 'terminate'
```

---

## 5. WordNet Semantic Analysis
### What is it?
**WordNet** is a large lexical database of English nouns, verbs, adjectives, and adverbs organized into sets of cognitive synonyms called **synsets**.

### Why did the guidelines require WordNet?
Because simple keyword matching misses the deeper meaning:
- If a contract says *"vendor disclaims all indemnification and liability"*, WordNet connects *"liability"* to broader concepts (**hypernyms**) like *indebtedness* and *legal obligation*.
- If a contract says *"we may discontinue or suspend services"*, WordNet connects *"suspend"* to narrower concepts (**hyponyms**) like *abort*, *interrupt*, or *terminate*.

### In Our Code:
```python
from nltk.corpus import wordnet as wn

syns = wn.synsets("liability")
primary_syn = syns[0]
print(primary_syn.definition())  # "the state of being legally obliged and responsible"
print(primary_syn.hypernyms())   # ['susceptibility.n.01', 'indebtedness.n.01']
```

---

## Summary of How They Connect

```text
Raw Paragraph
     │
     ▼ [1. Sentence Boundary]
Discrete Sentences
     │
     ▼ [2. Tokenization]
Tokens: ['shall', 'terminate', 'accounts']
     │
     ├───► [3. POS Tagging] ───► ('shall', MD), ('terminate', VB)
     │
     ├───► [4. Lemmatization] ─► 'terminate' (dictionary root)
     │
     └───► [5. WordNet] ──────► Definition: "bring to an end", Hyponyms: ['abort', 'cancel']
```
