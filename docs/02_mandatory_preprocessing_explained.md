# 2. Mandatory Preprocessing Explained

The instructor's guidelines require **5 specific preprocessing steps**. Here is what each step does, why it matters for legal text, and how our code implements it.

All of these are runnable in: [`preprocessing_pipeline.py`](../preprocessing_pipeline.py).

> [!IMPORTANT]
> **Two Separate Pipelines — Don't Confuse Them!**  
> The 5 NLTK preprocessing steps below are a standalone demo module — they show linguistic analysis. The BERT classification model does **NOT** use these cleaned tokens as input. Instead, BERT uses its own built-in tokenizer (`BertTokenizer`) on the raw clause text. This is because BERT's tokenizer is smarter — it preserves punctuation and subtle word forms that a general lemmatizer would remove.

---

## 1. Sentence Boundary Identification
### What is it?
Splitting a long block of paragraph text into separate individual sentences.

### Why is it tricky in Legal Text?
Legal contracts have abbreviations (like *e.g.*, *i.e.*, *LLC.*, *Sec.*) and bullet points. If you just split on every period (`.`), you might split *Amazon.com LLC.* right in the middle — which breaks the sentence wrongly!

### In Our Code:
We use NLTK's `sent_tokenize` — think of it as a smart full-stop detector:
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
Breaking a sentence into individual pieces called **tokens** (words, numbers, and punctuation marks).

### Why does it matter?
Computers cannot process raw sentences directly — they need isolated words. In legal documents, keeping punctuation and symbols (like `$1,000` or `100%`) matters a lot because money values determine how much a company is liable!

### In Our Code:
We use word-level tokenization (`nltk.word_tokenize`) and BERT's subword tokenization (`BertTokenizer`):
```python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(sentence)
```
**Example:**
> *"Amazon will not be liable."* $\rightarrow$ `['Amazon', 'will', 'not', 'be', 'liable', '.']`

---

## 3. Part-of-Speech (POS) Tagging
### What is it?
Labeling each word with its grammar role (e.g., Noun, Verb, Adjective, Modal Auxiliary).
- Think of it like labeling every word in a sentence as a "doing word", "naming word", "describing word", etc.

### Why does it matter in Contracts?
In legal contracts, the most dangerous words are **Modal Verbs** (POS tag `MD`) such as:
- **`shall` / `must`** $\rightarrow$ A non-negotiable legal **obligation** on the user. You HAVE to do this.
- **`may`** $\rightarrow$ An optional **right** the company is claiming for itself. They CAN do this if they want.

### In Our Code:
We use NLTK's Perceptron tagger (and we also built a from-scratch Hidden Markov Model tagger in our HMM PDF):
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
Reducing different forms of a word back to its dictionary root (called the **lemma**).
- For example: *terminating*, *terminates*, and *terminated* all reduce to the lemma **`terminate`**.
- Analogy: Lemmatization is like looking up a word in a dictionary to find its base form, instead of just chopping off the ending.

### Stemming vs. Lemmatization (Crucial Viva Point!):
- **Stemming** roughly chops off word endings using simple rules. For example, Porter Stemmer turns *indemnification* into `indemnif` — which is NOT even a real word!
- **Lemmatization** uses a vocabulary and grammar knowledge to return real dictionary words. For example, *obligations* becomes `obligation` — a real word.

### In Our Code:
We tell the lemmatizer what grammar role (POS) the word plays, so it makes smarter decisions:
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemma = lemmatizer.lemmatize("terminating", pos='v')  # Returns 'terminate'
```

---

## 5. WordNet Semantic Analysis
### What is it?
**WordNet** is a giant English dictionary organized by meaning — not just spelling. Words are grouped into **synsets** (sets of words that share the same meaning/concept).

### Why did the guidelines require WordNet?
Simple word matching is not enough to understand legal text:
- If a contract says *"vendor disclaims all indemnification and liability"*, WordNet connects *"liability"* to broader, related concepts (**hypernyms** — think "parent" categories) like *indebtedness* and *legal obligation*.
- If a contract says *"we may discontinue or suspend services"*, WordNet connects *"suspend"* to more specific words (**hyponyms** — think "child" categories) like *abort*, *interrupt*, or *terminate*.
- Analogy: WordNet is like a thesaurus that also knows which words are more general and which are more specific.

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
