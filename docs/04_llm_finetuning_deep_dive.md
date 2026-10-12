# 4. LLM Fine-Tuning Deep Dive

This guide explains the fine-tuning process implemented in [`fine_tune_bert.py`](../fine_tune_bert.py) in simple, step-by-step terms.

---

## 1. What Does "Fine-Tuning" Actually Mean?

Think of it like this:
- Imagine a student who has read the entire English dictionary and Wikipedia. They understand English grammar, vocabulary, and sentence structure. This is **Pre-training** — the model already knows general English.
- But this student has never read a legal contract before.
- Now you put them through a 2-week crash course on contract law, with real examples of dangerous clauses. Their understanding adjusts to focus on legal language. This is **Fine-Tuning** — the model adapts to a specific task.

In neural networks (programs loosely inspired by the human brain):
- A pretrained model already understands English word meanings (called **embeddings** — numbers that represent word meanings).
- During fine-tuning, we feed our legal training data through the model.
- The model makes a prediction. We measure how wrong it is using a **loss function** (a score that tells us how bad the guess was).
- We send that error backwards through the network — this is called **Backpropagation** — and update the internal numbers (**weights**) using an optimizer called **AdamW**.
- After many rounds, the model gets better at recognizing risky legal clauses.

---

## 2. Which Model Did We Choose and Why?

We selected **`prajjwal1/bert-tiny`** from Hugging Face (a popular website for sharing AI models):
- **Architecture:** `BertForSequenceClassification` — BERT (Bidirectional Encoder Representations from Transformers) set up to classify text into categories.
- **Number of Transformer Layers:** 2 layers
- **Hidden Embedding Dimension:** 128 (each word is represented as 128 numbers)
- **Attention Heads:** 2 (the model looks at the sentence from 2 different "perspectives" at once)
- **Total Parameters:** **4,386,307** (~4.4 Million weights — numbers that get updated during training)

### Why BERT-Tiny instead of BERT-Large (340M parameters) or Llama (8 Billion parameters)?
1. **Avoiding Catastrophic Overfitting:** When your dataset has only ~80–100 clauses, a giant 100M+ parameter model will just **memorize** the training sentences instead of learning general patterns. This is called overfitting — like a student who memorizes answers without understanding them. BERT-Tiny is small enough to actually learn.
2. **Computational Feasibility:** BERT-Tiny runs in under 30 seconds on a regular CPU. No expensive GPU needed.
3. **Same Core Mechanics:** BERT-Tiny uses the exact same Multi-Head Self-Attention (a way for the model to focus on different words at the same time), Layer Normalization, and Feed-Forward layers as the big BERT models.

---

## 3. The Fine-Tuning Configuration & Hyperparameters

Hyperparameters are the settings we choose before training starts — like setting the oven temperature before baking.

| Hyperparameter | Value Used | Why this value? |
|---|---|---|
| **Epochs** | `12` | Going through the dataset 12 times gives the model enough chances to adjust its weights smoothly. |
| **Batch Size** | `4` | Feed 4 clauses at a time. Small batches mean frequent weight updates, which helps on small datasets. |
| **Learning Rate** | `1e-4` (`0.0001`) | Small enough to not destroy what the model already knows, but large enough to adapt to legal language. |
| **Optimizer** | `AdamW` (`weight_decay=0.01`) | AdamW (Adam with Decoupled Weight Decay) is the standard optimizer for transformer models. It adjusts step sizes smartly per parameter and prevents overfitting via L2 regularization (a penalty that stops weights from growing too large). |
| **Loss Function** | `Weighted Cross-Entropy` | Handles the imbalance in our dataset — more on this below. |
| **Max Sequence Length** | `128` | Long enough to fit 95% of legal clauses without cutting them short. |

---

## 4. Why We Used "Weighted Cross-Entropy Loss"

In our training set, we have 28 High-risk clauses, 16 Medium-risk clauses, and 13 Low-risk clauses.  
Without any correction, a lazy model might just always predict "High Risk" to get an easy ~50% accuracy! That's useless.

To fix this, we calculate **Class Weights** — classes with fewer examples get a bigger penalty when the model gets them wrong:

$$w_{\text{class}} = \frac{N_{\text{total}}}{N_{\text{classes}} \times N_{\text{samples in class}}}$$

*Plain English: Rarer classes get a higher weight (more importance). This forces the model to take all 3 classes seriously — not just the most common one.*

- **Low Risk Weight:** `1.462` (Low has the fewest samples, so getting it wrong is penalized more)
- **Medium Risk Weight:** `1.188`
- **High Risk Weight:** `0.679` (High has the most samples, so it gets a lower penalty)

This forces the model to pay equal attention to all 3 categories during training.

---

## 5. Evidence of Actual Parameter Updates (Training Logs)

During `fine_tune_bert.py`, the training loss and validation accuracy changed across 12 epochs. Loss is like a mistake score — lower is better. Accuracy is how often the model is correct:

```text
Epoch 01/12 | Train Loss: 1.1194 | Val Loss: 1.0915 | Val Acc: 41.67%
Epoch 02/12 | Train Loss: 1.0793 | Val Loss: 1.0783 | Val Acc: 50.00%
Epoch 03/12 | Train Loss: 1.0314 | Val Loss: 1.0291 | Val Acc: 58.33%
Epoch 04/12 | Train Loss: 1.0058 | Val Loss: 1.0025 | Val Acc: 50.00%
Epoch 05/12 | Train Loss: 0.9524 | Val Loss: 0.9570 | Val Acc: 66.67%
Epoch 06/12 | Train Loss: 0.8922 | Val Loss: 0.9029 | Val Acc: 75.00%
Epoch 07/12 | Train Loss: 0.8283 | Val Loss: 0.8967 | Val Acc: 66.67%
Epoch 08/12 | Train Loss: 0.7449 | Val Loss: 0.8285 | Val Acc: 66.67%
Epoch 09/12 | Train Loss: 0.6832 | Val Loss: 0.8040 | Val Acc: 66.67%
Epoch 10/12 | Train Loss: 0.6181 | Val Loss: 0.7674 | Val Acc: 75.00%
Epoch 11/12 | Train Loss: 0.5110 | Val Loss: 0.7153 | Val Acc: 75.00%
Epoch 12/12 | Train Loss: 0.4742 | Val Loss: 0.6780 | Val Acc: 75.00%
```

### What proves that training really happened?
1. **Loss Reduced by 57.6%:** Train Loss went from `1.1194` down to `0.4742`. The model's mistakes got smaller and smaller — it was learning.
2. **Validation Accuracy Jumped:** Rose from `41.67%` to `75.00%`. The model got much better at predicting on data it hadn't trained on.
3. **Best-Checkpoint Selection:** After each epoch, the script saves the version with the lowest validation loss. The best version (Epoch 12, loss: 0.6780) is automatically reloaded before final evaluation — this prevents overfitting.
4. **Saved Weights:** PyTorch saved the updated neural network parameters to [`model_output/model.safetensors`](../model_output) — proof that real training happened.

---

## 6. How the Model Files Are Saved

When training finishes, the script creates the `model_output/` folder with these files:
- `model.safetensors`: The trained neural network weights (the actual "brain" of the model after learning).
- `config.json`: The model structure settings and label mappings (`0: Low, 1: Medium, 2: High`).
- `tokenizer.json` / `tokenizer_config.json`: The tokenizer vocabulary and rules (how words are converted to numbers).
- `evaluation_metrics.json`: All evaluation numbers, confusion matrix, and training details in JSON format.
- `sample_predictions.json`: Predictions on new unseen clauses, with confidence scores.
