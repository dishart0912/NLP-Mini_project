# 4. LLM Fine-Tuning Deep Dive

This guide explains the fine-tuning process implemented in [`fine_tune_bert.py`](../fine_tune_bert.py) in simple, step-by-step terms.

---

## 1. What Does "Fine-Tuning" Actually Mean?

Imagine a university student who has read the entire English dictionary and Wikipedia:
- The student understands English grammar, sentence structures, and general vocabulary. This is **Pre-training**.
- But the student has never seen a specialized contract clause.
- When you put the student through a specialized 2-week course on contract law with examples of dangerous clauses, their brain rewires slightly to focus on legal liability. This is **Fine-Tuning**.

In neural networks:
- A pretrained model already knows English word embeddings.
- During fine-tuning, we feed our domain-specific legal training data through the model.
- We calculate the error (loss) using a loss function.
- We send that error backwards through the neural network (**Backpropagation**) and **update the internal numerical weights** using an optimizer (**AdamW**).

---

## 2. Which Model Did We Choose and Why?

We selected **`prajjwal1/bert-tiny`** from Hugging Face:
- **Architecture:** `BertForSequenceClassification`
- **Number of Transformer Layers:** 2 layers
- **Hidden Embedding Dimension:** 128
- **Attention Heads:** 2
- **Total Parameters:** **4,386,307** (~4.4 Million weights)

### Why BERT-Tiny instead of BERT-Large (340M) or Llama (8 Billion)?
1. **Avoiding Catastrophic Overfitting:** When your domain dataset has ~80–100 clauses, attempting to fine-tune a massive 100M+ or 7B parameter model causes the model to simply memorize the exact sentences ("overfitting"), destroying its ability to generalize.
2. **Computational Feasibility:** BERT-Tiny runs easily and stably on CPU in under 30 seconds without needing an expensive GPU.
3. **True Transformer Mechanics:** It uses the exact same Multi-Head Self-Attention, Layer Normalization, and Feed-Forward mechanisms as BERT-Base.

---

## 3. The Fine-Tuning Configuration & Hyperparameters

| Hyperparameter | Value Used | Why this value? |
|---|---|---|
| **Epochs** | `12` | Going through the dataset 12 times gives the model enough iterations to adjust weights smoothly. |
| **Batch Size** | `4` | Feeding 4 clauses at a time ensures frequent gradient updates. |
| **Learning Rate** | `1e-4` (`0.0001`) | Small enough to avoid destroying pretrained knowledge, but large enough to adapt to legal terms. |
| **Optimizer** | `AdamW` (`weight_decay=0.01`) | Standard transformer optimizer that adapts step sizes per parameter while preventing overfitting via L2 regularization. |
| **Loss Function** | `Weighted Cross-Entropy` | Compensates for class imbalance so the model doesn't just guess the majority class. |
| **Max Sequence Length** | `128` | Sufficient to capture 95% of standard legal clauses without truncation. |

---

## 4. Why We Used "Weighted Cross-Entropy Loss"

In our training set, we have 28 High-risk clauses, 16 Medium-risk clauses, and 13 Low-risk clauses.
If we used standard loss, a lazy model might just predict "High" for every clause to get an easy 50% score!

To stop this, we calculate **Class Weights**:
$$w_{\text{class}} = \frac{N_{\text{total}}}{N_{\text{classes}} \times N_{\text{samples in class}}}$$
- **Low Risk Weight:** `1.462` (High penalty if the model misclassifies a Low clause)
- **Medium Risk Weight:** `1.188`
- **High Risk Weight:** `0.679`

This forces the optimizer to give equal attention to all 3 categories!

---

## 5. Evidence of Actual Parameter Updates (Training Logs)

During execution of `fine_tune_bert.py`, the training loss and validation accuracy changed across the 12 epochs:

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
1. **Loss Reduced by 57.6%:** Train Loss went from `1.1194` down to `0.4742`.
2. **Validation Accuracy Jumped:** Rose from `41.67%` to `75.00%`.
3. **Best-Checkpoint Selection:** The script monitors validation loss after every epoch. The checkpoint with the lowest validation loss (Epoch 12, loss: 0.6780) was automatically reloaded before test set evaluation to prevent overfitting.
4. **Saved Weights:** PyTorch saved the new neural network parameters directly into [`model_output/model.safetensors`](../model_output).

---

## 6. How the Model Files Are Saved

When training completes, the script creates the `model_output/` folder containing:
- `model.safetensors`: The trained best-checkpoint neural network weights.
- `config.json`: The model architecture settings and label mappings (`0: Low, 1: Medium, 2: High`).
- `tokenizer.json` / `tokenizer_config.json`: The tokenizer vocabulary and rules.
- `evaluation_metrics.json`: All evaluation numbers, confusion matrix, and training metadata in JSON format.
- `sample_predictions.json`: Unseen clause predictions with confidence scores.
