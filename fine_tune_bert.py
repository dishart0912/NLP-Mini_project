"""
BERT Fine-Tuning, Validation Checkpointing, and Baseline Benchmarking Module
Task: 3-Class Legal Clause Risk Classification (Low, Medium, High)
Model: prajjwal1/bert-tiny (Pretrained BERT sequence classification transformer)
Dataset: tc_clauses_expanded.csv (82 verified legal clauses across 11 platforms)

Engineering Highlights:
1. Strict seed setting across random, numpy, and torch.
2. Stratified train (70%), validation (15%), and held-out test (15%) splits.
3. Baseline evaluation: TF-IDF (1-2 grams) + Logistic Regression with balanced class weights.
4. Pretrained BERT-Tiny initialization with task-specific classification head.
5. Weighted Cross-Entropy Loss to penalize minority class errors.
6. Validation monitoring with BEST-CHECKPOINT SELECTION based on lowest validation loss.
7. Final evaluation on held-out test set using the reloaded best checkpoint.
8. Serialization of best model, tokenizer, training metadata, and evaluation results.
"""

import os
import sys
import copy
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

# 1. Dataset Loading & Schema Validation
DATA_PATH = "tc_clauses_expanded.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset file '{DATA_PATH}' not found. Please run 'python prepare_expanded_dataset.py' first."
    )

df = pd.read_csv(DATA_PATH)
TOTAL_CLAUSES = len(df)
print("=" * 75)
print(f"DATASET LOADED: '{DATA_PATH}' ({TOTAL_CLAUSES} contractual clauses)")
print("=" * 75)

LABEL2ID = {"Low": 0, "Medium": 1, "High": 2}
ID2LABEL = {0: "Low", 1: "Medium", 2: "High"}
CLASS_NAMES = ["Low", "Medium", "High"]

df["label"] = df["risk_level"].map(LABEL2ID)

# 2. Stratified Train / Validation / Test Splits (70% Train, 15% Val, 15% Test)
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["label"])

TRAIN_COUNT = len(train_df)
VAL_COUNT = len(val_df)
TEST_COUNT = len(test_df)

print(f"Dataset Split Sizes -> Train: {TRAIN_COUNT} (70%), Val: {VAL_COUNT} (15%), Test: {TEST_COUNT} (15%)")
print(f"Train Class Distribution: {train_df['risk_level'].value_counts().to_dict()}")
print(f"Val Class Distribution:   {val_df['risk_level'].value_counts().to_dict()}")
print(f"Test Class Distribution:  {test_df['risk_level'].value_counts().to_dict()}")

# Compute balanced class weights strictly on the training set
weights = compute_class_weight("balanced", classes=np.unique(train_df["label"]), y=train_df["label"])
class_weights_tensor = torch.tensor(weights, dtype=torch.float)
print(f"Computed Training Class Weights (Low, Med, High): {weights.round(3).tolist()}")
print("=" * 75)

# 3. Classical Baseline Model (TF-IDF + Logistic Regression)
print("\n[STEP 1] TRAINING & EVALUATING CLASSICAL BASELINE (TF-IDF + Logistic Regression)...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
X_train_tfidf = tfidf.fit_transform(train_df["clause_text"])
X_test_tfidf = tfidf.transform(test_df["clause_text"])

baseline_clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
baseline_clf.fit(X_train_tfidf, train_df["label"])
baseline_preds = baseline_clf.predict(X_test_tfidf)

baseline_acc = accuracy_score(test_df["label"], baseline_preds)
baseline_prec, baseline_rec, baseline_f1, _ = precision_recall_fscore_support(
    test_df["label"], baseline_preds, average="macro", zero_division=0
)
baseline_conf_mat = confusion_matrix(test_df["label"], baseline_preds, labels=[0, 1, 2])

print(f"Baseline Test Accuracy: {baseline_acc * 100:.2f}%")
print(f"Baseline Macro Precision: {baseline_prec:.4f} | Recall: {baseline_rec:.4f} | F1: {baseline_f1:.4f}")
print("Baseline Confusion Matrix:\n", baseline_conf_mat)

# 4. Tokenizer & Dataset Loader Setup
MODEL_NAME = "prajjwal1/bert-tiny"
print(f"\n[STEP 2] INITIALIZING PRETRAINED TRANSFORMER BACKBONE: {MODEL_NAME}")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class LegalClauseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = LegalClauseDataset(train_df["clause_text"], train_df["label"], tokenizer)
val_dataset = LegalClauseDataset(val_df["clause_text"], val_df["label"], tokenizer)
test_dataset = LegalClauseDataset(test_df["clause_text"], test_df["label"], tokenizer)

BATCH_SIZE = 4
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Model Initialization
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
class_weights_tensor = class_weights_tensor.to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

print(f"Model Architecture: BertForSequenceClassification (device: {device})")
print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 6. Fine-Tuning with Best-Checkpoint Selection
LEARNING_RATE = 1e-4
EPOCHS = 12
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print("\n" + "=" * 75)
print(f"STARTING FINE-TUNING: {EPOCHS} Epochs, LR={LEARNING_RATE}, Batch Size={BATCH_SIZE}, Optimizer=AdamW")
print("Validation Checkpoint Metric: Lowest Validation Loss")
print("=" * 75)

training_history = []
best_val_loss = float("inf")
best_epoch = -1
best_model_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            val_loss = loss_fn(outputs.logits, labels)
            total_val_loss += val_loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_acc = accuracy_score(val_targets, val_preds)

    is_best = avg_val_loss < best_val_loss
    if is_best:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())

    marker = " [BEST CHECKPOINT]" if is_best else ""
    training_history.append({
        "epoch": epoch,
        "train_loss": round(avg_train_loss, 4),
        "val_loss": round(avg_val_loss, 4),
        "val_accuracy": round(val_acc * 100, 2),
        "is_best": is_best
    })
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%{marker}")

print("=" * 75)
print(f"TRAINING COMPLETE. Best validation loss achieved at Epoch {best_epoch:02d} (Loss: {best_val_loss:.4f}).")
print("Reloading best checkpoint weights for final test set evaluation...")
model.load_state_dict(best_model_state)
print("=" * 75)

# 7. Model Evaluation on Held-Out Test Set (Using Reloaded Best Checkpoint)
print("\n[STEP 3] EVALUATING BEST CHECKPOINT ON HELD-OUT TEST SET...")
model.eval()
test_preds, test_targets, test_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        test_preds.extend(preds)
        test_targets.extend(labels.cpu().numpy())
        test_probs.extend(probs)

test_acc = accuracy_score(test_targets, test_preds)
test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
    test_targets, test_preds, average="macro", zero_division=0
)
conf_mat = confusion_matrix(test_targets, test_preds, labels=[0, 1, 2])
class_rep = classification_report(test_targets, test_preds, target_names=CLASS_NAMES, labels=[0, 1, 2], zero_division=0)

print(f"\nFinal Best-Checkpoint Performance on Held-Out Test Set ({TEST_COUNT} clauses):")
print(f"Accuracy:      {test_acc * 100:.2f}%")
print(f"Macro Prec:    {test_prec:.4f}")
print(f"Macro Recall:  {test_rec:.4f}")
print(f"Macro F1:      {test_f1:.4f}")
print("\nClassification Report:\n", class_rep)
print("Confusion Matrix (Rows=True, Cols=Pred [Low, Medium, High]):")
print(conf_mat)

# 8. Save Best Fine-Tuned Model Checkpoint and Tokenizer
SAVE_DIR = "model_output"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nBest model checkpoint and tokenizer saved successfully to '{SAVE_DIR}/'.")

# Save structured evaluation results and metadata to JSON
results = {
    "model_name": MODEL_NAME,
    "architecture": "BertForSequenceClassification (Tiny)",
    "total_parameters": 4386307,
    "random_seed": RANDOM_SEED,
    "dataset_total_clauses": TOTAL_CLAUSES,
    "train_clauses": TRAIN_COUNT,
    "val_clauses": VAL_COUNT,
    "test_clauses": TEST_COUNT,
    "epochs_trained": EPOCHS,
    "best_epoch": best_epoch,
    "best_val_loss": round(best_val_loss, 4),
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "loss_function": "Weighted Cross-Entropy",
    "class_weights": weights.round(3).tolist(),
    "training_history": training_history,
    "baseline_metrics": {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(baseline_acc * 100, 2),
        "macro_precision": round(baseline_prec, 4),
        "macro_recall": round(baseline_rec, 4),
        "macro_f1": round(baseline_f1, 4),
        "confusion_matrix": baseline_conf_mat.tolist()
    },
    "fine_tuned_metrics": {
        "model": "Fine-Tuned BERT-Tiny (Best Checkpoint)",
        "accuracy": round(test_acc * 100, 2),
        "macro_precision": round(test_prec, 4),
        "macro_recall": round(test_rec, 4),
        "macro_f1": round(test_f1, 4),
        "confusion_matrix": conf_mat.tolist()
    }
}

with open(os.path.join(SAVE_DIR, "evaluation_metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

# 9. Real-World Sample Inference
print("\n" + "=" * 75)
print("[STEP 4] RUNNING SAMPLE INFERENCE ON UNSEEN CONTRACT CLAUSES")
print("=" * 75)

unseen_clauses = [
    "Amazon reserves the right in its sole discretion to terminate any user account without warning or compensation.",
    "Please check our Privacy Notice to learn more about how cookies and personal identifiers are utilized.",
    "Users must maintain the confidentiality of their secret credentials and update passwords every 90 days.",
    "UNDER NO CIRCUMSTANCES SHALL THE VENDOR BE LIABLE FOR ANY CONSEQUENTIAL OR PUNITIVE DAMAGES ARISING FROM SERVICE INTERRUPTION.",
    "Any dispute arising under this Agreement shall be resolved exclusively through confidential binding arbitration."
]

inference_results = []
for clause in unseen_clauses:
    inputs = tokenizer(clause, truncation=True, padding="max_length", max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_idx]
        conf = float(probs[pred_idx])

    inference_results.append({
        "clause": clause,
        "predicted_risk": pred_label,
        "confidence": round(conf * 100, 2),
        "probabilities": {ID2LABEL[i]: round(float(probs[i]) * 100, 2) for i in range(3)}
    })
    print(f"\nClause: \"{clause[:80]}...\"")
    print(f" -> Predicted Risk: {pred_label} (Confidence: {conf * 100:.2f}%)")
    print(f"    Distribution: Low={probs[0]*100:.1f}%, Med={probs[1]*100:.1f}%, High={probs[2]*100:.1f}%")

with open(os.path.join(SAVE_DIR, "sample_predictions.json"), "w") as f:
    json.dump(inference_results, f, indent=2)

print("\n" + "=" * 75)
print("ALL PIPELINE TASKS COMPLETED SUCCESSFULLY. METRICS & CHECKPOINT PERSISTED.")
print("=" * 75)
