"""
BERT Fine-Tuning, Evaluation, and Baseline Comparison Script
Task: Legal Clause Risk Classification (Low, Medium, High)
Model: prajjwal1/bert-tiny (Pretrained BERT sequence classifier, ~4.4M parameters)
Dataset: tc_clauses_expanded.csv (82 real-world contractual clauses)

Pipeline:
1. Loads dataset and stratifies into Train (70%), Val (15%), Test (15%).
2. Computes class weights for balanced multi-class learning.
3. Evaluates classical baseline (TF-IDF + Logistic Regression).
4. Loads pretrained BERT-Tiny and fine-tunes with AdamW and Weighted Cross-Entropy Loss for 12 epochs.
5. Logs epoch-by-epoch loss convergence and validation accuracy.
6. Evaluates on held-out test set: Accuracy, Precision, Recall, Macro F1, Confusion Matrix.
7. Saves trained model checkpoint and tokenizer into ./model_output.
8. Generates sample predictions with confidence scores on unseen clauses.
"""

import os
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

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Dataset Loading
DATA_PATH = "tc_clauses_expanded.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset {DATA_PATH} not found. Please run prepare_expanded_dataset.py first.")

df = pd.read_csv(DATA_PATH)
print("=" * 70)
print(f"DATASET LOADED: {DATA_PATH} ({len(df)} contractual clauses)")
print("=" * 70)

LABEL2ID = {"Low": 0, "Medium": 1, "High": 2}
ID2LABEL = {0: "Low", 1: "Medium", 2: "High"}
df["label"] = df["risk_level"].map(LABEL2ID)

# 2. Stratified Splits
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"])

print(f"Splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Train counts: {train_df['risk_level'].value_counts().to_dict()}")
print(f"Test counts:  {test_df['risk_level'].value_counts().to_dict()}")

# Compute balanced class weights for loss function
weights = compute_class_weight("balanced", classes=np.unique(train_df["label"]), y=train_df["label"])
class_weights_tensor = torch.tensor(weights, dtype=torch.float)
print(f"Calculated Class Weights (Low, Med, High): {weights.round(3).tolist()}")
print("=" * 70)

# 3. Baseline Model (TF-IDF + Logistic Regression)
print("\n[STEP 1] TRAINING CLASSICAL BASELINE (TF-IDF + Logistic Regression)...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
X_train_tfidf = tfidf.fit_transform(train_df["clause_text"])
X_test_tfidf = tfidf.transform(test_df["clause_text"])

baseline_clf = LogisticRegression(class_weight="balanced", random_state=42)
baseline_clf.fit(X_train_tfidf, train_df["label"])
baseline_preds = baseline_clf.predict(X_test_tfidf)

baseline_acc = accuracy_score(test_df["label"], baseline_preds)
baseline_prec, baseline_rec, baseline_f1, _ = precision_recall_fscore_support(
    test_df["label"], baseline_preds, average="macro", zero_division=0
)
print(f"Baseline Test Accuracy: {baseline_acc * 100:.2f}% | Macro F1: {baseline_f1:.4f}")

# 4. Dataset Loader
MODEL_NAME = "prajjwal1/bert-tiny"
print(f"\n[STEP 2] INITIALIZING PRETRAINED MODEL: {MODEL_NAME}")
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

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 5. Model Loading
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

# 6. Fine-Tuning Execution
LEARNING_RATE = 1e-4
EPOCHS = 12
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

print("\n" + "=" * 70)
print(f"STARTING FINE-TUNING: {EPOCHS} Epochs, LR={LEARNING_RATE}, Batch Size=4, Weighted Cross-Entropy")
print("=" * 70)

training_history = []

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

    # Validation
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

    training_history.append({
        "epoch": epoch,
        "train_loss": round(avg_train_loss, 4),
        "val_loss": round(avg_val_loss, 4),
        "val_accuracy": round(val_acc * 100, 2)
    })
    print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

print("=" * 70)
print("FINE-TUNING COMPLETE. WEIGHT UPDATES PERSISTED.")
print("=" * 70)

# 7. Model Evaluation on Held-Out Test Set
print("\n[STEP 3] EVALUATING ON HELD-OUT TEST SET...")
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
conf_mat = confusion_matrix(test_targets, test_preds)
target_names = ["Low", "Medium", "High"]
class_rep = classification_report(test_targets, test_preds, target_names=target_names, zero_division=0)

print(f"\nFinal Fine-Tuned Model Performance:")
print(f"Accuracy:      {test_acc * 100:.2f}%")
print(f"Macro Prec:    {test_prec:.4f}")
print(f"Macro Recall:  {test_rec:.4f}")
print(f"Macro F1:      {test_f1:.4f}")
print("\nClassification Report:\n", class_rep)
print("Confusion Matrix (Rows=True, Cols=Pred [Low, Medium, High]):")
print(conf_mat)

# 8. Save Fine-Tuned Model and Tokenizer
SAVE_DIR = "model_output"
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel checkpoint and tokenizer saved successfully to '{SAVE_DIR}/'.")

# Save evaluation results to JSON
results = {
    "model_name": MODEL_NAME,
    "architecture": "BertForSequenceClassification (Tiny)",
    "total_parameters": 4386307,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": 4,
    "loss_function": "Weighted Cross-Entropy",
    "class_weights": weights.round(3).tolist(),
    "training_history": training_history,
    "baseline_metrics": {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(baseline_acc * 100, 2),
        "macro_precision": round(baseline_prec, 4),
        "macro_recall": round(baseline_rec, 4),
        "macro_f1": round(baseline_f1, 4)
    },
    "fine_tuned_metrics": {
        "model": "Fine-Tuned BERT-Tiny",
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
print("\n" + "=" * 70)
print("[STEP 4] RUNNING SAMPLE INFERENCE ON UNSEEN CONTRACT CLAUSES")
print("=" * 70)

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

print("\n" + "=" * 70)
print("ALL TASKS COMPLETED SUCCESSFULLY. METRICS & PREDICTIONS PERSISTED.")
print("=" * 70)
