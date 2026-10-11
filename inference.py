"""
Standalone Inference Script for Legal Clause Risk Analysis
Uses the fine-tuned BERT model saved in ./model_output.
Allows command-line input or batch evaluation.
"""

import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

MODEL_DIR = "model_output"
ID2LABEL = {0: "Low", 1: "Medium", 2: "High"}

if not os.path.exists(MODEL_DIR):
    print(f"Error: Directory '{MODEL_DIR}' does not exist. Please run fine_tune_bert.py first.")
    sys.exit(1)

print(f"Loading fine-tuned model and tokenizer from '{MODEL_DIR}'...")
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model successfully loaded on {device}!\n")

def predict_clause_risk(clause_text):
    inputs = tokenizer(
        clause_text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    return {
        "text": clause_text,
        "predicted_risk": ID2LABEL[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {ID2LABEL[i]: float(probs[i]) for i in range(3)}
    }

if __name__ == "__main__":
    test_clauses = [
        "Amazon reserves the right in its sole discretion to terminate any user account without warning or compensation.",
        "Please check our Privacy Notice to learn more about how cookies and personal identifiers are utilized.",
        "Users must maintain the confidentiality of their secret credentials and update passwords every 90 days.",
        "UNDER NO CIRCUMSTANCES SHALL THE VENDOR BE LIABLE FOR ANY CONSEQUENTIAL OR PUNITIVE DAMAGES ARISING FROM SERVICE INTERRUPTION.",
        "Any dispute arising under this Agreement shall be resolved exclusively through confidential binding arbitration."
    ]

    print("=" * 75)
    print("INFERENCE DEMONSTRATION ON CONTRACTUAL CLAUSES")
    print("=" * 75)

    for i, c in enumerate(test_clauses, 1):
        res = predict_clause_risk(c)
        print(f"\n[Clause {i}] \"{res['text']}\"")
        print(f"  -> Predicted Risk : {res['predicted_risk'].upper()} (Confidence: {res['confidence']*100:.2f}%)")
        print(f"  -> Probabilities  : Low={res['probabilities']['Low']*100:.1f}%, "
              f"Medium={res['probabilities']['Medium']*100:.1f}%, "
              f"High={res['probabilities']['High']*100:.1f}%")

    print("\n" + "=" * 75)
