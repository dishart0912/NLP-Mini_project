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

def load_inference_model(model_dir=MODEL_DIR):
    """
    Loads model and tokenizer in evaluation mode.
    Fails gracefully with helpful instructions if artifacts are missing.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Error: Model directory '{model_dir}' does not exist. "
            "Please run 'python fine_tune_bert.py' first to train and save the model."
        )

    required_files = ["config.json"]
    for rf in required_files:
        if not os.path.exists(os.path.join(model_dir, rf)):
            raise FileNotFoundError(f"Missing required artifact '{rf}' in '{model_dir}'.")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def predict_clause_risk(clause_text, model=None, tokenizer=None, device=None):
    """
    Predicts clause risk category with per-class probability distribution.
    Gracefully handles empty strings, non-string types, and malformed inputs.
    
    NOTE ON CONFIDENCE SCORES:
    Softmax outputs are mathematical model probability distributions over 3 discrete
    classes, representing relative model confidence. They are NOT legal guarantees of
    liability or formally calibrated risk probabilities.
    """
    if not isinstance(clause_text, str) or not clause_text.strip():
        return {
            "text": clause_text if isinstance(clause_text, str) else "",
            "predicted_risk": "UNKNOWN",
            "confidence": 0.0,
            "probabilities": {ID2LABEL[i]: 0.0 for i in range(3)},
            "status": "empty_input",
            "warning": "Input clause is empty or whitespace-only."
        }

    if model is None or tokenizer is None or device is None:
        model, tokenizer, device = load_inference_model()

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
        "probabilities": {ID2LABEL[i]: float(probs[i]) for i in range(3)},
        "status": "success"
    }

if __name__ == "__main__":
    print("=" * 75)
    print("INFERENCE DEMONSTRATION ON CONTRACTUAL CLAUSES")
    print("=" * 75)

    try:
        model, tokenizer, device = load_inference_model()
        print(f"Model successfully loaded on {device}!\n")
    except Exception as e:
        print(f"[!] Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    test_clauses = [
        "Amazon reserves the right in its sole discretion to terminate any user account without warning or compensation.",
        "Please check our Privacy Notice to learn more about how cookies and personal identifiers are utilized.",
        "Users must maintain the confidentiality of their secret credentials and update passwords every 90 days.",
        "UNDER NO CIRCUMSTANCES SHALL THE VENDOR BE LIABLE FOR ANY CONSEQUENTIAL OR PUNITIVE DAMAGES ARISING FROM SERVICE INTERRUPTION.",
        "Any dispute arising under this Agreement shall be resolved exclusively through confidential binding arbitration."
    ]

    for i, c in enumerate(test_clauses, 1):
        res = predict_clause_risk(c, model, tokenizer, device)
        print(f"\n[Clause {i}] \"{res['text']}\"")
        print(f"  -> Predicted Risk : {res['predicted_risk'].upper()} (Confidence: {res['confidence']*100:.2f}%)")
        print(f"  -> Probabilities  : Low={res['probabilities']['Low']*100:.1f}%, "
              f"Medium={res['probabilities']['Medium']*100:.1f}%, "
              f"High={res['probabilities']['High']*100:.1f}%")

    print("\n[+] Edge-Case Verification: Testing empty string...")
    empty_res = predict_clause_risk("   ", model, tokenizer, device)
    print(f"    Empty status: {empty_res['status']} | Predicted: {empty_res['predicted_risk']}")

    print("\n" + "=" * 75)
