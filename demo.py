"""
Interactive Live Demo Script for Terms & Conditions Risk Analyzer
Windows CP1252 / UTF-8 resilient version.
Allows the user/examiner to:
1. Choose from pre-loaded realistic contract clauses (Amazon, Alibaba, Twitter, etc.)
2. Type or paste any custom contract clause live
3. View instant risk analysis with confidence levels and probability bars
"""

import os
import sys
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

MODEL_DIR = "model_output"
ID2LABEL = {0: "Low", 1: "Medium", 2: "High"}

if not os.path.exists(MODEL_DIR):
    print(f"\n[!] Error: Model directory '{MODEL_DIR}' not found. Please run 'python fine_tune_bert.py' first.")
    sys.exit(1)

print("\n" + "=" * 75)
print("  LOADING FINE-TUNED LEGAL CLAUSE RISK ANALYZER...")
print("=" * 75)

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def analyze_clause(clause_text):
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

    pred_label = ID2LABEL[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs

def render_bar(percentage, length=25):
    filled = int(round(length * (percentage / 100)))
    # Use ASCII characters (# and -) to ensure 100% compatibility across all Windows terminal code pages
    return "#" * filled + "-" * (length - filled)

SAMPLE_CLAUSES = [
    (
        "Amazon Unilateral Termination",
        "Amazon reserves the right in its sole discretion to terminate accounts, terminate your rights to use Amazon Services, or cancel orders without notice."
    ),
    (
        "Alibaba Liability Disclaimer",
        "UNDER NO CIRCUMSTANCES SHALL ALIBABA.COM BE HELD LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, PUNITIVE, INCIDENTAL OR CONSEQUENTIAL DAMAGES INCLUDING LOSS OF PROFITS."
    ),
    (
        "Microsoft Mandatory Arbitration",
        "You and Microsoft agree that any dispute or claim shall be resolved exclusively by binding individual arbitration before the American Arbitration Association, waiving class actions."
    ),
    (
        "Standard Privacy Notice",
        "Please review our Privacy Notice, which also governs your use of our Services, to understand how we collect and protect personal information."
    ),
    (
        "Account Password Security",
        "You are responsible for maintaining the confidentiality of your account and password and for restricting access to your computer."
    )
]

def display_result(clause_text, label, conf, probs):
    icons = {
        "Low": "[LOW RISK]    - Standard informational or mutual operational clause.",
        "Medium": "[MEDIUM RISK] - Operational requirement, user duty, or standard license.",
        "High": "[HIGH RISK]   - WARNING! Restricts consumer rights, limits liability, or allows termination!"
    }

    print("\n" + "-" * 75)
    print(f"CLAUSE ANALYZED:")
    print(f"\"{clause_text}\"")
    print("-" * 75)
    print(f"PREDICTED RISK LEVEL : {label.upper()}")
    print(f"ASSESSMENT           : {icons[label]}")
    print(f"CONFIDENCE           : {conf * 100:.2f}% (Softmax Model Output)\n")
    print("PROBABILITY BREAKDOWN:")
    print(f"  [LOW RISK]    : {probs[0]*100:5.1f}%  |{render_bar(probs[0]*100)}|")
    print(f"  [MEDIUM RISK] : {probs[1]*100:5.1f}%  |{render_bar(probs[1]*100)}|")
    print(f"  [HIGH RISK]   : {probs[2]*100:5.1f}%  |{render_bar(probs[2]*100)}|")
    print("-" * 75)
    print("NOTE: Confidence scores represent mathematical model distributions across classes,")
    print("not formal legal guarantees. High-risk predictions highlight clauses for user review.")
    print("-" * 75)

def run_interactive_demo():
    while True:
        print("\n" + "=" * 75)
        print("  AUTOMATED TERMS & CONDITIONS RISK ANALYZER - LIVE DEMO MENU")
        print("=" * 75)
        print("Choose an option:")
        for idx, (title, _) in enumerate(SAMPLE_CLAUSES, 1):
            print(f"  [{idx}] Test Sample: {title}")
        print("  [6] Type or paste your OWN custom legal clause")
        print("  [0] Exit Demo")
        print("=" * 75)

        try:
            choice = input("\nEnter choice (0-6): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting demo. Goodbye!")
            break

        if choice == "0":
            print("\nExiting demo. Thank you!")
            break
        elif choice in ["1", "2", "3", "4", "5"]:
            idx = int(choice) - 1
            title, text = SAMPLE_CLAUSES[idx]
            label, conf, probs = analyze_clause(text)
            display_result(text, label, conf, probs)
        elif choice == "6":
            try:
                custom_text = input("\nEnter or paste the contractual clause:\n> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not custom_text:
                print("[!] Empty clause provided. Try again.")
                continue
            label, conf, probs = analyze_clause(custom_text)
            display_result(custom_text, label, conf, probs)
        else:
            print("[!] Invalid option. Please choose between 0 and 6.")

if __name__ == "__main__":
    run_interactive_demo()
