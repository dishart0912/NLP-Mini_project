"""
Flowchart Generator for LLM Fine-Tuning Pipeline
Generates high-resolution flowchart image and Mermaid diagram for report/presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(12, 14), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Styles
    box_style = dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB", edgecolor="#2980B9", linewidth=2)
    process_style = dict(boxstyle="round,pad=0.5", facecolor="#E8F8F5", edgecolor="#16A085", linewidth=2)
    model_style = dict(boxstyle="round,pad=0.5", facecolor="#FEF9E7", edgecolor="#F39C12", linewidth=2)
    eval_style = dict(boxstyle="round,pad=0.5", facecolor="#FDEDEC", edgecolor="#C0392B", linewidth=2)
    output_style = dict(boxstyle="round,pad=0.5", facecolor="#F4ECF7", edgecolor="#8E44AD", linewidth=2)

    boxes = [
        (5, 13.0, "Raw Contract Corpus\n(Amazon.txt, Alibaba.txt, tc_clauses.csv)", box_style),
        (5, 11.4, "Stage 1: Mandatory Preprocessing\n• Sentence Segmentation (sent_tokenize)\n• Tokenization (word_tokenize)\n• POS Tagging & POS-informed Lemmatization\n• WordNet Semantic Analysis (Hypernyms / Hyponyms)", process_style),
        (5, 9.7, "Stage 2: Dataset Preparation & Stratification\n• 82 Domain Clauses across 11 Companies\n• Stratified Split: Train (70%), Val (15%), Test (15%)\n• Balanced Class Weight Calculation", process_style),
        (5, 8.0, "Stage 3: Tokenization & PyTorch DataLoader\n• BertTokenizer (prajjwal1/bert-tiny)\n• Max Length = 128, PyTorch Tensors, Batch Size = 4", process_style),
        (5, 6.2, "Stage 4: Transformer Architecture & Fine-Tuning\n• Pretrained BERT-Tiny (~4.4M Parameters)\n• AdamW Optimizer (LR = 1e-4, Weight Decay = 0.01)\n• Weighted Cross-Entropy Loss (12 Epochs)\n• Best-Validation-Checkpoint Selection (Min Val Loss)", model_style),
        (5, 4.3, "Stage 5: Evaluation & Baseline Benchmarking\n• Baseline: TF-IDF + Logistic Regression (Prec: 0.6528, F1: 0.60)\n• Fine-Tuned BERT: Acc: 61.54% | Prec: 0.8485 (+19.6%)\n• 100% High-Risk Recall (6/6 Test Clauses Detected)", eval_style),
        (5, 2.5, "Stage 6: Artifact Serialization & Best Checkpoint\n• PyTorch weights saved to /model_output\n• BertTokenizer configs & vocab saved\n• evaluation_metrics.json & sample_predictions.json", output_style),
        (5, 0.9, "Stage 7: Production Inference Pipeline\n• Unseen Clause Ingestion\n• Softmax Probability Distribution\n• Automated Clause Risk Flagging (Low / Medium / High)", output_style)
    ]

    for x, y, text, style in boxes:
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', bbox=style)

    # Arrows
    arrow_props = dict(facecolor='#2C3E50', edgecolor='#2C3E50', width=1.5, headwidth=8, headlength=10)
    transitions = [
        (12.4, 11.9),
        (10.7, 10.2),
        (9.1, 8.5),
        (7.4, 6.9),
        (5.3, 4.8),
        (3.6, 3.1),
        (1.8, 1.4)
    ]

    for y_start, y_end in transitions:
        ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))

    plt.title("End-to-End LLM Fine-Tuning & Evaluation Flowchart\nAutomated Terms & Conditions Clause Risk Analyzer",
              fontsize=13, fontweight='bold', pad=20, color="#1A5276")
    plt.tight_layout()
    plt.savefig("fine_tuning_flowchart.png", dpi=300, bbox_inches='tight')
    print("Flowchart successfully saved to fine_tuning_flowchart.png")

if __name__ == "__main__":
    draw_flowchart()
