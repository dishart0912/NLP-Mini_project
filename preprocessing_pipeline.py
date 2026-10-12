"""
Comprehensive Legal Text Preprocessing Module
Covers all 5 mandatory preprocessing requirements:
1. Sentence boundary identification
2. Tokenization
3. Part-of-Speech (POS) tagging
4. POS-informed Lemmatization
5. WordNet semantic analysis

ARCHITECTURAL NOTE:
This module demonstrates classical NLP linguistic processing on legal clauses.
The BERT fine-tuning model (fine_tune_bert.py) utilizes its own subword WordPiece
tokenizer (BertTokenizer) to preserve full textual context, syntactic structure,
and word inflections directly from raw clause strings. This module operates as an
independent analytical and verification component.
"""

import sys
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn

# Global flag to avoid repeatedly checking/downloading NLTK corpora
_NLTK_CHECKED = False

def ensure_nltk_corpora():
    """
    Ensures required NLTK corpora are downloaded and unzipped once per session.
    Fails gracefully with helpful instructions if network is unavailable.
    """
    global _NLTK_CHECKED
    if _NLTK_CHECKED:
        return True

    packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    import zipfile
    import os

    for resource, pkg in packages:
        try:
            nltk.data.find(resource)
        except (LookupError, AttributeError):
            try:
                nltk.download(pkg, quiet=True)
                # On Windows, sometimes NLTK downloads the zip without unzipping it
                for search_dir in nltk.data.path:
                    zip_candidate = os.path.join(search_dir, resource + '.zip')
                    if os.path.exists(zip_candidate):
                        target_dir = os.path.dirname(zip_candidate)
                        with zipfile.ZipFile(zip_candidate) as zf:
                            zf.extractall(target_dir)
            except Exception as e:
                print(f"[!] Warning: Could not download or extract NLTK package '{pkg}': {e}", file=sys.stderr)
                print(f"    Please run 'python -m nltk.downloader {pkg}' manually if missing.", file=sys.stderr)
    
    _NLTK_CHECKED = True
    return True

def get_wordnet_pos(treebank_tag):
    """
    Maps Penn Treebank POS tags to WordNet POS categories.
    Accurate POS mapping prevents lemmatizing verbs as nouns or adjectives as verbs.
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def preprocess_legal_clause(clause_text):
    """
    Executes the 5 mandatory preprocessing steps on a given legal clause text.
    Handles empty, whitespace, and non-string inputs gracefully.
    """
    if not isinstance(clause_text, str) or not clause_text.strip():
        return {
            "original_text": clause_text if isinstance(clause_text, str) else "",
            "sentences": [],
            "tokens": [],
            "pos_tags": [],
            "lemmas": [],
            "semantic_insights": {},
            "status": "empty_input"
        }

    ensure_nltk_corpora()
    
    # 1. Sentence Boundary Identification
    try:
        sentences = sent_tokenize(clause_text)
    except Exception:
        sentences = [s.strip() for s in clause_text.split('.') if s.strip()]

    # 2. Tokenization
    try:
        tokens = word_tokenize(clause_text)
    except Exception:
        tokens = clause_text.split()

    # 3. Part-of-Speech Tagging
    try:
        pos_tags = pos_tag(tokens)
    except Exception:
        pos_tags = [(w, 'NN') for w in tokens]

    # 4. Lemmatization (POS-informed WordNet mapping)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = []
    for word, tag in pos_tags:
        try:
            wn_pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, wn_pos)
        except Exception:
            lemma = word
        lemmatized_tokens.append(lemma)
    
    # 5. WordNet Semantic Analysis
    # NOTE: WordNet is a general-purpose lexical database. While highly effective
    # for everyday English, it has known domain limitations for specialized legalese
    # (e.g. archaic Latin maxims, complex statutory terms). We target core legal nouns & verbs.
    legal_keywords = [
        'liable', 'liability', 'terminate', 'termination', 'dispute',
        'arbitration', 'warranty', 'breach', 'indemnify', 'indemnification',
        'damages', 'severability', 'jurisdiction'
    ]
    semantic_insights = {}
    
    for word in tokens:
        clean_word = word.lower()
        if clean_word in legal_keywords and clean_word not in semantic_insights:
            try:
                synsets = wn.synsets(clean_word)
                if synsets:
                    primary_syn = synsets[0]
                    hypernyms = [h.name().split('.')[0] for h in primary_syn.hypernyms()]
                    hyponyms = [h.name().split('.')[0] for h in primary_syn.hyponyms()]
                    semantic_insights[clean_word] = {
                        "synset": primary_syn.name(),
                        "pos": primary_syn.pos(),
                        "definition": primary_syn.definition(),
                        "hypernyms": hypernyms,
                        "hyponyms": hyponyms[:5]
                    }
            except Exception:
                pass
                
    return {
        "original_text": clause_text,
        "sentences": sentences,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmatized_tokens,
        "semantic_insights": semantic_insights,
        "status": "success"
    }

if __name__ == "__main__":
    import json
    import os
    import pandas as pd

    print("=" * 75)
    print("  MANDATORY PREPROCESSING PIPELINE (FULL DOCUMENT & CLAUSE ANALYSIS)")
    print("=" * 75)

    doc_file = "amazon.txt"
    output_dir = "preprocessing_output"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(doc_file):
        with open(doc_file, "r", encoding="utf-8") as f:
            full_text = f.read()

        print(f"\n[STEP 1] LOADING ENTIRE DOCUMENT: '{doc_file}'")
        print(f"         Total Characters : {len(full_text):,}")
        print(f"         Total Words      : {len(full_text.split()):,}")

        # Process the entire document
        full_result = preprocess_legal_clause(full_text)
        all_sentences = full_result["sentences"]
        total_clauses = len(all_sentences)
        print(f"         Clauses Identified: {total_clauses} discrete contractual obligations")

        # Process each individual clause to create structured document output
        detailed_clauses = []
        for idx, sentence in enumerate(all_sentences, 1):
            clause_res = preprocess_legal_clause(sentence)
            detailed_clauses.append({
                "clause_id": idx,
                "clause_text": sentence,
                "token_count": len(clause_res["tokens"]),
                "tokens": clause_res["tokens"],
                "pos_tags": clause_res["pos_tags"],
                "lemmas": clause_res["lemmas"],
                "legal_terms_found": list(clause_res["semantic_insights"].keys()),
                "semantic_insights": clause_res["semantic_insights"]
            })

        # Save complete structured output to preprocessing_output/
        json_output_path = os.path.join(output_dir, "amazon_preprocessed_full.json")
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "document": doc_file,
                "total_clauses": total_clauses,
                "clauses": detailed_clauses,
                "full_document_legal_terms": full_result["semantic_insights"]
            }, f, indent=2)

        csv_summary_path = os.path.join(output_dir, "amazon_clauses_summary.csv")
        summary_rows = [
            {
                "clause_id": c["clause_id"],
                "clause_text": c["clause_text"],
                "token_count": c["token_count"],
                "legal_keywords": ", ".join(c["legal_terms_found"]) if c["legal_terms_found"] else "None"
            }
            for c in detailed_clauses
        ]
        pd.DataFrame(summary_rows).to_csv(csv_summary_path, index=False)

        print(f"\n[+] FULL DOCUMENT ANALYSIS SAVED TO '{output_dir}/':")
        print(f"    - Structured JSON: {json_output_path}")
        print(f"    - Clause CSV Summary: {csv_summary_path}")

        # Display Clause 1 on the terminal
        first_clause = detailed_clauses[0]
        print("\n" + "=" * 75)
        print(f"  COMMAND-LINE PREVIEW: CLAUSE 1 of {total_clauses}")
        print("=" * 75)
        print(f"Text:\n\"{first_clause['clause_text']}\"")
        print(f"\n1. Sentence Boundary : Isolated as Clause #1 of {total_clauses}")
        print(f"2. Tokenization      : {first_clause['token_count']} tokens")
        print("   Sample Tokens     :", first_clause['tokens'][:10], "...")
        print(f"3. POS Tagging       : Sample pairs ->", first_clause['pos_tags'][:6])
        print(f"4. Lemmatization     : Sample lemmas ->", list(zip(first_clause['tokens'][:6], first_clause['lemmas'][:6])))
        print(f"5. WordNet Analysis  :")
        if first_clause['semantic_insights']:
            for term, data in first_clause['semantic_insights'].items():
                print(f"   * Term '{term}': {data['synset']} | Def: {data['definition']}")
                print(f"     Hypernyms: {data['hypernyms']} | Hyponyms: {data['hyponyms']}")
        else:
            print("   (Introductory clause - no high-liability statutory terms detected in this sentence)")

        # Also show WordNet ontology on the document's legal keywords found overall
        print("\n" + "-" * 75)
        print(f"DOCUMENT-WIDE WORDNET ONTOLOGY ({len(full_result['semantic_insights'])} Legal Terms Detected):")
        print("-" * 75)
        for term, data in list(full_result['semantic_insights'].items())[:3]:
            print(f"   * Term: '{term}'")
            print(f"     Synset     : {data['synset']} ({data['pos']})")
            print(f"     Definition : {data['definition']}")
            print(f"     Hypernyms  : {data['hypernyms']}")
            print(f"     Hyponyms   : {data['hyponyms']}")

    else:
        print(f"[!] '{doc_file}' not found. Falling back to default clause verification.")

    print("\n[+] Edge-Case Verification: Testing empty string...")
    empty_res = preprocess_legal_clause("   ")
    print(f"    Status: {empty_res['status']} | Tokens: {len(empty_res['tokens'])}")
    print("=" * 75)
