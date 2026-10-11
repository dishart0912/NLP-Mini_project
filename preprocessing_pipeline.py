"""
Comprehensive Legal Text Preprocessing Module
Covers all 5 mandatory preprocessing requirements:
1. Sentence boundary identification
2. Tokenization
3. Lemmatization
4. POS tagging
5. WordNet semantic analysis

Preserves existing experiments 1-5 while providing an integrated, executable verification pipeline.
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn

def ensure_nltk_corpora():
    packages = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4']
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            pass

def preprocess_legal_clause(clause_text):
    """
    Executes the 5 mandatory preprocessing steps on a given legal clause text.
    """
    ensure_nltk_corpora()
    
    # 1. Sentence Boundary Identification
    sentences = sent_tokenize(clause_text)
    
    # 2. Tokenization
    tokens = word_tokenize(clause_text)
    
    # 3. POS Tagging
    pos_tags = pos_tag(tokens)
    
    # 4. Lemmatization (POS-informed mapping)
    lemmatizer = WordNetLemmatizer()
    def get_wordnet_pos(treebank_tag):
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

    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]
    
    # 5. WordNet Semantic Analysis (Hypernyms, Hyponyms, Definitions, Synsets)
    legal_keywords = ['liable', 'liability', 'terminate', 'termination', 'dispute', 'arbitration', 'warranty']
    semantic_insights = {}
    
    for word in tokens:
        clean_word = word.lower()
        if clean_word in legal_keywords and clean_word not in semantic_insights:
            synsets = wn.synsets(clean_word)
            if synsets:
                primary_syn = synsets[0]
                hypernyms = [h.name().split('.')[0] for h in primary_syn.hypernyms()]
                hyponyms = [h.name().split('.')[0] for h in primary_syn.hyponyms()]
                semantic_insights[clean_word] = {
                    "synset": primary_syn.name(),
                    "definition": primary_syn.definition(),
                    "hypernyms": hypernyms,
                    "hyponyms": hyponyms[:5]
                }
                
    return {
        "original_text": clause_text,
        "sentences": sentences,
        "tokens": tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmatized_tokens,
        "semantic_insights": semantic_insights
    }

if __name__ == "__main__":
    sample_clause = (
        "Amazon will not be liable for any damages of any kind arising from the use of any "
        "Amazon Service, and reserves the right to terminate accounts without notice."
    )
    print("=" * 60)
    print("MANDATORY PREPROCESSING VERIFICATION PIPELINE")
    print("=" * 60)
    result = preprocess_legal_clause(sample_clause)
    print("\n1. Sentence Boundary Identification:")
    for i, s in enumerate(result['sentences'], 1):
        print(f"   [{i}] {s}")
        
    print(f"\n2. Tokenization ({len(result['tokens'])} tokens):")
    print("  ", result['tokens'][:12], "...")
    
    print(f"\n3. POS Tagging (Sample pairs):")
    print("  ", result['pos_tags'][:8])
    
    print(f"\n4. POS-Informed Lemmatization (Sample lemmas):")
    print("  ", list(zip(result['tokens'][:8], result['lemmas'][:8])))
    
    print(f"\n5. WordNet Semantic Analysis:")
    for term, data in result['semantic_insights'].items():
        print(f"   * Term: '{term}'")
        print(f"     Synset: {data['synset']}")
        print(f"     Definition: {data['definition']}")
        print(f"     Hypernyms: {data['hypernyms']}")
        print(f"     Hyponyms: {data['hyponyms']}")
    print("=" * 60)
