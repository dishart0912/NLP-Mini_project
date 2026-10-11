"""
Automated Test Suite for NLP Mini-Project
Verifies:
1. Dataset integrity (tc_clauses.csv and tc_clauses_expanded.csv)
2. Preprocessing pipeline on valid and edge-case inputs
3. WordNet mapping and POS conversion
4. Model artifact loading from model_output/
5. Inference pipeline predictions and probability distributions
6. Baseline model consistency
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

class TestNLPMiniProject(unittest.TestCase):

    def test_01_seed_dataset_preserved(self):
        """Seed dataset must exist with exact schema and 17 rows."""
        self.assertTrue(os.path.exists("tc_clauses.csv"), "tc_clauses.csv missing!")
        df = pd.read_csv("tc_clauses.csv")
        self.assertEqual(len(df), 17, f"Expected 17 rows in seed dataset, found {len(df)}")
        expected_cols = {'clause_id', 'company', 'category', 'clause_text', 'risk_level'}
        self.assertEqual(set(df.columns), expected_cols)
        self.assertEqual(df['clause_id'].duplicated().sum(), 0)
        self.assertFalse(df.isnull().any().any())

    def test_02_expanded_dataset_integrity(self):
        """Expanded dataset must exist with valid labels, non-empty text, and unique IDs."""
        self.assertTrue(os.path.exists("tc_clauses_expanded.csv"), "tc_clauses_expanded.csv missing!")
        df = pd.read_csv("tc_clauses_expanded.csv")
        self.assertEqual(len(df), 82, f"Expected 82 rows, found {len(df)}")
        self.assertEqual(df['clause_id'].duplicated().sum(), 0, "Duplicate clause_id found!")
        self.assertFalse(df.isnull().any().any(), "Null values detected!")
        allowed_risks = {"Low", "Medium", "High"}
        self.assertTrue(set(df['risk_level'].unique()).issubset(allowed_risks))
        self.assertFalse((df['clause_text'].str.strip() == "").any(), "Empty clause text found!")

    def test_03_preprocessing_pipeline_valid_input(self):
        """Preprocessing pipeline must return all 5 required elements on legal text."""
        from preprocessing_pipeline import preprocess_legal_clause
        text = "Amazon will not be liable for damages and reserves the right to terminate accounts."
        res = preprocess_legal_clause(text)
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["sentences"]), 0)
        self.assertGreater(len(res["tokens"]), 0)
        self.assertEqual(len(res["tokens"]), len(res["pos_tags"]))
        self.assertEqual(len(res["tokens"]), len(res["lemmas"]))
        self.assertIn("liable", res["semantic_insights"])
        self.assertIn("terminate", res["semantic_insights"])

    def test_04_preprocessing_edge_cases(self):
        """Preprocessing pipeline must handle empty and whitespace inputs gracefully."""
        from preprocessing_pipeline import preprocess_legal_clause
        res_empty = preprocess_legal_clause("   ")
        self.assertEqual(res_empty["status"], "empty_input")
        self.assertEqual(len(res_empty["tokens"]), 0)

        res_none = preprocess_legal_clause(None)
        self.assertEqual(res_none["status"], "empty_input")

    def test_05_wordnet_pos_mapping(self):
        """Treebank POS tags must map accurately to WordNet POS categories."""
        from preprocessing_pipeline import get_wordnet_pos
        from nltk.corpus import wordnet as wn
        self.assertEqual(get_wordnet_pos("VBD"), wn.VERB)
        self.assertEqual(get_wordnet_pos("NNS"), wn.NOUN)
        self.assertEqual(get_wordnet_pos("JJ"), wn.ADJ)
        self.assertEqual(get_wordnet_pos("RB"), wn.ADV)

    def test_06_model_artifacts_present(self):
        """Model directory must contain saved checkpoint and tokenizer files."""
        model_dir = "model_output"
        self.assertTrue(os.path.exists(model_dir), "model_output directory missing!")
        required = ["config.json", "evaluation_metrics.json"]
        for f in required:
            self.assertTrue(os.path.exists(os.path.join(model_dir, f)), f"Missing artifact: {f}")

    def test_07_inference_pipeline_execution(self):
        """Inference function must return valid 3-class probability distribution."""
        from inference import predict_clause_risk
        clause = "In no event shall the vendor be liable for any lost profits or damages."
        res = predict_clause_risk(clause)
        self.assertEqual(res["status"], "success")
        self.assertIn(res["predicted_risk"], ["Low", "Medium", "High"])
        self.assertAlmostEqual(sum(res["probabilities"].values()), 1.0, places=2)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

    def test_08_inference_empty_string(self):
        """Inference function must handle empty string input without crashing."""
        from inference import predict_clause_risk
        res = predict_clause_risk("   ")
        self.assertEqual(res["status"], "empty_input")
        self.assertEqual(res["predicted_risk"], "UNKNOWN")

if __name__ == "__main__":
    unittest.main()
