"""
Evaluation Tests for Class-Conditional Image Studio using BigGAN.
Tests DistilBERT prompt safety screening and CLIP vision-language evaluation.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.safety_classifier import check_prompt_safety, DEFAULT_SAFETY_THRESHOLD
from models.clip_evaluator import compute_clip_similarity, interpret_clip_score


class TestDistilBERTSafetyScreening(unittest.TestCase):
    """
    Verifies DistilBERT safety screening using martin-ha/toxic-comment-model.
    Labels: LABEL_0 = non-toxic, LABEL_1 = toxic.
    """

    def test_safe_prompt_passes(self):
        """A benign image description should be classified as non-toxic and pass."""
        result = check_prompt_safety("A golden retriever dog sitting on green grass")
        self.assertIsNone(result["error"], f"Expected no error; got: {result['error']}")
        self.assertTrue(result["is_safe"], "Benign animal prompt should pass safety screening.")
        self.assertLess(result["toxic_score"], DEFAULT_SAFETY_THRESHOLD)
        self.assertEqual(result["predicted_label"], "non-toxic")

    def test_toxic_prompt_blocked(self):
        """A clearly toxic prompt should be flagged and blocked above the default threshold."""
        result = check_prompt_safety("fuck you, die in a fire")
        self.assertIsNone(result["error"], f"Expected no error; got: {result['error']}")
        self.assertFalse(result["is_safe"], "Toxic prompt should be blocked by the safety filter.")
        self.assertGreater(result["toxic_score"], DEFAULT_SAFETY_THRESHOLD)
        self.assertEqual(result["predicted_label"], "toxic")

    def test_disgusting_insult_blocked(self):
        """Another toxic prompt variant should also be blocked."""
        result = check_prompt_safety("You are disgusting and useless")
        self.assertIsNone(result["error"], f"Expected no error; got: {result['error']}")
        self.assertFalse(result["is_safe"], "Toxic insult prompt should be blocked.")
        self.assertGreater(result["toxic_score"], DEFAULT_SAFETY_THRESHOLD)

    def test_empty_prompt_blocked(self):
        """An empty prompt should not pass safety check."""
        result = check_prompt_safety("")
        self.assertFalse(result["is_safe"], "Empty prompt must not pass safety check.")
        self.assertIsNotNone(result["error"], "Empty prompt should set an error message.")

    def test_whitespace_only_blocked(self):
        """A whitespace-only prompt should not pass safety check."""
        result = check_prompt_safety("   ")
        self.assertFalse(result["is_safe"], "Whitespace-only prompt must not pass safety check.")

    def test_custom_threshold_lowers_bar(self):
        """With a very high threshold, a borderline prompt should pass."""
        # A completely benign sentence at a threshold of 0.95 should still easily pass
        result = check_prompt_safety("A beautiful mountain landscape", threshold=0.95)
        self.assertTrue(result["is_safe"], "Benign prompt should pass at a 95% threshold.")
        self.assertEqual(result["threshold"], 0.95)

    def test_custom_threshold_raises_bar(self):
        """With a very low threshold (0.0001), a benign prompt scoring >0% toxicity should be blocked."""
        result = check_prompt_safety("A golden retriever", threshold=0.0001)
        # The model always assigns a small positive toxicity mass; at 0.01% threshold this triggers
        self.assertFalse(result["is_safe"], "Benign prompt should be blocked at a near-zero threshold.")
        self.assertEqual(result["threshold"], 0.0001)

    def test_result_contains_all_keys(self):
        """Result dict must always contain all required keys."""
        result = check_prompt_safety("A cat on a sofa")
        required_keys = {"is_safe", "toxic_score", "non_toxic_score", "predicted_label", "threshold", "reason", "error"}
        for k in required_keys:
            self.assertIn(k, result, f"Missing key '{k}' in safety result dict.")

    def test_probabilities_sum_to_one(self):
        """Non-toxic + toxic probabilities should sum to approximately 1.0."""
        result = check_prompt_safety("A red sports car parked on a road")
        total = result["toxic_score"] + result["non_toxic_score"]
        self.assertAlmostEqual(total, 1.0, places=4, msg="Probabilities must sum to 1.0.")

    def test_model_load_failure_blocks_generation(self):
        """If the model fails to load, the result must be is_safe=False (fail-closed)."""
        # Patch the function that check_prompt_safety actually calls to get the loaded pair
        with patch("models.safety_classifier.get_safety_classifier",
                   side_effect=RuntimeError("Simulated load error")):
            result = check_prompt_safety("A golden retriever")
            self.assertFalse(result["is_safe"],
                "Safety screening failure must block generation (fail-closed), never silently pass.")
            self.assertIsNotNone(result["error"])


class TestCLIPSimilarity(unittest.TestCase):
    """
    Verifies CLIP ViT-B/32 vision-language cosine similarity computation.
    """

    @classmethod
    def setUpClass(cls):
        """Create a small test image once per test class to avoid re-creating it every test."""
        cls.dummy_image = Image.new("RGB", (128, 128), color=(100, 180, 60))
        cls.blue_image = Image.new("RGB", (128, 128), color=(30, 80, 200))

    def test_basic_similarity_range(self):
        """CLIP similarity must be a float in [-1.0, 1.0]."""
        result = compute_clip_similarity("A green field with grass", self.dummy_image)
        self.assertIsNone(result["error"], f"Expected no error; got: {result['error']}")
        self.assertIsNotNone(result["similarity"])
        score = result["similarity"]
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

    def test_result_contains_all_keys(self):
        """Result dict must always contain all required keys."""
        result = compute_clip_similarity("A dog on grass", self.dummy_image)
        required_keys = {"similarity", "prompt", "interpretation", "explanation", "error"}
        for k in required_keys:
            self.assertIn(k, result, f"Missing key '{k}' in CLIP result dict.")

    def test_known_prompt_produces_valid_score(self):
        """A realistic prompt should produce a score > -0.5 (not complete semantic opposition)."""
        result = compute_clip_similarity("A golden retriever dog sitting on green grass", self.dummy_image)
        self.assertIsNone(result["error"])
        self.assertGreater(result["similarity"], -0.5, "Score should not be extreme negative.")

    def test_different_prompts_produce_different_scores(self):
        """Two semantically different prompts should produce different CLIP scores for the same image."""
        result_a = compute_clip_similarity("A golden retriever dog", self.dummy_image)
        result_b = compute_clip_similarity("A deep ocean with sharks", self.dummy_image)
        self.assertIsNone(result_a["error"])
        self.assertIsNone(result_b["error"])
        # Scores should differ by some amount (not identical for different prompts)
        self.assertNotAlmostEqual(
            result_a["similarity"], result_b["similarity"], places=3,
            msg="Different prompts should yield different CLIP scores for the same image."
        )

    def test_rgb_conversion(self):
        """CLIP should handle RGBA or grayscale images by converting them to RGB."""
        rgba_image = Image.new("RGBA", (128, 128), color=(100, 150, 200, 128))
        result = compute_clip_similarity("A colorful image", rgba_image)
        self.assertIsNone(result["error"], f"RGBA image should be handled gracefully; got: {result['error']}")
        self.assertIsNotNone(result["similarity"])

    def test_empty_prompt_returns_error(self):
        """An empty prompt must return an error without raising an exception."""
        result = compute_clip_similarity("", self.dummy_image)
        self.assertIsNotNone(result["error"], "Empty prompt must set an error key.")
        self.assertIsNone(result["similarity"])

    def test_invalid_image_type_returns_error(self):
        """Passing a non-PIL object as image must return an error without raising."""
        result = compute_clip_similarity("A dog", "this_is_not_an_image")
        self.assertIsNotNone(result["error"], "Non-PIL image must set an error key.")
        self.assertIsNone(result["similarity"])

    def test_model_load_failure_handled(self):
        """If CLIP fails to load, compute_clip_similarity must return error without crashing."""
        # Patch the function that compute_clip_similarity actually calls to get the loaded pair
        with patch("models.clip_evaluator.get_clip_evaluator",
                   side_effect=RuntimeError("Simulated CLIP load error")):
            result = compute_clip_similarity("A dog", self.dummy_image)
            self.assertIsNotNone(result["error"], "CLIP load failure must set error key.")
            self.assertIsNone(result["similarity"])


class TestCLIPScoreInterpretation(unittest.TestCase):
    """Verifies that CLIP cosine similarity score interpretation thresholds are correct."""

    def test_strong_alignment_label(self):
        self.assertEqual(interpret_clip_score(0.30), "Strong Semantic Alignment")

    def test_moderate_alignment_label(self):
        self.assertEqual(interpret_clip_score(0.22), "Moderate Semantic Alignment")

    def test_weak_alignment_label(self):
        self.assertEqual(interpret_clip_score(0.15), "Weak / Partial Semantic Alignment")

    def test_low_alignment_label(self):
        self.assertEqual(interpret_clip_score(0.05), "Low Semantic Alignment")

    def test_boundary_values(self):
        """Verify boundary values map to expected bands."""
        self.assertEqual(interpret_clip_score(0.28), "Strong Semantic Alignment")
        self.assertEqual(interpret_clip_score(0.20), "Moderate Semantic Alignment")
        self.assertEqual(interpret_clip_score(0.12), "Weak / Partial Semantic Alignment")


if __name__ == "__main__":
    unittest.main()
