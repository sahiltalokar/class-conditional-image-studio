"""
Tests for Prompt & Truncation Comparison Grid (5 prompts x 3 truncations = 15 images).
Verifies sequential generation, distinct seeds, dimensions, error handling, and latency.
"""

import os
import sys
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.loader import load_biggan_model
from utils.comparison_service import run_comparison_grid, DEFAULT_TRUNCATIONS


class TestComparisonGrid(unittest.TestCase):
    """Verifies the multi-prompt and truncation comparison pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_biggan_model()

    def test_five_prompts_three_truncations(self):
        prompts = [
            "golden retriever",
            "sports car",
            "tiger shark",
            "espresso",
            "pizza",
        ]
        truncations = [0.2, 0.4, 0.8]

        progress_calls = []
        def track_progress(cur, tot, txt):
            progress_calls.append((cur, tot, txt))

        result = run_comparison_grid(
            model=self.model,
            prompts=prompts,
            truncations=truncations,
            base_seed=12345,
            progress_callback=track_progress,
        )

        # 1. Structure checks
        self.assertEqual(len(result["rows"]), 5, "Must produce exactly 5 rows.")
        self.assertEqual(result["total_images"], 15, "Must successfully generate 15 images.")
        self.assertEqual(result["success_count"], 15)
        self.assertEqual(result["failure_count"], 0)
        self.assertGreater(len(progress_calls), 0, "Progress callback should be invoked.")

        # 2. Per-row and per-cell checks
        seen_seeds = set()
        for r_idx, row in enumerate(result["rows"]):
            self.assertIsNotNone(row["class_id"], f"Row {r_idx} must have valid class_id.")
            self.assertIsNone(row["error"], f"Row {r_idx} should have no error.")
            self.assertEqual(len(row["items"]), 3, f"Row {r_idx} must have 3 truncation items.")

            for c_idx, item in enumerate(row["items"]):
                self.assertEqual(item["truncation"], truncations[c_idx])
                self.assertIsNotNone(item["seed"])
                self.assertNotIn(item["seed"], seen_seeds, f"Seed {item['seed']} must be unique across cells.")
                seen_seeds.add(item["seed"])

                # Check generated image
                img = item["image"]
                self.assertIsInstance(img, Image.Image)
                self.assertEqual(img.size, (128, 128), "Image must be 128x128 pixels.")
                self.assertEqual(img.mode, "RGB", "Image must be RGB.")

                # Check generation time
                self.assertGreater(item["generation_time"], 0.0)

        self.assertEqual(len(seen_seeds), 15, "All 15 images must have unique seeds.")
        print(f"\n[TestComparisonGrid] Successfully generated 15 images in {result['total_time']:.3f} seconds total.")

    def test_invalid_and_empty_prompt_handling(self):
        mixed_prompts = [
            "goldfish",
            "",  # empty prompt
            "cliff",
        ]
        result = run_comparison_grid(
            model=self.model,
            prompts=mixed_prompts,
            truncations=[0.2, 0.5],
            base_seed=42,
        )

        self.assertEqual(len(result["rows"]), 3)
        # Row 0 (goldfish) succeeds: 2 images
        self.assertEqual(len(result["rows"][0]["items"]), 2)
        self.assertIsNone(result["rows"][0]["error"])

        # Row 1 (empty) handled gracefully
        self.assertIsNotNone(result["rows"][1]["error"])
        self.assertEqual(len(result["rows"][1]["items"]), 0)

        # Row 2 (cliff) succeeds: 2 images
        self.assertEqual(len(result["rows"][2]["items"]), 2)
        self.assertIsNone(result["rows"][2]["error"])

        self.assertEqual(result["success_count"], 4)
        self.assertEqual(result["failure_count"], 2)

    def test_screens_and_maps_all_prompts_before_generation(self):
        events = []

        def fake_safety(prompt, threshold):
            events.append("safety")
            return {"is_safe": True, "toxic_score": 0.0}

        def fake_mapping(prompt):
            events.append("mapping")
            return True, {
                "class_id": 207,
                "class_name": "golden retriever",
                "explanation": "test",
            }, None

        def fake_generation(model, class_id, truncation, seed):
            events.append("generation")
            return Image.new("RGB", (128, 128)), 0.1

        with patch("utils.comparison_service.check_prompt_safety", side_effect=fake_safety), \
             patch("utils.comparison_service.map_prompt_to_imagenet", side_effect=fake_mapping), \
             patch(
                 "utils.comparison_service.release_safety_classifier",
                 side_effect=lambda: events.append("release_safety"),
             ), \
             patch("utils.comparison_service.generate_single_image", side_effect=fake_generation):
            result = run_comparison_grid(
                model=object(),
                prompts=["golden retriever", "sports car"],
                truncations=[0.2],
                base_seed=42,
            )

        release_index = events.index("release_safety")
        first_generation_index = events.index("generation")
        self.assertEqual(events.count("safety"), 2)
        self.assertEqual(events.count("mapping"), 2)
        self.assertLess(events.index("mapping", events.index("mapping") + 1), release_index)
        self.assertLess(release_index, first_generation_index)
        self.assertEqual(result["success_count"], 2)


if __name__ == "__main__":
    unittest.main()
