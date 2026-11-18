"""
Core Unit and Integration Tests for Class-Conditional Image Studio using BigGAN.
Tests ImageNet taxonomy, sampling, BigGAN forward pass, and Groq parsing.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import torch
from PIL import Image
from groq import APIStatusError

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.imagenet_utils import (
    load_imagenet_classes,
    validate_class_id,
    get_class_by_id,
    fuzzy_search_imagenet,
)
from utils.sampling import (
    generate_truncated_noise,
    prepare_class_vector,
    tensor_to_pil,
)
from utils.groq_client import (
    map_prompt_to_imagenet,
    get_groq_config,
    _sanitize_error_message,
    _reconcile_class_mapping,
)
from models.loader import load_biggan_model


class TestImageNetTaxonomy(unittest.TestCase):
    """Verifies that the ImageNet-1K taxonomy is loaded and validated correctly."""

    def test_class_count_and_keys(self):
        classes = load_imagenet_classes()
        self.assertEqual(len(classes), 1000, "Taxonomy must contain exactly 1,000 classes.")
        for idx, item in enumerate(classes):
            self.assertEqual(item["class_id"], idx)
            self.assertIn("name", item)
            self.assertIn("wnid", item)
            self.assertIn("display_name", item)

    def test_known_classes(self):
        self.assertEqual(get_class_by_id(0)["name"], "tench")
        self.assertEqual(get_class_by_id(1)["name"], "goldfish")
        self.assertEqual(get_class_by_id(207)["name"], "golden retriever")
        self.assertEqual(get_class_by_id(817)["name"], "sports car")
        self.assertEqual(get_class_by_id(983)["name"], "scuba diver")

    def test_validation_boundaries(self):
        self.assertTrue(validate_class_id(0)[0])
        self.assertTrue(validate_class_id(500)[0])
        self.assertTrue(validate_class_id(999)[0])
        self.assertFalse(validate_class_id(-1)[0])
        self.assertFalse(validate_class_id(1000)[0])
        self.assertFalse(validate_class_id("invalid")[0])
        self.assertFalse(validate_class_id(None)[0])

    def test_fuzzy_search(self):
        matches = fuzzy_search_imagenet("golden retriever", top_k=3)
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0]["class_id"], 207)

        matches_shark = fuzzy_search_imagenet("shark", top_k=5)
        names = [m["name"] for m in matches_shark]
        self.assertTrue(any("shark" in n for n in names))


class TestSamplingUtilities(unittest.TestCase):
    """Verifies noise sampling, one-hot vector creation, and tensor-to-image conversion."""

    def test_class_vector(self):
        vec = prepare_class_vector(207, num_classes=1000)
        self.assertEqual(vec.shape, (1, 1000))
        self.assertEqual(vec[0, 207].item(), 1.0)
        self.assertEqual(vec.sum().item(), 1.0)

        with self.assertRaises(ValueError):
            prepare_class_vector(1000)

    def test_truncated_noise(self):
        trunc = 0.5
        noise = generate_truncated_noise(batch_size=1, dim_z=128, truncation=trunc, seed=42)
        self.assertEqual(noise.shape, (1, 128))
        self.assertEqual(noise.dtype, torch.float32)
        self.assertTrue((noise >= -1.0).all().item())
        self.assertTrue((noise <= 1.0).all().item())

    def test_tensor_to_pil(self):
        dummy_tensor = torch.zeros(1, 3, 128, 128)
        img = tensor_to_pil(dummy_tensor)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (128, 128))
        self.assertEqual(img.mode, "RGB")


class TestBigGANModelExecution(unittest.TestCase):
    """Verifies that BigGAN-deep-128 loads and generates valid images."""

    @classmethod
    def setUpClass(cls):
        cls.model = load_biggan_model()

    def test_model_loaded_and_eval_mode(self):
        self.assertIsNotNone(self.model)
        self.assertFalse(self.model.training, "BigGAN must be in eval mode.")

    def test_forward_pass_and_image_generation(self):
        truncation = 0.5
        noise = generate_truncated_noise(batch_size=1, dim_z=128, truncation=truncation, seed=123)
        class_vec = prepare_class_vector(207)

        with torch.no_grad():
            output = self.model(noise, class_vec, truncation)

        self.assertEqual(output.shape, (1, 3, 128, 128))
        self.assertEqual(output.dtype, torch.float32)

        img = tensor_to_pil(output)
        self.assertEqual(img.size, (128, 128))
        self.assertEqual(img.mode, "RGB")


class TestGroqResponseParsing(unittest.TestCase):
    """Verifies parsing of valid, malformed, and out-of-range Groq outputs."""

    def test_empty_prompt(self):
        success, res, err = map_prompt_to_imagenet("")
        self.assertFalse(success)
        self.assertIn("empty", err.lower())

    def test_missing_api_key(self):
        success, res, err = map_prompt_to_imagenet("a fluffy dog", api_key="")
        self.assertFalse(success)
        self.assertIn("not configured", err.lower())

    def test_api_key_placeholder_is_not_treated_as_a_credential(self):
        success, res, err = map_prompt_to_imagenet(
            "a fluffy dog", api_key="YOUR_GROQ_API_KEY"
        )
        self.assertFalse(success)
        self.assertIn("not configured", err.lower())

    def test_groq_settings_are_read_from_streamlit_secrets(self):
        with patch(
            "utils.groq_client.st.secrets",
            {"GROQ_API_KEY": "test-key", "GROQ_MODEL": "test-model"},
        ), patch.dict(os.environ, {"GROQ_API_KEY": "", "GROQ_MODEL": ""}):
            self.assertEqual(get_groq_config(), ("test-key", "test-model"))

    def test_missing_groq_secrets_use_no_key_default(self):
        with patch("utils.groq_client.st.secrets", {}), patch.dict(
            os.environ, {"GROQ_API_KEY": "", "GROQ_MODEL": ""}
        ):
            self.assertEqual(get_groq_config(), (None, "openai/gpt-oss-20b"))

    @patch("utils.groq_client.Groq")
    def test_valid_groq_json_response(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "class_id": 207,
            "class_name": "golden retriever",
            "explanation": "The prompt describes a golden retriever."
        })
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        success, res, err = map_prompt_to_imagenet("golden retriever", api_key="gsk_test123")
        self.assertTrue(success)
        self.assertIsNone(err)
        self.assertEqual(res["class_id"], 207)
        self.assertEqual(res["class_name"], "golden retriever")

    def test_class_reconciliation(self):
        # 1. Sports car with mismatched class_id 0 -> should reconcile to 817
        valid, cid, cname = _reconcile_class_mapping(0, "sports car")
        self.assertTrue(valid)
        self.assertEqual(cid, 817)
        self.assertEqual(cname, "sports car")

        # 2. Golden retriever as string ID "207"
        valid, cid, cname = _reconcile_class_mapping("207", "golden retriever")
        self.assertTrue(valid)
        self.assertEqual(cid, 207)

        # 3. WordNet ID
        valid, cid, cname = _reconcile_class_mapping("n02099601", "")
        self.assertTrue(valid)
        self.assertEqual(cid, 207)

    def test_error_sanitization(self):
        raw = "Failed request with key gsk_1234567890abcdef1234 and Bearer gsk_secret_token"
        sanitized = _sanitize_error_message(raw)
        self.assertNotIn("gsk_1234567890abcdef1234", sanitized)
        self.assertIn("[REDACTED", sanitized)

    @patch("utils.groq_client.Groq")
    def test_out_of_range_class_id(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "class_id": 1500,
            "class_name": "fictional creature",
            "explanation": "Out of range index."
        })
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        success, res, err = map_prompt_to_imagenet("alien monster", api_key="gsk_test123")
        self.assertFalse(success)
        self.assertIn("could not be resolved", err.lower())

    @patch("utils.groq_client.Groq")
    def test_malformed_json_response(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client
        mock_choice = MagicMock()
        mock_choice.message.content = "Not a valid JSON string"
        mock_client.chat.completions.create.return_value.choices = [mock_choice]

        success, res, err = map_prompt_to_imagenet("dog", api_key="gsk_test123")
        self.assertFalse(success)
        self.assertIn("failed to parse structured json", err.lower())

    @patch("utils.groq_client.Groq")
    def test_api_status_400_captures_message(self, mock_groq_cls):
        mock_client = MagicMock()
        mock_groq_cls.return_value = mock_client

        # Simulate 400 error from Groq
        mock_err = APIStatusError(
            message="Error code: 400 - {'error': {'message': 'Failed to validate JSON (json_validate_failed)'}}",
            response=MagicMock(status_code=400),
            body={"error": {"message": "Failed to validate JSON (json_validate_failed)"}}
        )
        mock_client.chat.completions.create.side_effect = mock_err

        success, res, err = map_prompt_to_imagenet("sports car", api_key="gsk_test123")
        self.assertFalse(success)
        self.assertIn("400", err)
        self.assertIn("Failed to validate JSON", err)


if __name__ == "__main__":
    unittest.main()
