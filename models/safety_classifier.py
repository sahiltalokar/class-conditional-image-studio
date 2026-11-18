"""
Prompt safety screening module using DistilBERT (martin-ha/toxic-comment-model).
Performs sequence classification to detect harmful or toxic text prompts
prior to Groq mapping and BigGAN image synthesis.
"""

import logging
import gc
from typing import Dict, Any
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

SAFETY_MODEL_REPO = "martin-ha/toxic-comment-model"
DEFAULT_SAFETY_THRESHOLD = 0.5

def _load_safety_model_raw():
    """
    Downloads/loads the tokenizer and sequence classification model onto CPU.
    First tries online check (for fresh downloads), and falls back to cached
    local files if offline (e.g. within restricted sandboxes).
    Returns:
        tuple: (tokenizer, model)
    """
    logger.info(f"Loading safety classifier from {SAFETY_MODEL_REPO}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(SAFETY_MODEL_REPO)
    except Exception as net_err:
        error_text = str(net_err).lower()
        if isinstance(net_err, MemoryError) or any(
            phrase in error_text
            for phrase in ("out of memory", "not enough memory", "cannot allocate memory")
        ):
            raise
        logger.warning(f"Online load failed ({net_err}). Attempting local_files_only=True cache load...")
        tokenizer = AutoTokenizer.from_pretrained(SAFETY_MODEL_REPO, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(SAFETY_MODEL_REPO, local_files_only=True)

    model.to("cpu").eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading safety filter (DistilBERT)...")
def _cached_safety_classifier():
    return _load_safety_model_raw()


def get_safety_classifier():
    """
    Retrieves the safety classifier (tokenizer, model).
    The model is loaded only when a prompt is screened.
    """
    return _cached_safety_classifier()


def release_safety_classifier():
    """Release the cached classifier after prompt screening is complete."""
    _cached_safety_classifier.clear()
    gc.collect()


def check_prompt_safety(
    prompt: str,
    threshold: float = DEFAULT_SAFETY_THRESHOLD
) -> Dict[str, Any]:
    """
    Screens a user text prompt for toxicity or harmful content.

    Args:
        prompt: User input string.
        threshold: Float toxicity score cutoff (0.0 to 1.0) above which prompt is flagged.

    Returns:
        dict containing:
            is_safe (bool): True if prompt is safe to process, False if blocked.
            toxic_score (float): Model probability of toxic class [0.0, 1.0].
            non_toxic_score (float): Model probability of non-toxic class [0.0, 1.0].
            predicted_label (str): "toxic" or "non-toxic".
            threshold (float): Threshold used for the decision.
            reason (str): Human-readable justification.
            error (str or None): Error message if screening failed.
    """
    cleaned = (prompt or "").strip()
    if not cleaned:
        return {
            "is_safe": False,
            "toxic_score": 0.0,
            "non_toxic_score": 0.0,
            "predicted_label": "empty",
            "threshold": threshold,
            "reason": "Prompt is empty. Please enter a descriptive text prompt.",
            "error": "Empty prompt."
        }

    try:
        tokenizer, model = get_safety_classifier()
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # martin-ha/toxic-comment-model: LABEL_0 = non-toxic, LABEL_1 = toxic
        non_toxic_prob = float(probs[0].item())
        toxic_prob = float(probs[1].item())

        is_safe = toxic_prob < threshold
        predicted_label = "toxic" if toxic_prob >= 0.5 else "non-toxic"

        if is_safe:
            reason = f"Prompt passed safety screening (Toxicity: {toxic_prob:.1%}, Threshold: {threshold:.1%})."
        else:
            reason = (
                f"Prompt flagged as potentially toxic or harmful (Toxicity confidence: {toxic_prob:.1%}, "
                f"exceeding safety threshold {threshold:.1%}). Image generation blocked."
            )

        return {
            "is_safe": is_safe,
            "toxic_score": toxic_prob,
            "non_toxic_score": non_toxic_prob,
            "predicted_label": predicted_label,
            "threshold": threshold,
            "reason": reason,
            "error": None
        }

    except Exception as e:
        logger.error(f"Error during prompt safety screening: {e}")
        # Fail closed: never silently pass a prompt if safety check fails
        return {
            "is_safe": False,
            "toxic_score": 1.0,
            "non_toxic_score": 0.0,
            "predicted_label": "error",
            "threshold": threshold,
            "reason": f"Safety screening could not be completed ({str(e)}). Request halted for security.",
            "error": str(e)
        }
