"""
Vision-Language evaluation module using CLIP (openai/clip-vit-base-patch32).
Computes semantic alignment (cosine similarity) between user text prompts
and synthesized images from BigGAN-deep-128.
"""

import logging
import gc
from typing import Dict, Any
import torch
from PIL import Image
import streamlit as st
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

CLIP_MODEL_REPO = "openai/clip-vit-base-patch32"

def _load_clip_model_raw():
    """
    Downloads/loads the CLIP processor and vision-language model onto CPU.
    First tries online check (for fresh downloads), and falls back to cached
    local files if offline (e.g. within restricted sandboxes).
    Returns:
        tuple: (processor, model)
    """
    logger.info(f"Loading CLIP model from {CLIP_MODEL_REPO}...")
    try:
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_REPO)
        model = CLIPModel.from_pretrained(CLIP_MODEL_REPO)
    except Exception as net_err:
        error_text = str(net_err).lower()
        if isinstance(net_err, MemoryError) or any(
            phrase in error_text
            for phrase in ("out of memory", "not enough memory", "cannot allocate memory")
        ):
            raise
        logger.warning(f"Online load failed ({net_err}). Attempting local_files_only=True cache load...")
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_REPO, local_files_only=True)
        model = CLIPModel.from_pretrained(CLIP_MODEL_REPO, local_files_only=True)

    model.to("cpu").eval()
    return processor, model


@st.cache_resource(show_spinner="Loading CLIP evaluation model (ViT-B/32)...")
def _cached_clip_evaluator():
    return _load_clip_model_raw()


def get_clip_evaluator():
    """
    Retrieves the CLIP evaluator (processor, model).
    The model is loaded only when an image is explicitly evaluated.
    """
    return _cached_clip_evaluator()


def release_clip_evaluator():
    """Release the cached evaluator after an explicit CLIP evaluation."""
    _cached_clip_evaluator.clear()
    gc.collect()


def interpret_clip_score(score: float) -> str:
    """
    Provides a standardized qualitative interpretation of CLIP ViT-B/32 cosine similarity scores.
    Typical raw text-image cosine similarities for natural imagery range between 0.15 and 0.35.
    """
    if score >= 0.28:
        return "Strong Semantic Alignment"
    elif score >= 0.20:
        return "Moderate Semantic Alignment"
    elif score >= 0.12:
        return "Weak / Partial Semantic Alignment"
    else:
        return "Low Semantic Alignment"


def compute_clip_similarity(
    prompt: str,
    image: Image.Image
) -> Dict[str, Any]:
    """
    Calculates cosine similarity between a text prompt and an image using CLIP.

    Args:
        prompt: Natural language prompt string.
        image: PIL Image instance (RGB).

    Returns:
        dict containing:
            similarity (float or None): Cosine similarity value [-1.0, 1.0].
            prompt (str): Evaluated text.
            interpretation (str): Qualitative explanation.
            explanation (str): Clarification that score is alignment, not accuracy percentage.
            error (str or None): Error description if evaluation failed.
    """
    cleaned_prompt = (prompt or "").strip()
    if not cleaned_prompt:
        return {
            "similarity": None,
            "prompt": prompt,
            "interpretation": "N/A",
            "explanation": "No text prompt provided for evaluation.",
            "error": "Prompt cannot be empty."
        }

    if not isinstance(image, Image.Image):
        return {
            "similarity": None,
            "prompt": prompt,
            "interpretation": "N/A",
            "explanation": "Provided object is not a valid PIL Image.",
            "error": "Invalid image object."
        }

    try:
        # Ensure image is in RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        processor, model = get_clip_evaluator()
        inputs = processor(text=[cleaned_prompt], images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = model(**inputs)
            # L2-normalize text and image projection embeddings
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            cosine_sim = (text_embeds * image_embeds).sum(dim=-1).item()

        score = float(cosine_sim)
        interpretation = interpret_clip_score(score)

        explanation = (
            "CLIP similarity is a vision-language cosine similarity score [-1.0 to +1.0] "
            "measuring semantic alignment in ViT-B/32 joint embedding space. "
            "It is not an accuracy percentage and does not measure photorealism or BigGAN synthesis fidelity."
        )

        return {
            "similarity": score,
            "prompt": cleaned_prompt,
            "interpretation": interpretation,
            "explanation": explanation,
            "error": None
        }

    except Exception as e:
        logger.error(f"Error computing CLIP similarity: {e}")
        return {
            "similarity": None,
            "prompt": prompt,
            "interpretation": "Error",
            "explanation": f"CLIP evaluation failed: {str(e)}",
            "error": str(e)
        }
