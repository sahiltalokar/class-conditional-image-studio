"""
Groq API integration for natural-language prompt to ImageNet-1K class mapping.
Includes strict JSON Schema enforcement, reasoning-model optimization for openai/gpt-oss-20b,
reconciliation of class IDs and names, and descriptive error handling.
"""

import os
import re
import json
import logging
from groq import Groq, RateLimitError, AuthenticationError, APIConnectionError, APIStatusError
import streamlit as st

from utils.imagenet_utils import load_imagenet_classes, validate_class_id, fuzzy_search_imagenet

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-oss-20b"
PLACEHOLDER_API_KEYS = {"your_groq_api_key", "gsk_your_groq_api_key_here"}

# Strict JSON Schema following Groq structured outputs specification
STRICT_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "imagenet_class_mapping",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "class_id": {
                    "type": "integer",
                    "description": "ImageNet-1K class ID between 0 and 999",
                    "minimum": 0,
                    "maximum": 999,
                },
                "class_name": {
                    "type": "string",
                    "description": "The exact name of the selected ImageNet class",
                },
                "explanation": {
                    "type": "string",
                    "description": "Short 1-2 sentence explanation of why this class was chosen",
                },
            },
            "required": ["class_id", "class_name", "explanation"],
            "additionalProperties": False,
        },
    },
}


def get_groq_config():
    """
    Retrieves Groq API key and model name from Streamlit Secrets or Environment Variables.
    Returns:
        tuple[str | None, str]: (api_key, model_name)
    """
    api_key = None
    model_name = DEFAULT_MODEL

    # Read Streamlit Secrets first; use environment variables as a local fallback.
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
        if "GROQ_MODEL" in st.secrets and st.secrets["GROQ_MODEL"]:
            model_name = st.secrets["GROQ_MODEL"]
    except Exception:
        pass

    if not api_key or not api_key.strip() or api_key.strip().lower() in PLACEHOLDER_API_KEYS:
        api_key = os.environ.get("GROQ_API_KEY") or None
    if not model_name or model_name == DEFAULT_MODEL:
        model_name = os.environ.get("GROQ_MODEL") or DEFAULT_MODEL

    if api_key and (
        not api_key.strip() or api_key.strip().lower() in PLACEHOLDER_API_KEYS
    ):
        api_key = None

    return api_key, model_name


def _sanitize_error_message(error_text):
    """
    Removes any accidentally exposed API keys or sensitive strings from error messages.
    """
    if not error_text:
        return "Unknown error"
    # Redact any gsk_ API keys
    sanitized = re.sub(r"gsk_[A-Za-z0-9_\-]+", "[REDACTED_API_KEY]", str(error_text))
    # Redact bearer tokens
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9_\.\-]+", "Bearer [REDACTED]", sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


def _reconcile_class_mapping(raw_class_id, raw_class_name):
    """
    Validates and reconciles the LLM's returned class_id and class_name against
    the official 1,000 ImageNet categories.
    Handles:
    - String integers ("207" -> 207)
    - WordNet synset IDs ("n02099601" -> 207)
    - Discrepancies where class_name is accurate but class_id was misindexed
    """
    classes = load_imagenet_classes()
    name_to_class = {c["name"].lower(): c for c in classes}
    wnid_to_class = {c["wnid"].lower(): c for c in classes}

    cleaned_name = str(raw_class_name).strip().lower() if raw_class_name else ""

    # Check if raw_class_id is a WordNet ID (e.g. 'n02099601')
    if isinstance(raw_class_id, str) and raw_class_id.lower() in wnid_to_class:
        target = wnid_to_class[raw_class_id.lower()]
        return True, target["class_id"], target["name"]

    # Try parsing class_id as integer
    parsed_id = None
    try:
        parsed_id = int(raw_class_id)
    except (ValueError, TypeError):
        parsed_id = None

    # Check if class_name matches an exact ImageNet class
    if cleaned_name in name_to_class:
        canonical_class = name_to_class[cleaned_name]
        # If parsed_id doesn't match canonical ID for this name, prioritize canonical ID
        if parsed_id != canonical_class["class_id"]:
            logger.info(
                f"Reconciled class ID: model returned id={parsed_id} but name='{cleaned_name}', "
                f"using canonical id={canonical_class['class_id']}"
            )
        return True, canonical_class["class_id"], canonical_class["name"]

    # If class_id is a valid integer between 0 and 999
    if parsed_id is not None and 0 <= parsed_id < 1000:
        target = classes[parsed_id]
        return True, target["class_id"], target["name"]

    # If name was slightly varied (e.g. plural or hyphenated), check if fuzzy match is high confidence
    if cleaned_name:
        matches = fuzzy_search_imagenet(cleaned_name, top_k=1)
        if matches:
            best = matches[0]
            from difflib import SequenceMatcher
            sim = SequenceMatcher(None, cleaned_name, best["name"].lower()).ratio()
            if sim >= 0.7 or cleaned_name in best["name"].lower() or best["name"].lower() in cleaned_name:
                return True, best["class_id"], best["name"]

    return False, None, None


def map_prompt_to_imagenet(prompt, api_key=None, model_name=None):
    """
    Calls the Groq API to map a user's natural language text prompt to the
    closest valid ImageNet-1K class ID (0 to 999).
    Uses strict JSON Schema enforcement and optimizes token parameters for reasoning models.

    Args:
        prompt (str): Natural language description.
        api_key (str, optional): Groq API key.
        model_name (str, optional): Groq model name.

    Returns:
        tuple[bool, dict | None, str | None]: (success, result_dict, error_message)
    """
    if not prompt or not prompt.strip():
        return False, None, "The entered prompt is empty. Please describe an object or animal to generate."

    configured_key, configured_model = get_groq_config()
    final_key = configured_key if api_key is None else api_key
    final_model = configured_model if model_name is None else model_name

    if not final_key or final_key.strip().lower() in PLACEHOLDER_API_KEYS:
        return (
            False,
            None,
            "Groq API key not configured. Add GROQ_API_KEY to .streamlit/secrets.toml or use Manual Class Selection.",
        )

    system_prompt = (
        "You are an ImageNet-1K taxonomy specialist for a generative AI system.\n"
        "Your task: Given a user's text prompt, select the single closest matching ImageNet-1K class ID (integer from 0 to 999).\n"
        "Rules:\n"
        "1. ImageNet-1K has 1,000 specific classes across animals (e.g. 207: 'golden retriever', 2: 'great white shark'), "
        "vehicles (e.g. 817: 'sports car'), food (e.g. 963: 'pizza', 923: 'plate'), "
        "household items (e.g. 504: 'coffee mug', 553: 'file', 725: 'pitcher'), and scenes.\n"
        "2. Identify the core subject of the user prompt and match it to its closest ImageNet-1K category.\n"
        "3. You must return valid JSON with class_id (integer 0-999), class_name, and a 1-sentence explanation."
    )

    # Determine if target model is a reasoning model (such as openai/gpt-oss-20b)
    is_reasoning_model = any(k in final_model.lower() for k in ["gpt-oss", "o1", "o3", "deepseek-r1", "qwq"])

    # Build request kwargs
    request_kwargs = {
        "model": final_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User prompt: {prompt.strip()}"},
        ],
        "temperature": 0.1,
        "max_completion_tokens": 2048,  # Generous headroom for reasoning models + JSON output
    }

    # Constrain reasoning chain length for reasoning models to prevent timeout and validation errors
    if is_reasoning_model:
        request_kwargs["reasoning_effort"] = "low"

    # Attempt 1: Strict JSON Schema mode
    request_kwargs["response_format"] = STRICT_JSON_SCHEMA

    client = Groq(api_key=final_key, timeout=12.0, max_retries=0)

    try:
        response = client.chat.completions.create(**request_kwargs)
        raw_content = response.choices[0].message.content.strip()

    except APIStatusError as e:
        status_code = getattr(e, "status_code", 400)
        error_detail = None

        # Extract specific error message from Groq response body
        if hasattr(e, "body") and isinstance(e.body, dict):
            err_dict = e.body.get("error", {})
            if isinstance(err_dict, dict):
                error_detail = err_dict.get("message")
        if not error_detail:
            error_detail = getattr(e, "message", str(e))

        clean_error = _sanitize_error_message(error_detail)
        logger.warning(f"Groq primary request failed with Status {status_code}: {clean_error}")

        # If HTTP 400 occurred (e.g. json_schema rejected or parameter mismatch), attempt fallback payload
        if status_code == 400:
            try:
                logger.info("Attempting fallback with standard json_object response format...")
                fallback_kwargs = {
                    "model": final_model,
                    "messages": [
                        {"role": "system", "content": system_prompt + "\nRespond with a valid JSON object."},
                        {"role": "user", "content": f"User prompt: {prompt.strip()}"},
                    ],
                    "temperature": 0.2,
                    "max_completion_tokens": 1024,
                    "response_format": {"type": "json_object"},
                }
                response = client.chat.completions.create(**fallback_kwargs)
                raw_content = response.choices[0].message.content.strip()
            except Exception as fb_err:
                fb_clean = _sanitize_error_message(getattr(fb_err, "message", str(fb_err)))
                logger.error(f"Groq fallback failed: {fb_clean}")
                return False, None, f"Groq request rejected (Status 400): {clean_error}"
        else:
            return False, None, f"Groq API error (Status {status_code}): {clean_error}"

    except RateLimitError as e:
        logger.warning("Groq rate limit reached.")
        return False, None, "Groq API rate limit reached (HTTP 429). Falling back to local ImageNet class search."
    except AuthenticationError as e:
        logger.error("Groq authentication failed.")
        return False, None, "Invalid GROQ_API_KEY. Please verify Streamlit Secrets or the GROQ_API_KEY environment variable."
    except APIConnectionError as e:
        logger.warning(f"Groq API connection failed: {_sanitize_error_message(e)}")
        return False, None, "Network error: Unable to reach Groq API. Please check internet connection."
    except Exception as e:
        clean_msg = _sanitize_error_message(str(e))
        logger.error(f"Unexpected error in Groq class mapping: {clean_msg}")
        return False, None, f"Class mapping failed: {clean_msg}"

    # Parse and validate returned JSON
    try:
        # Strip markdown fences if present
        clean_json = raw_content
        if clean_json.startswith("```json"):
            clean_json = clean_json[7:]
        if clean_json.startswith("```"):
            clean_json = clean_json[3:]
        if clean_json.endswith("```"):
            clean_json = clean_json[:-3]
        clean_json = clean_json.strip()

        parsed = json.loads(clean_json)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode failure on content '{raw_content}': {e}")
        return False, None, "Failed to parse structured JSON from Groq. Falling back to local ImageNet search."

    if not isinstance(parsed, dict) or "class_id" not in parsed:
        return False, None, "Groq response was missing required 'class_id' field."

    # Reconcile class_id and class_name against taxonomy
    is_valid, resolved_id, resolved_name = _reconcile_class_mapping(
        parsed.get("class_id"), parsed.get("class_name")
    )

    if not is_valid:
        return (
            False,
            None,
            f"Returned ImageNet class ID ({parsed.get('class_id')}) or name ({parsed.get('class_name')}) could not be resolved to ImageNet-1K.",
        )

    result = {
        "class_id": resolved_id,
        "class_name": resolved_name,
        "explanation": parsed.get(
            "explanation", f"Mapped to ImageNet class {resolved_id} ({resolved_name})."
        ),
        "model_used": final_model,
    }
    return True, result, None
