"""
ImageNet-1K Taxonomy Management and Search Utilities.
"""

import os
import json
from difflib import SequenceMatcher

ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
CLASSES_PATH = os.path.join(ASSETS_DIR, "imagenet_classes.json")

_CACHED_CLASSES = None


def load_imagenet_classes():
    """
    Loads the canonical 1,000 ImageNet-1K classes.
    Returns:
        list[dict]: List of class metadata dictionaries containing class_id, wnid, name, display_name.
    """
    global _CACHED_CLASSES
    if _CACHED_CLASSES is not None:
        return _CACHED_CLASSES

    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"ImageNet classes taxonomy file not found at: {CLASSES_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        _CACHED_CLASSES = json.load(f)

    if len(_CACHED_CLASSES) != 1000:
        raise ValueError(f"Expected 1,000 ImageNet classes, but loaded {len(_CACHED_CLASSES)}.")

    return _CACHED_CLASSES


def validate_class_id(class_id):
    """
    Validates whether class_id is a valid ImageNet class index (0 to 999).

    Args:
        class_id (any): The candidate class ID.

    Returns:
        tuple[bool, dict | None]: (True, class_dict) if valid, (False, None) otherwise.
    """
    try:
        idx = int(class_id)
    except (ValueError, TypeError):
        return False, None

    if 0 <= idx < 1000:
        classes = load_imagenet_classes()
        return True, classes[idx]
    return False, None


def get_class_by_id(class_id):
    """
    Retrieves class metadata for a given class ID.
    Raises ValueError if class_id is out of range.
    """
    is_valid, class_info = validate_class_id(class_id)
    if not is_valid:
        raise ValueError(f"Class ID {class_id} is out of ImageNet range [0, 999].")
    return class_info


def fuzzy_search_imagenet(query, top_k=5):
    """
    Searches ImageNet classes for keywords or closest semantic string matches.
    Useful as an offline fallback when Groq API is unavailable.

    Args:
        query (str): User text query.
        top_k (int): Number of top matches to return.

    Returns:
        list[dict]: Top matching class dictionaries ordered by relevance score.
    """
    classes = load_imagenet_classes()
    query_clean = query.strip().lower()

    if not query_clean:
        return classes[:top_k]

    scored = []
    query_tokens = set(query_clean.split())

    for c in classes:
        name_clean = c["name"].lower()
        name_tokens = set(name_clean.split())

        # Exact substring match bonus
        score = 0.0
        if query_clean in name_clean:
            score += 0.6
        elif name_clean in query_clean:
            score += 0.5

        # Token intersection
        token_overlap = len(query_tokens.intersection(name_tokens))
        if token_overlap > 0:
            score += 0.4 * token_overlap

        # String similarity ratio
        ratio = SequenceMatcher(None, query_clean, name_clean).ratio()
        score += 0.3 * ratio

        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]
