"""
Comparison grid generation service for Class-Conditional Image Studio using BigGAN.
Executes multi-prompt and multi-truncation generation sequentially for low memory footprint.
"""

import time
import random
import torch
from PIL import Image

from utils.groq_client import map_prompt_to_imagenet
from utils.imagenet_utils import validate_class_id, fuzzy_search_imagenet
from utils.sampling import generate_truncated_noise, prepare_class_vector, tensor_to_pil
from models.loader import get_cached_biggan_model
from models.safety_classifier import (
    check_prompt_safety,
    DEFAULT_SAFETY_THRESHOLD,
    release_safety_classifier,
)


DEFAULT_TRUNCATIONS = [0.2, 0.4, 0.8]


def generate_single_image(model, class_id, truncation, seed=None):
    """
    Generates a single 128x128 image for a given class ID and truncation.
    Releases tensor references immediately to preserve CPU memory.

    Args:
        model: BigGAN model instance in eval mode.
        class_id (int): Valid ImageNet class index (0-999).
        truncation (float): Truncation threshold in (0, 1].
        seed (int, optional): Random seed.

    Returns:
        tuple[Image.Image, float]: (PIL image, generation latency in seconds)
    """
    t0 = time.perf_counter()
    noise_tensor = generate_truncated_noise(
        batch_size=1, dim_z=model.config.z_dim, truncation=truncation, seed=seed
    )
    class_tensor = prepare_class_vector(class_id, num_classes=model.config.num_classes)

    with torch.no_grad():
        output_tensor = model(noise_tensor, class_tensor, truncation)

    pil_img = tensor_to_pil(output_tensor)
    latency = time.perf_counter() - t0

    # Explicitly release tensor references
    del output_tensor
    del noise_tensor
    del class_tensor

    return pil_img, latency


def run_comparison_grid(
    model,
    prompts,
    truncations=None,
    base_seed=None,
    safety_threshold=DEFAULT_SAFETY_THRESHOLD,
    progress_callback=None,
):
    """
    Processes multiple natural-language prompts across multiple truncation values sequentially.
    Screens each prompt for toxicity using DistilBERT before Groq mapping and BigGAN synthesis.

    Args:
        model: Optional preloaded BigGAN model instance.
        prompts (list[str]): List of prompt strings (e.g. 5 prompts).
        truncations (list[float]): List of truncation values (default: [0.2, 0.4, 0.8]).
        base_seed (int, optional): Base seed to generate reproducible distinct seeds.
        safety_threshold (float): Cutoff above which prompt is flagged and blocked.
        progress_callback (callable, optional): Callback func(current_step, total_steps, status_text).

    Returns:
        dict: {
            "rows": list of row dictionaries with prompt, class info, safety, and list of image items,
            "total_images": int,
            "total_time": float,
            "success_count": int,
            "failure_count": int
        }
    """
    if truncations is None:
        truncations = DEFAULT_TRUNCATIONS

    total_steps = len(prompts) * len(truncations)
    current_step = 0
    start_all = time.perf_counter()

    rows = []
    success_count = 0
    failure_count = 0

    if base_seed is None:
        base_seed = random.randint(10000, 999999)

    try:
        for row_idx, raw_prompt in enumerate(prompts):
            prompt_text = raw_prompt.strip() if raw_prompt else ""
            row_data = {
                "row_index": row_idx,
                "original_prompt": prompt_text,
                "class_id": None,
                "class_name": None,
                "wnid": None,
                "mapping_source": None,
                "mapping_explanation": None,
                "safety": None,
                "error": None,
                "items": [],
            }

            # Handle empty prompt
            if not prompt_text:
                row_data["error"] = "Empty prompt provided."
                rows.append(row_data)
                failure_count += len(truncations)
                if progress_callback:
                    current_step += len(truncations)
                    progress_callback(current_step, total_steps, f"Skipped empty prompt {row_idx + 1}")
                continue

            # Screen and map every prompt before loading BigGAN.
            if progress_callback:
                progress_callback(current_step, total_steps, f"Screening prompt {row_idx + 1}/{len(prompts)} for safety...")

            safety_res = check_prompt_safety(prompt_text, threshold=safety_threshold)
            row_data["safety"] = safety_res

            if not safety_res["is_safe"]:
                row_data["error"] = f"Safety Blocked: {safety_res['reason']}"
                rows.append(row_data)
                failure_count += len(truncations)
                if progress_callback:
                    current_step += len(truncations)
                    progress_callback(
                        current_step,
                        total_steps,
                        f"Prompt {row_idx + 1} blocked by safety screening ({safety_res['toxic_score']:.1%} toxic)",
                    )
                continue

            # Map prompt to ImageNet class (Groq first, then local fallback).
            class_id = None
            class_name = None
            source_mode = None
            explanation = None

            try:
                success, groq_res, err_msg = map_prompt_to_imagenet(prompt_text)
                if success and groq_res:
                    class_id = groq_res["class_id"]
                    class_name = groq_res["class_name"]
                    explanation = groq_res.get("explanation", "")
                    source_mode = f"Groq LLM ({groq_res.get('model_used', 'LLM')})"
                else:
                    matches = fuzzy_search_imagenet(prompt_text, top_k=1)
                    if matches:
                        best = matches[0]
                        class_id = best["class_id"]
                        class_name = best["name"]
                        explanation = f"Offline fallback selected '{best['name']}'"
                        source_mode = "Local Keyword Search (Fallback)"
                    else:
                        row_data["error"] = f"Could not map prompt to ImageNet: {err_msg}"
                        rows.append(row_data)
                        failure_count += len(truncations)
                        continue
            except Exception as e:
                matches = fuzzy_search_imagenet(prompt_text, top_k=1)
                if matches:
                    best = matches[0]
                    class_id = best["class_id"]
                    class_name = best["name"]
                    explanation = f"Exception fallback selected '{best['name']}'"
                    source_mode = "Local Keyword Search (Fallback)"
                else:
                    row_data["error"] = f"Mapping error: {str(e)}"
                    rows.append(row_data)
                    failure_count += len(truncations)
                    continue

            # Validate class ID.
            is_valid, class_info = validate_class_id(class_id)
            if not is_valid:
                row_data["error"] = f"Invalid class ID {class_id}."
                rows.append(row_data)
                failure_count += len(truncations)
                continue

            row_data["class_id"] = class_id
            row_data["class_name"] = class_info["name"]
            row_data["wnid"] = class_info.get("wnid", "")
            row_data["mapping_source"] = source_mode
            row_data["mapping_explanation"] = explanation
            rows.append(row_data)

    finally:
        release_safety_classifier()

    if model is None and any(row["class_id"] is not None for row in rows):
        model = get_cached_biggan_model()

    # Generate images sequentially after releasing the safety model.
    for row_data in rows:
        class_id = row_data["class_id"]
        if class_id is None:
            continue
        row_idx = row_data["row_index"]
        class_name = row_data["class_name"]

        for col_idx, trunc in enumerate(truncations):
            current_step += 1
            # Ensure unique, reproducible seeds per cell
            cell_seed = (base_seed + (row_idx * 1000) + (col_idx * 137)) % 1000000

            if progress_callback:
                progress_callback(
                    current_step,
                    total_steps,
                    f"Generating prompt {row_idx + 1}/{len(prompts)} ('{class_name}') at ψ={trunc}...",
                )

            item = {
                "truncation": trunc,
                "seed": cell_seed,
                "image": None,
                "generation_time": None,
                "clip_score": None,
                "error": None,
            }

            try:
                img, gen_time = generate_single_image(
                    model=model,
                    class_id=class_id,
                    truncation=trunc,
                    seed=cell_seed,
                )
                item["image"] = img
                item["generation_time"] = gen_time
                success_count += 1
            except Exception as e:
                item["error"] = str(e)
                failure_count += 1

            row_data["items"].append(item)

    total_time = time.perf_counter() - start_all

    return {
        "rows": rows,
        "truncations": truncations,
        "base_seed": base_seed,
        "total_images": success_count,
        "total_time": total_time,
        "success_count": success_count,
        "failure_count": failure_count,
    }
