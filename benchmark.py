"""
Reproducible benchmark for the existing BigGAN Image Studio pipeline.

Run:
    python benchmark.py

The script uses the project's existing loaders and generation services.
External measurements are marked unavailable if a service or API is not configured.
"""

from __future__ import annotations

import csv
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from models.clip_evaluator import compute_clip_similarity
from models.loader import load_biggan_model
from models.safety_classifier import check_prompt_safety
from utils.comparison_service import run_comparison_grid
from utils.groq_client import get_groq_config, map_prompt_to_imagenet
from utils.sampling import generate_truncated_noise, prepare_class_vector, tensor_to_pil


OUTPUT_CSV = ROOT / "docs" / "BENCHMARK_RESULTS.csv"


def peak_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    if hasattr(process, "memory_info"):
        return process.memory_info().rss / (1024 * 1024)
    return float("nan")


def environment_rows() -> list[dict]:
    cpu = platform.processor() or "Not reported"
    ram_gb = psutil.virtual_memory().total / (1024**3)
    rows = [
        {
            "metric": "hardware.cpu",
            "value": cpu,
            "unit": "string",
            "status": "measured",
            "details": "platform.processor()",
        },
        {
            "metric": "hardware.ram",
            "value": f"{ram_gb:.2f}",
            "unit": "GB",
            "status": "measured",
            "details": "psutil.virtual_memory().total",
        },
        {
            "metric": "software.os",
            "value": platform.platform(),
            "unit": "string",
            "status": "measured",
            "details": "platform.platform()",
        },
        {
            "metric": "software.python",
            "value": platform.python_version(),
            "unit": "string",
            "status": "measured",
            "details": "platform.python_version()",
        },
        {
            "metric": "software.torch",
            "value": torch.__version__,
            "unit": "string",
            "status": "measured",
            "details": "torch.__version__",
        },
        {
            "metric": "software.transformers",
            "value": __import__("transformers").__version__,
            "unit": "string",
            "status": "measured",
            "details": "transformers.__version__",
        },
        {
            "metric": "compute.device",
            "value": "cuda" if torch.cuda.is_available() else "cpu",
            "unit": "string",
            "status": "measured",
            "details": "torch.cuda.is_available()",
        },
    ]
    return rows


def append_row(
    rows: list[dict],
    metric: str,
    value,
    unit: str,
    status: str,
    details: str,
) -> None:
    rows.append(
        {
            "metric": metric,
            "value": value,
            "unit": unit,
            "status": status,
            "details": details,
        }
    )


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    measured_at = datetime.now(timezone.utc).isoformat()
    rows = environment_rows()
    process_peak = peak_rss_mb()

    cold_start = time.perf_counter()
    model = load_biggan_model()
    cold_load_seconds = time.perf_counter() - cold_start
    append_row(
        rows,
        "biggan_cold_start_loading_time",
        f"{cold_load_seconds:.6f}",
        "seconds",
        "measured",
        "Existing load_biggan_model(); fresh process",
    )
    append_row(
        rows,
        "biggan_loaded_peak_rss",
        f"{peak_rss_mb():.2f}",
        "MB",
        "measured",
        "Peak/resident process RSS after BigGAN loading",
    )

    generation_times = []
    first_image = None
    for seed in (42, 43, 44):
        noise = generate_truncated_noise(1, model.config.z_dim, 0.5, seed)
        class_vector = prepare_class_vector(207, model.config.num_classes)
        start = time.perf_counter()
        with torch.no_grad():
            output = model(noise, class_vector, 0.5)
        first_image = tensor_to_pil(output)
        elapsed = time.perf_counter() - start
        generation_times.append(elapsed)

    append_row(
        rows,
        "single_image_generation_time",
        f"{generation_times[0]:.6f}",
        "seconds",
        "measured",
        "Existing forward pass and PIL conversion path; seed 42",
    )
    append_row(
        rows,
        "average_generation_time",
        f"{sum(generation_times) / len(generation_times):.6f}",
        "seconds",
        "measured",
        "Mean of seeds 42, 43, 44",
    )

    grid_start = time.perf_counter()
    grid = run_comparison_grid(
        model=model,
        prompts=[
            "golden retriever",
            "sports car",
            "great white shark",
            "espresso",
            "pizza",
        ],
        truncations=[0.2, 0.4, 0.8],
        base_seed=42,
    )
    grid_elapsed = time.perf_counter() - grid_start
    append_row(
        rows,
        "15_image_comparison_grid_execution_time",
        f"{grid_elapsed:.6f}",
        "seconds",
        "measured",
        f"Existing run_comparison_grid(); successful images={grid['success_count']}",
    )

    api_key, groq_model = get_groq_config()
    if api_key:
        groq_start = time.perf_counter()
        success, result, error = map_prompt_to_imagenet(
            "A golden retriever dog sitting on green grass"
        )
        groq_elapsed = time.perf_counter() - groq_start
        append_row(
            rows,
            "groq_class_mapping_latency",
            f"{groq_elapsed:.6f}",
            "seconds",
            "measured",
            f"success={success}; model={groq_model}; error={error or 'none'}",
        )
    else:
        append_row(
            rows,
            "groq_class_mapping_latency",
            "",
            "seconds",
            "not_measured",
            "GROQ_API_KEY is not configured",
        )

    safety_start = time.perf_counter()
    safety = check_prompt_safety("A golden retriever dog sitting on green grass")
    safety_elapsed = time.perf_counter() - safety_start
    append_row(
        rows,
        "distilbert_inference_latency",
        f"{safety_elapsed:.6f}",
        "seconds",
        "measured" if safety.get("error") is None else "not_measured",
        f"toxic_score={safety.get('toxic_score')}; error={safety.get('error')}",
    )

    if first_image is not None:
        clip_start = time.perf_counter()
        clip = compute_clip_similarity(
            "A golden retriever dog sitting on green grass",
            first_image,
        )
        clip_elapsed = time.perf_counter() - clip_start
        append_row(
            rows,
            "clip_evaluation_latency",
            f"{clip_elapsed:.6f}",
            "seconds",
            "measured" if clip.get("error") is None else "not_measured",
            f"similarity={clip.get('similarity')}; error={clip.get('error')}",
        )

    append_row(
        rows,
        "peak_process_rss",
        f"{peak_rss_mb():.2f}",
        "MB",
        "measured",
        "Resident process RSS at end of benchmark",
    )
    append_row(
        rows,
        "benchmark.measurement_timestamp_utc",
        measured_at,
        "string",
        "measured",
        "Run timestamp",
    )
    append_row(
        rows,
        "benchmark.process_peak_rss_initial",
        f"{process_peak:.2f}",
        "MB",
        "measured",
        "Resident process RSS before model loading",
    )

    fieldnames = ["metric", "value", "unit", "status", "details", "measured_at"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row["measured_at"] = measured_at
            writer.writerow(row)

    print(f"Wrote benchmark results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
