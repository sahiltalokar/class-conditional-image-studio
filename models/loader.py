"""
Model loading and caching utility for BigGAN-deep-128.
Designed for CPU inference and Streamlit caching.
"""

import os
import json
import logging
import gc
import torch
import streamlit as st
from huggingface_hub import hf_hub_download

from models.biggan import BigGAN, BigGANConfig

logger = logging.getLogger(__name__)

REPO_ID = "osanseviero/BigGAN-deep-128"
CONFIG_FILENAME = "config.json"
WEIGHTS_FILENAME = "pytorch_model.bin"


def get_checkpoint_files():
    """
    Ensures config.json and pytorch_model.bin exist locally.
    First checks local directories (e.g. weights/), then checks Hugging Face cache,
    and downloads from Hugging Face Hub if not found.
    Returns:
        tuple (config_path, weights_path)
    """
    # 1. Check direct local paths in the workspace
    local_weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
    local_cfg = os.path.join(local_weights_dir, CONFIG_FILENAME)
    local_wt = os.path.join(local_weights_dir, WEIGHTS_FILENAME)

    if os.path.exists(local_cfg) and os.path.exists(local_wt):
        logger.info(f"Using local weights from {local_weights_dir}")
        return local_cfg, local_wt

    # 2. Check Hugging Face Hub / download if necessary
    try:
        config_path = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILENAME)
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHTS_FILENAME)
        return config_path, weights_path
    except Exception as e:
        logger.error(f"Failed to retrieve checkpoint from Hugging Face Hub: {e}")
        raise RuntimeError(
            f"BigGAN-deep-128 checkpoint could not be found or downloaded. "
            f"Please ensure network access or place config.json and pytorch_model.bin into ./weights/."
        ) from e


def load_biggan_model():
    """
    Instantiates BigGAN-deep-128 and loads pretrained weights onto CPU.
    Returns:
        BigGAN: Initialized model in eval mode.
    """
    config_path, weights_path = get_checkpoint_files()

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    config = BigGANConfig.from_dict(config_data)
    model = BigGAN(config).to("cpu")

    # Load weights onto CPU
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.warning(f"BigGAN load missing keys: {missing}")
    if unexpected:
        logger.warning(f"BigGAN load unexpected keys: {unexpected}")

    model.eval()
    return model


@st.cache_resource(show_spinner="Loading BigGAN-deep-128 model into memory...")
def get_cached_biggan_model():
    """
    Cached wrapper for Streamlit to prevent reloading the model on every rerun.
    """
    return load_biggan_model()


def release_cached_biggan_model():
    """Release the cached generator before loading another large model."""
    get_cached_biggan_model.clear()
    gc.collect()
