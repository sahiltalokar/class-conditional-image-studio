"""
Sampling and image post-processing utilities for BigGAN.
"""

import numpy as np
import torch
from PIL import Image
from scipy.stats import truncnorm


def generate_truncated_noise(batch_size=1, dim_z=128, truncation=0.5, seed=None):
    """
    Generates a truncated normal noise vector for BigGAN.
    Sampled from standard normal truncated at [-2, 2] and scaled by truncation parameter.

    Args:
        batch_size (int): Number of noise samples to generate.
        dim_z (int): Latent noise dimension (128 for BigGAN-deep-128).
        truncation (float): Truncation threshold in (0, 1].
        seed (int, optional): Random seed for reproducible generation.

    Returns:
        torch.FloatTensor: Tensor of shape (batch_size, dim_z).
    """
    random_state = np.random.RandomState(seed) if seed is not None else None
    values = truncnorm.rvs(-2.0, 2.0, size=(batch_size, dim_z), random_state=random_state).astype(np.float32)
    noise_array = truncation * values
    return torch.from_numpy(noise_array)


def prepare_class_vector(class_id, num_classes=1000, batch_size=1):
    """
    Creates a one-hot class conditioning vector for BigGAN.

    Args:
        class_id (int): Integer class ID in [0, num_classes - 1].
        num_classes (int): Total number of classes (1000 for ImageNet-1K).
        batch_size (int): Batch size.

    Returns:
        torch.FloatTensor: One-hot tensor of shape (batch_size, num_classes).
    """
    if not (0 <= class_id < num_classes):
        raise ValueError(f"Invalid class_id {class_id}. Must be between 0 and {num_classes - 1}.")

    one_hot = torch.zeros(batch_size, num_classes, dtype=torch.float32)
    one_hot[:, class_id] = 1.0
    return one_hot


def tensor_to_pil(tensor_image):
    """
    Converts a BigGAN output tensor [-1, 1] to a valid 128x128 PIL RGB Image.

    Args:
        tensor_image (torch.Tensor): Output tensor of shape (3, H, W) or (1, 3, H, W).

    Returns:
        PIL.Image.Image: 128x128 RGB PIL Image.
    """
    if tensor_image.dim() == 4:
        tensor_image = tensor_image[0]

    # Detach and move to CPU numpy
    arr = tensor_image.detach().cpu().numpy()

    # Transpose from (C, H, W) to (H, W, C)
    arr = arr.transpose((1, 2, 0))

    # Scale from [-1, 1] to [0, 255]
    arr = np.clip(((arr + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")
