"""Imaging artifacts for phantoms: scanner blur (point-spread function) and noise."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter

from ..volume import ImageGeometry


def apply_blur(image: np.ndarray, geometry: ImageGeometry, sigma_mm: float) -> np.ndarray:
    """Isotropic Gaussian blur with standard deviation ``sigma_mm`` (array is ``[k, j, i]``)."""
    if sigma_mm <= 0:
        return image
    si, sj, sk = geometry.spacing
    sigma_voxels = (sigma_mm / sk, sigma_mm / sj, sigma_mm / si)
    return gaussian_filter(image.astype(float), sigma=sigma_voxels, mode="nearest")


def add_noise(image: np.ndarray, sigma_hu: float, seed: Optional[int]) -> np.ndarray:
    """Additive Gaussian noise; deterministic for a given ``seed``."""
    if sigma_hu <= 0:
        return image
    rng = np.random.default_rng(seed)
    return image + rng.normal(0.0, sigma_hu, size=image.shape)


def to_ct_int16(image: np.ndarray) -> np.ndarray:
    """Round to integer HU and clip to the int16 range used by CT."""
    return np.clip(np.rint(image), -32768, 32767).astype(np.int16)
