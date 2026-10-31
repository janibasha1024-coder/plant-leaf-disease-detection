"""Resize, denoise, and contrast steps."""

from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP,
    CLAHE_TILE,
    MAX_SIDE_PX,
)


def resize_max_side(image: np.ndarray, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    """Shrink the longest side to max_side. Smaller photographs stay as they are."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image.copy()
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray) -> np.ndarray:
    """Reduce speckle while keeping leaf edges."""
    return cv2.bilateralFilter(
        image,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Even out contrast on the lightness channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    adjusted = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((adjusted, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def variance_of_laplacian(image: np.ndarray) -> float:
    """Higher values mean a sharper photograph."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
