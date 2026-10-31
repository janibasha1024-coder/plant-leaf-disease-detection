"""Separate the main leaf region from the background."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import (
    GREEN_HUE_HIGH,
    GREEN_HUE_LOW,
    GREEN_SAT_MIN,
    GREEN_VAL_MIN,
    MIN_CONTOUR_AREA,
    MIN_LEAF_AREA_RATIO,
    YELLOW_HUE_HIGH,
    YELLOW_HUE_LOW,
    YELLOW_SAT_MIN,
    YELLOW_VAL_MIN,
)


@dataclass
class LeafMask:
    mask: np.ndarray
    found: bool
    multiple_regions: bool
    area_ratio: float


def _excess_green(image: np.ndarray) -> np.ndarray:
    blue, green, red = cv2.split(image.astype(np.int16))
    excess = 2 * green - red - blue
    excess = np.clip(excess, 0, None)
    if int(excess.max()) == 0:
        return np.zeros(image.shape[:2], dtype=np.uint8)
    scaled = cv2.normalize(excess, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _vegetation_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        (GREEN_HUE_LOW, GREEN_SAT_MIN, GREEN_VAL_MIN),
        (GREEN_HUE_HIGH, 255, 255),
    )
    yellow = cv2.inRange(
        hsv,
        (YELLOW_HUE_LOW, YELLOW_SAT_MIN, YELLOW_VAL_MIN),
        (YELLOW_HUE_HIGH, 255, 255),
    )
    return cv2.bitwise_or(green, yellow)


def segment_leaf(image: np.ndarray) -> LeafMask:
    """Keep the largest vegetation region and fill holes inside its outline."""
    combined = cv2.bitwise_or(_excess_green(image), _vegetation_mask(image))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [contour for contour in contours if cv2.contourArea(contour) >= MIN_CONTOUR_AREA]
    empty = np.zeros(image.shape[:2], dtype=np.uint8)
    if not large:
        return LeafMask(empty, False, False, 0.0)

    chosen = max(large, key=cv2.contourArea)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [chosen], -1, 255, thickness=cv2.FILLED)
    area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if area_ratio < MIN_LEAF_AREA_RATIO:
        return LeafMask(empty, False, len(large) > 1, area_ratio)
    return LeafMask(mask, True, len(large) > 1, area_ratio)
