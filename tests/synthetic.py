"""Photographs drawn in code so the tests have a known shape."""

from __future__ import annotations

import cv2
import numpy as np


def leaf_scene(
    width: int = 420,
    height: int = 420,
    spots: bool = False,
    second_leaf: bool = False,
    blur: float = 0,
) -> np.ndarray:
    image = np.full((height, width, 3), (230, 228, 224), dtype=np.uint8)
    cv2.ellipse(image, (210, 210), (150, 110), 0, 0, 360, (45, 165, 50), thickness=-1)
    if second_leaf:
        cv2.ellipse(image, (70, 70), (40, 28), 0, 0, 360, (40, 150, 45), thickness=-1)
    if spots:
        cv2.circle(image, (200, 210), 34, (25, 45, 115), thickness=-1)
        cv2.circle(image, (250, 180), 14, (30, 80, 190), thickness=-1)
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return image
