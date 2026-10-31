"""Checks for uploaded leaf photographs."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.config import MAX_IMAGE_BYTES, MAX_PIXELS, MIN_SIDE_PX

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageValidationError(ValueError):
    """The upload is missing, too large, or not a readable JPEG or PNG."""


def load_image(filename: str, data: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> np.ndarray:
    """Return a BGR image, or raise ImageValidationError with a message for the user."""
    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError("Choose a JPG, JPEG, or PNG file.")
    if not data:
        raise ImageValidationError("This file is empty.")
    if len(data) > max_bytes:
        limit_mb = max(1, max_bytes // (1024 * 1024))
        raise ImageValidationError(f"This file is larger than {limit_mb} MB. Use a smaller image.")

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise ImageValidationError("The file content is not a JPEG or PNG image.")
            rgb = image.convert("RGB")
            width, height = rgb.size
            if width < MIN_SIDE_PX or height < MIN_SIDE_PX:
                raise ImageValidationError("This image is smaller than 64×64 pixels.")
            if width * height > MAX_PIXELS:
                raise ImageValidationError("This image has too many pixels. Use a smaller photo.")
            pixels = np.array(rgb)
    except ImageValidationError:
        raise
    except Exception as exc:
        raise ImageValidationError("This file could not be read as an image.") from exc

    return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
