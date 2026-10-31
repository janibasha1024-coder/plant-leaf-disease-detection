from io import BytesIO

import pytest
from PIL import Image

from src.validate import ImageValidationError, load_image


def _png(size=(80, 80), color=(20, 140, 40)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (90, 70), (30, 120, 40)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_png_and_jpeg_load():
    png = load_image("leaf.png", _png())
    jpeg = load_image("leaf.jpg", _jpeg())
    assert png.shape == (80, 80, 3)
    assert jpeg.shape == (70, 90, 3)
    assert load_image("LEAF.JPEG", _jpeg()).shape[2] == 3


def test_wrong_extension_is_rejected():
    with pytest.raises(ImageValidationError, match="JPG, JPEG, or PNG"):
        load_image("notes.txt", _png())


def test_gif_content_is_rejected():
    buffer = BytesIO()
    Image.new("RGB", (80, 80), (0, 0, 255)).save(buffer, format="GIF")
    with pytest.raises(ImageValidationError, match="not a JPEG or PNG"):
        load_image("leaf.jpg", buffer.getvalue())


def test_empty_corrupt_tiny_and_large_files_are_rejected():
    with pytest.raises(ImageValidationError, match="empty"):
        load_image("leaf.png", b"")
    with pytest.raises(ImageValidationError, match="could not be read"):
        load_image("leaf.jpg", b"not-a-real-image-file")
    with pytest.raises(ImageValidationError, match="64"):
        load_image("leaf.png", _png(size=(20, 80)))
    with pytest.raises(ImageValidationError, match="larger than"):
        load_image("leaf.jpg", b"x" * 150, max_bytes=100)
