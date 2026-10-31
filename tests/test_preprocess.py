import cv2
import numpy as np

from src.preprocess import apply_clahe, denoise, resize_max_side, variance_of_laplacian


def test_resize_limits_the_long_side_and_keeps_small_images():
    large = np.zeros((2000, 1000, 3), dtype=np.uint8)
    resized = resize_max_side(large, 900)
    assert resized.shape == (900, 450, 3)

    small = np.zeros((100, 80, 3), dtype=np.uint8)
    assert resize_max_side(small, 900).shape == small.shape


def test_denoise_lowers_variance_of_speckle():
    rng = np.random.default_rng(0)
    clean = np.full((90, 90, 3), 120, dtype=np.uint8)
    noise = rng.integers(0, 50, clean.shape, dtype=np.uint8)
    noisy = cv2.add(clean, noise)
    filtered = denoise(noisy)
    assert filtered.shape == noisy.shape
    assert noisy.astype(np.float32).var() > filtered.astype(np.float32).var()


def test_clahe_preserves_shape_and_type():
    image = np.full((80, 80, 3), 80, dtype=np.uint8)
    image[20:60, 20:60] = 180
    adjusted = apply_clahe(image)
    assert adjusted.shape == image.shape
    assert adjusted.dtype == np.uint8


def test_blur_lowers_the_sharpness_score():
    sharp = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.rectangle(sharp, (20, 20), (100, 100), (255, 255, 255), thickness=-1)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 5)
    assert variance_of_laplacian(blurred) < variance_of_laplacian(sharp)
