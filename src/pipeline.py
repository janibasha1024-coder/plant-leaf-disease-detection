"""Run the leaf photograph through the processing steps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analyze import (
    RegionStats,
    classify_lighting,
    measure_regions,
    render_highlight,
    render_segmented,
)
from src.preprocess import apply_clahe, denoise, resize_max_side, variance_of_laplacian
from src.segment import segment_leaf


@dataclass
class AnalysisResult:
    original: np.ndarray
    preprocessed: np.ndarray
    leaf_mask: np.ndarray
    segmented: np.ndarray
    highlight: np.ndarray
    affected_mask: np.ndarray
    stats: RegionStats


def analyze_bgr(image: np.ndarray) -> AnalysisResult:
    """Resize, denoise, find the leaf, and measure color on the resized original.

    Contrast adjustment is returned for display. It is not used for the area percentage,
    so the percentage stays tied to the photograph rather than to the enhanced copy.
    """
    original = resize_max_side(image)
    denoised = denoise(original)
    preprocessed = apply_clahe(denoised)
    leaf = segment_leaf(denoised)
    sharpness = variance_of_laplacian(original)
    lighting = classify_lighting(original, leaf.mask if leaf.found else None)
    stats, affected = measure_regions(original, leaf, sharpness, lighting)
    return AnalysisResult(
        original=original,
        preprocessed=preprocessed,
        leaf_mask=leaf.mask,
        segmented=render_segmented(original, leaf.mask),
        highlight=render_highlight(original, leaf.mask, affected),
        affected_mask=affected,
        stats=stats,
    )
