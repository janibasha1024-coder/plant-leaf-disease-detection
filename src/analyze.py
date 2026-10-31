"""Measure color groups inside the segmented leaf."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import (
    BRIGHT_LIGHTING,
    DARK_LIGHTING,
    LAB_BROWN_A,
    LAB_HEALTHY_A,
    LAB_YELLOW_A_HIGH,
    LAB_YELLOW_A_LOW,
    LAB_YELLOW_B,
    LOCALIZED_AFFECTED,
    REPORT_COLOR_SHARE,
    SOFT_SHARPNESS,
    SPOT_BORDER_PX,
    SPOT_DELTA,
    SPOT_MIN_AREA,
    SPOT_SIGMA,
    WIDESPREAD_AFFECTED,
)
from src.segment import LeafMask


@dataclass
class RegionStats:
    leaf_found: bool
    leaf_area_ratio: float
    affected_percent: float | None
    healthy_percent: float | None
    yellow_percent: float | None
    brown_percent: float | None
    dark_percent: float | None
    other_percent: float | None
    hue_std: float | None
    edge_density: float | None
    sharpness: float
    lighting: str
    multiple_regions: bool
    warnings: list[str] = field(default_factory=list)
    summary: str = ""


def classify_lighting(image: np.ndarray, leaf_mask: np.ndarray | None = None) -> str:
    """Describe the brightness of the leaf, or of the whole frame when no leaf is found."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if leaf_mask is not None and np.any(leaf_mask):
        level = float(hsv[:, :, 2][leaf_mask > 0].mean())
    else:
        level = float(hsv[:, :, 2].mean())
    if level < DARK_LIGHTING:
        return "dark"
    if level > BRIGHT_LIGHTING:
        return "bright"
    return "adequate"


def _share(count: int, total: int) -> float | None:
    if total <= 0:
        return None
    return 100.0 * count / float(total)


def _dark_spots(image: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Compact pixels that are darker than the surrounding leaf, away from the outline."""
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    surrounding = cv2.GaussianBlur(lightness, (0, 0), SPOT_SIGMA)
    distance = cv2.distanceTransform((leaf_mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    darker = (surrounding.astype(np.int16) - lightness.astype(np.int16)) >= SPOT_DELTA
    raw = (darker & (distance >= SPOT_BORDER_PX)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    spots = np.zeros(leaf_mask.shape, dtype=bool)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= SPOT_MIN_AREA:
            spots[labels == index] = True
    return spots


def _color_groups(image: np.ndarray, leaf_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Assign every leaf pixel to one color group. Groups do not overlap."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    leaf = leaf_mask > 0
    dark = _dark_spots(image, leaf_mask) & leaf
    yellow = (
        leaf
        & ~dark
        & (b_channel >= LAB_YELLOW_B)
        & (a_channel >= LAB_YELLOW_A_LOW)
        & (a_channel < LAB_YELLOW_A_HIGH)
        & (lightness >= 55)
    )
    brown = leaf & ~dark & ~yellow & (a_channel >= LAB_BROWN_A)
    healthy = leaf & ~dark & ~yellow & ~brown & (a_channel <= LAB_HEALTHY_A)
    other = leaf & ~dark & ~yellow & ~brown & ~healthy
    return {
        "dark": dark,
        "yellow": yellow,
        "brown": brown,
        "healthy": healthy,
        "other": other,
    }


def affected_mask_from_groups(groups: dict[str, np.ndarray]) -> np.ndarray:
    affected = groups["dark"] | groups["yellow"] | groups["brown"] | groups["other"]
    return affected.astype(np.uint8) * 255


def build_summary(stats: RegionStats) -> str:
    """Plain-language description. This is not a disease name or a diagnosis."""
    if not stats.leaf_found:
        if stats.leaf_area_ratio > 0:
            coverage = (
                f" The largest green region was about {stats.leaf_area_ratio * 100:.0f}% of the image,"
                " which is too small to measure."
            )
        else:
            coverage = " No connected green region was large enough to treat as a leaf."
        return (
            "No clear leaf region could be separated from the background."
            + coverage
            + " A single leaf on a plain background is the photograph this check expects."
            " This is a visual check of the photograph, not a diagnosis."
        )

    affected = stats.affected_percent or 0.0
    if affected < LOCALIZED_AFFECTED:
        pattern = "little visible discoloration"
    elif affected < WIDESPREAD_AFFECTED:
        pattern = "localized discoloration"
    else:
        pattern = "widespread discoloration"

    if stats.leaf_area_ratio > 0.90:
        opening = (
            "The green region covers almost the whole photograph, so this is not one separated leaf. "
            "Shade between leaves can be included in the count. "
        )
    else:
        opening = f"The selected leaf region covers about {stats.leaf_area_ratio * 100:.0f}% of the image. "
    sentences = [
        (
            f"{opening}"
            f"About {affected:.1f}% of that region differs from the green reference. "
            f"The pattern looks like {pattern}."
        )
    ]
    details = []
    if stats.yellow_percent is not None and stats.yellow_percent >= REPORT_COLOR_SHARE:
        details.append(f"yellowing covers about {stats.yellow_percent:.1f}%")
    if stats.brown_percent is not None and stats.brown_percent >= REPORT_COLOR_SHARE:
        details.append(f"brown or dull patches cover about {stats.brown_percent:.1f}%")
    if stats.dark_percent is not None and stats.dark_percent >= REPORT_COLOR_SHARE:
        details.append(f"very dark spots cover about {stats.dark_percent:.1f}%")
    if stats.other_percent is not None and stats.other_percent >= REPORT_COLOR_SHARE:
        details.append(f"other off-green color covers about {stats.other_percent:.1f}%")
    if details:
        sentences.append("Inside the leaf, " + ", ".join(details) + ".")
    else:
        sentences.append("No large yellow, brown, or dark region stood out inside the leaf.")
    sentences.append("This is a visual check of the photograph, not a diagnosis of a plant disease.")
    return " ".join(sentences)


def _warnings(stats: RegionStats) -> list[str]:
    warnings = []
    if stats.sharpness < SOFT_SHARPNESS:
        warnings.append("The photograph is soft or blurred, so the outline and spot edges may be off.")
    if stats.lighting == "dark":
        warnings.append("The leaf is dark. Spot colors will be less reliable until the photo is brighter.")
    elif stats.lighting == "bright":
        warnings.append("The leaf is very bright. Pale spots may be washed out.")
    if stats.multiple_regions:
        warnings.append("More than one large green region is visible. Measurements use the largest region only.")
    if stats.leaf_found and stats.leaf_area_ratio > 0.90:
        warnings.append("The selected region covers almost the whole image, so some background may be included.")
    if stats.affected_percent is not None and stats.affected_percent > 70:
        warnings.append(
            "Most of the selected region was marked as discolored. A red or purple leaf, "
            "heavy shadow, or a loose outline can cause that."
        )
    return warnings


def measure_regions(
    image: np.ndarray,
    leaf: LeafMask,
    sharpness: float,
    lighting: str,
) -> tuple[RegionStats, np.ndarray]:
    """Count color groups on the resized original photograph."""
    affected = np.zeros(image.shape[:2], dtype=np.uint8)
    stats = RegionStats(
        leaf_found=leaf.found,
        leaf_area_ratio=leaf.area_ratio,
        affected_percent=None,
        healthy_percent=None,
        yellow_percent=None,
        brown_percent=None,
        dark_percent=None,
        other_percent=None,
        hue_std=None,
        edge_density=None,
        sharpness=sharpness,
        lighting=lighting,
        multiple_regions=leaf.multiple_regions,
    )
    if not leaf.found:
        stats.summary = build_summary(stats)
        stats.warnings = _warnings(stats)
        return stats, affected

    groups = _color_groups(image, leaf.mask)
    total = int(np.count_nonzero(leaf.mask))
    healthy_count = int(np.count_nonzero(groups["healthy"]))
    yellow_count = int(np.count_nonzero(groups["yellow"]))
    brown_count = int(np.count_nonzero(groups["brown"]))
    dark_count = int(np.count_nonzero(groups["dark"]))
    other_count = int(np.count_nonzero(groups["other"]))
    affected_count = yellow_count + brown_count + dark_count + other_count

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    leaf_pixels = leaf.mask > 0
    stats.healthy_percent = _share(healthy_count, total)
    stats.yellow_percent = _share(yellow_count, total)
    stats.brown_percent = _share(brown_count, total)
    stats.dark_percent = _share(dark_count, total)
    stats.other_percent = _share(other_count, total)
    stats.affected_percent = _share(affected_count, total)
    stats.hue_std = float(hsv[:, :, 0][leaf_pixels].std())
    stats.edge_density = float(np.count_nonzero(edges[leaf_pixels])) / float(total)
    affected = affected_mask_from_groups(groups)
    stats.summary = build_summary(stats)
    stats.warnings = _warnings(stats)
    return stats, affected


def render_segmented(image: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Place the leaf on a white field."""
    segmented = np.full_like(image, 255)
    if np.any(leaf_mask):
        selected = leaf_mask > 0
        segmented[selected] = image[selected]
    return segmented


def render_highlight(image: np.ndarray, leaf_mask: np.ndarray, affected: np.ndarray) -> np.ndarray:
    """Outline the leaf and tint pixels that were not counted as green."""
    highlighted = image.copy()
    if affected is not None and np.any(affected):
        tint = np.zeros_like(image)
        tint[:, :] = (36, 48, 196)
        blended = cv2.addWeighted(image, 0.45, tint, 0.55, 0)
        selected = affected > 0
        highlighted[selected] = blended[selected]
    if leaf_mask is not None and np.any(leaf_mask):
        contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, (248, 248, 246), 2)
    return highlighted
