"""Small chart of the color groups inside the leaf."""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analyze import RegionStats


def area_chart_png(stats: RegionStats) -> bytes | None:
    """Return a PNG chart, or None when the leaf was not measured."""
    if not stats.leaf_found:
        return None
    labels = ["Green reference", "Yellow", "Brown", "Dark", "Other"]
    values = [
        stats.healthy_percent or 0.0,
        stats.yellow_percent or 0.0,
        stats.brown_percent or 0.0,
        stats.dark_percent or 0.0,
        stats.other_percent or 0.0,
    ]
    colors = ["#2F6B4F", "#C4A35A", "#8C5A3C", "#4A5248", "#8D938C"]
    figure, axis = plt.subplots(figsize=(6.4, 2.6), dpi=130)
    figure.patch.set_facecolor("#F6F5F2")
    axis.set_facecolor("#F6F5F2")
    positions = range(len(labels))
    axis.barh(list(positions), values, color=colors, height=0.62)
    axis.set_yticks(list(positions), labels)
    axis.set_xlim(0, 100)
    axis.set_xlabel("Share of the selected leaf (%)")
    axis.invert_yaxis()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=9)
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    return buffer.getvalue()
