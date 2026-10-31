import numpy as np

from src.analyze import build_summary, classify_lighting, measure_regions
from src.pipeline import analyze_bgr
from src.segment import segment_leaf
from tests.synthetic import leaf_scene


def test_spots_raise_the_affected_share_and_groups_sum_to_the_leaf():
    plain = analyze_bgr(leaf_scene())
    spotted = analyze_bgr(leaf_scene(spots=True))
    assert plain.stats.leaf_found
    assert spotted.stats.affected_percent > plain.stats.affected_percent
    shares = [
        spotted.stats.healthy_percent,
        spotted.stats.yellow_percent,
        spotted.stats.brown_percent,
        spotted.stats.dark_percent,
        spotted.stats.other_percent,
    ]
    assert abs(sum(shares) - 100.0) < 0.2
    assert "not a diagnosis" in spotted.stats.summary


def test_gray_frame_has_no_percentage():
    result = analyze_bgr(np.full((300, 300, 3), 150, dtype=np.uint8))
    assert not result.stats.leaf_found
    assert result.stats.affected_percent is None
    assert "No clear leaf" in result.stats.summary
    assert not np.any(result.affected_mask)


def test_soft_and_dark_images_add_warnings():
    soft = analyze_bgr(leaf_scene(blur=8))
    assert any("blurred" in warning for warning in soft.stats.warnings)

    dark = leaf_scene() // 8
    result = analyze_bgr(dark)
    assert result.stats.lighting == "dark"
    assert any("dark" in warning for warning in result.stats.warnings)


def test_summary_for_a_missing_leaf_states_the_limit():
    image = np.full((300, 300, 3), 160, dtype=np.uint8)
    leaf = segment_leaf(image)
    stats, _ = measure_regions(image, leaf, sharpness=100, lighting=classify_lighting(image))
    text = build_summary(stats)
    assert "not a diagnosis" in text
    assert "accuracy" not in text.lower()
