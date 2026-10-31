import numpy as np

from src.segment import segment_leaf
from tests.synthetic import leaf_scene


def test_green_ellipse_is_found_and_a_gray_frame_is_not():
    leaf = segment_leaf(leaf_scene())
    assert leaf.found
    assert leaf.area_ratio > 0.15

    gray = np.full((300, 300, 3), 140, dtype=np.uint8)
    missed = segment_leaf(gray)
    assert not missed.found
    assert missed.area_ratio == 0.0


def test_two_leaves_are_reported():
    leaf = segment_leaf(leaf_scene(second_leaf=True))
    assert leaf.found
    assert leaf.multiple_regions
