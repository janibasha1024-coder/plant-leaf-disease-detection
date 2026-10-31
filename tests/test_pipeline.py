from pathlib import Path

from src.charts import area_chart_png
from src.pipeline import analyze_bgr
from src.validate import load_image
from tests.synthetic import leaf_scene

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "sample_images"


def test_chart_is_a_png_when_a_leaf_is_found():
    result = analyze_bgr(leaf_scene(spots=True))
    chart = area_chart_png(result.stats)
    assert chart is not None
    assert chart.startswith(b"\x89PNG")
    assert area_chart_png(analyze_bgr(leaf_scene() * 0).stats) is None


def test_sample_photographs_can_be_measured():
    paths = sorted(path for path in SAMPLES.glob("*.jpg") if not path.name.startswith("._"))
    assert len(paths) >= 3
    for path in paths:
        result = analyze_bgr(load_image(path.name, path.read_bytes()))
        assert result.original.ndim == 3
        assert max(result.original.shape[:2]) <= 900
        assert result.preprocessed.shape == result.original.shape
        assert "diagnosis" in result.stats.summary
        if result.stats.leaf_found:
            assert 0.0 <= result.stats.affected_percent <= 100.0
