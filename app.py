"""Portfolio demonstration for plant-leaf image processing."""

from __future__ import annotations

from pathlib import Path

import cv2
import streamlit as st

from src.charts import area_chart_png
from src.pipeline import analyze_bgr
from src.validate import ImageValidationError, load_image

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_images"
METHODOLOGY_PATH = ROOT / "docs" / "methodology.md"

SAMPLES = [
    {
        "file": "healthy-tomato-leaves.jpg",
        "label": "Tomato leaves",
        "note": (
            "Photograph by Dwight Sipler, titled as healthy tomato leaves (CC BY 2.0). "
            "The title is the photographer's description, not a result from this application."
        ),
    },
    {
        "file": "tomato-bacterial-speck.jpg",
        "label": "Tomato leaf with small spots",
        "note": (
            "Public-domain photograph by Alan Collmer of a tomato leaf with bacterial speck. "
            "The disease name is from the source caption, not from this application."
        ),
    },
    {
        "file": "pepper-bacterial-leaf-spot.jpg",
        "label": "Pepper leaf with spots",
        "note": (
            "Public-domain photograph by Scot Nelson of bacterial leaf spot on pepper. "
            "The disease name is from the source caption, not from this application."
        ),
    },
    {
        "file": "basil-leaf-spot.jpg",
        "label": "Basil leaf with spots",
        "note": (
            "Public-domain photograph by Scot Nelson of leaf spot on sweet basil. "
            "The disease name is from the source caption, not from this application."
        ),
    },
]

LIGHTING_LABELS = {"adequate": "Adequate", "dark": "Dark", "bright": "Bright"}

STYLE = """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 980px;}
    h1 {font-size: 1.75rem; font-weight: 650; color: #1c3328; letter-spacing: -0.02em;}
    h2 {font-size: 1.22rem; font-weight: 650; color: #1c3328;}
    h3 {font-size: 1.05rem; color: #24362c;}
    [data-testid="stSidebar"] {background-color: #f3f5f2; border-right: 1px solid #e1e5df;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e3e7e2;
        padding: 0.7rem 0.75rem;
        border-radius: 6px;
    }
    .note {
        background: #ffffff;
        border: 1px solid #e3e7e2;
        border-left: 3px solid #2f6b4f;
        padding: 0.85rem 1rem;
        border-radius: 4px;
        color: #1c3328;
    }
    #MainMenu, footer, header {visibility: hidden;}
</style>
"""

LIMITS = """
<div class="note">
<b>How to read this result.</b>
The percentage compares colors inside the detected leaf with a green reference.
Veins, shadows, holes, dust, and leaves that are naturally red or purple can raise it.
Faint spots, or a leaf that was not separated cleanly, can lower it.
The application does not name a disease and does not report an accuracy figure.
</div>
"""


def _rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _samples():
    return [item for item in SAMPLES if (SAMPLE_DIR / item["file"]).is_file()]


def _run(filename: str, data: bytes, note: str) -> None:
    try:
        image = load_image(filename, data)
    except ImageValidationError as exc:
        st.session_state["analysis_error"] = str(exc)
        st.session_state.pop("analysis", None)
        return
    st.session_state["analysis_error"] = None
    st.session_state["analysis"] = {
        "result": analyze_bgr(image),
        "note": note,
        "name": filename,
    }


def _show_analysis() -> None:
    error = st.session_state.get("analysis_error")
    if error:
        st.error(error)
    saved = st.session_state.get("analysis")
    if not saved:
        return
    result = saved["result"]
    stats = result.stats
    if saved["note"]:
        st.caption(saved["note"])

    st.subheader("Original and preprocessing")
    original_col, processed_col = st.columns(2)
    original_col.image(
        _rgb(result.original),
        caption="Original, with the longest side reduced to 900 pixels when needed",
        width="stretch",
    )
    processed_col.image(
        _rgb(result.preprocessed),
        caption="Noise reduced, then contrast adjusted on the lightness channel",
        width="stretch",
    )

    st.subheader("Leaf segmentation")
    leaf_col, highlight_col = st.columns(2)
    leaf_caption = "Leaf region on a white field" if stats.leaf_found else "No leaf region was kept"
    leaf_col.image(_rgb(result.segmented), caption=leaf_caption, width="stretch")
    highlight_col.image(
        _rgb(result.highlight),
        caption="Leaf outline, with non-green pixels tinted",
        width="stretch",
    )

    st.subheader("Analysis")
    if stats.leaf_found:
        affected_col, cover_col, sharp_col, light_col = st.columns(4)
        affected_col.metric("Affected area", f"{stats.affected_percent:.1f}%")
        cover_col.metric("Leaf coverage", f"{stats.leaf_area_ratio * 100:.0f}%")
        sharp_col.metric("Sharpness", f"{stats.sharpness:.0f}")
        light_col.metric("Lighting", LIGHTING_LABELS.get(stats.lighting, stats.lighting))
        chart = area_chart_png(stats)
        if chart is not None:
            st.image(chart, width="stretch")
    st.write(stats.summary)
    for warning in stats.warnings:
        st.warning(warning)
    if stats.leaf_found:
        with st.expander("Extra measurements"):
            st.write(
                f"Hue variation inside the leaf: {stats.hue_std:.1f} OpenCV hue units. "
                f"Edge density inside the leaf: {stats.edge_density:.3f}."
            )
            st.write(
                "Green reference "
                f"{stats.healthy_percent:.1f}%, yellow {stats.yellow_percent:.1f}%, "
                f"brown {stats.brown_percent:.1f}%, dark {stats.dark_percent:.1f}%, "
                f"other {stats.other_percent:.1f}%."
            )
    st.markdown(LIMITS, unsafe_allow_html=True)


def page_home() -> None:
    st.title("Plant Leaf Disease Detection")
    st.write(
        "This demonstration studies a plant-leaf photograph with classical image processing. "
        "It separates a likely leaf from the background, then estimates how much of that region "
        "differs from a green reference. The result is a visual symptom check. It is not a diagnosis."
    )
    st.subheader("What the application does")
    st.markdown(
        """
- Accepts a JPG, JPEG, or PNG photograph and rejects other files
- Resizes, reduces noise, and adjusts contrast
- Separates the largest vegetation region
- Marks yellow, brown, dark, and other off-green pixels inside that region
- Reports an approximate affected-area percentage, with the limits of the method
        """
    )
    st.subheader("Technologies")
    st.write("Python, OpenCV, NumPy, Matplotlib, Pillow, and Streamlit.")
    st.markdown(
        """
<div class="note">
Leaf Doctor, April 2022 – July 2022. Undergraduate project for the B.Tech in Electronics
and Communication Engineering at Kallam Haranadhareddy Institute of Technology.
</div>
        """,
        unsafe_allow_html=True,
    )


def page_analyze() -> None:
    st.title("Analyze Leaf")
    st.write(
        "Upload one leaf photograph, or open a sample. "
        "A single leaf on a plain background gives the most stable outline."
    )
    uploaded = st.file_uploader("Leaf photograph", type=["jpg", "jpeg", "png"])
    st.caption("JPG, JPEG, or PNG. Maximum size 8 MB.")
    samples = _samples()
    labels = ["None"] + [item["label"] for item in samples]
    choice = st.selectbox("Sample photograph", labels)
    if st.button("Run analysis", type="primary"):
        if uploaded is not None:
            _run(uploaded.name, uploaded.getvalue(), "")
        elif choice != "None":
            item = samples[labels.index(choice) - 1]
            path = SAMPLE_DIR / item["file"]
            _run(item["file"], path.read_bytes(), item["note"])
        else:
            st.session_state["analysis_error"] = "Upload a photograph or choose a sample."
            st.session_state.pop("analysis", None)
    _show_analysis()


def page_methodology() -> None:
    st.title("Methodology")
    if METHODOLOGY_PATH.is_file():
        st.markdown(METHODOLOGY_PATH.read_text(encoding="utf-8"))
    else:
        st.error("The methodology note is missing.")


def page_about() -> None:
    st.title("About Project")
    st.subheader("Academic project")
    st.write(
        "Leaf Doctor was an undergraduate project for the B.Tech in Electronics and Communication "
        "Engineering at Kallam Haranadhareddy Institute of Technology, from April 2022 to July 2022. "
        "It studies plant-leaf photographs with Python and OpenCV and reports visible discoloration. "
        "It does not use a machine-learning classifier."
    )
    st.subheader("Limits")
    st.write(
        "Color thresholds can mark veins, shadows, and natural leaf color as if they were symptoms. "
        "They can also miss faint spots. A photograph with several leaves, a busy garden background, "
        "or poor light is harder to separate. The wording on the analysis page is limited to what "
        "those steps can support."
    )


def main() -> None:
    st.set_page_config(page_title="Plant Leaf Disease Detection", layout="centered")
    st.markdown(STYLE, unsafe_allow_html=True)
    st.sidebar.markdown("**Plant Leaf Disease Detection**")
    st.sidebar.caption("Image processing demonstration")
    page = st.sidebar.radio(
        "Pages",
        ["Home", "Analyze Leaf", "Methodology", "About Project"],
        label_visibility="collapsed",
    )
    if page == "Home":
        page_home()
    elif page == "Analyze Leaf":
        page_analyze()
    elif page == "Methodology":
        page_methodology()
    else:
        page_about()


if __name__ == "__main__":
    main()
