"""
Fruit Ripeness Detector — Streamlit App
Powered by YOLOv8n-OBB (Oriented Bounding Boxes)

To run locally:
    pip install streamlit ultralytics opencv-python-headless pillow
    streamlit run app.py

To run inside Google Colab (see instructions at the bottom of this file).
"""

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
import time

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Fruit Ripeness Detector",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
        background: linear-gradient(90deg, #e35d5b, #f9a03f, #4caf50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f7f7f9;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #eee;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">🍎🍌🥭 Fruit Ripeness Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">YOLOv8n-OBB model trained to detect apples, bananas, and mangoes at three ripeness stages</p>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar — settings
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    weights_path = st.text_input(
        "Model weights path (.pt)",
        value="best.pt",
        help="Path to your trained YOLOv8-OBB weights file (e.g. runs/obb/.../weights/best.pt)",
    )

    conf_threshold = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.1, 1.0, 0.45, 0.05)

    st.markdown("---")
    st.header("📤 Input")
    mode = st.radio("Choose input type", ["Image", "Video"], horizontal=True)

    st.markdown("---")
    st.header("ℹ️ About")
    st.caption(
        "This app detects fruit ripeness stages using a custom-trained "
        "YOLOv8 nano Oriented Bounding Box model. Upload a photo or video "
        "of apples, bananas, or mangoes to see the model in action."
    )


# ------------------------------------------------------------------
# Model loading (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    # If the weights file isn't in the repo (common: .pt files are too big for GitHub),
    # try downloading it from a URL stored in Streamlit secrets (WEIGHTS_URL).
    if not os.path.exists(path):
        weights_url = st.secrets.get("WEIGHTS_URL", "") if hasattr(st, "secrets") else ""
        if weights_url:
            import urllib.request
            with st.spinner("Downloading model weights (first run only)..."):
                urllib.request.urlretrieve(weights_url, path)
        else:
            return None
    if not os.path.exists(path):
        return None
    return YOLO(path)


model = load_model(weights_path)

if model is None:
    st.error(
        f"⚠️ Could not find weights file at `{weights_path}`. "
        "Update the path in the sidebar (it should point to your best.pt file)."
    )
    st.stop()

CLASS_COLORS = {
    "apple-ripe": "#4caf50",
    "apple-unripe": "#8bc34a",
    "apple-overripe": "#795548",
    "banana-ripe": "#ffeb3b",
    "banana-unripe": "#cddc39",
    "banana-overripe": "#6d4c41",
    "mango-ripe": "#ff9800",
    "mango-unripe": "#009688",
    "mango-overripe": "#5d4037",
}

# ------------------------------------------------------------------
# Helper: run detection and build results table
# ------------------------------------------------------------------
def run_detection(image_bgr):
    results = model.predict(source=image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
    r = results[0]
    annotated = r.plot()  # BGR image with boxes drawn
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    rows = []
    if r.obb is not None and len(r.obb) > 0:
        for cls_id, conf in zip(r.obb.cls.tolist(), r.obb.conf.tolist()):
            name = model.names[int(cls_id)]
            rows.append({"Class": name, "Confidence": f"{conf * 100:.1f}%"})
    elif r.boxes is not None and len(r.boxes) > 0:
        for cls_id, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            name = model.names[int(cls_id)]
            rows.append({"Class": name, "Confidence": f"{conf * 100:.1f}%"})

    return annotated_rgb, rows


# ------------------------------------------------------------------
# Image mode
# ------------------------------------------------------------------
if mode == "Image":
    uploaded_file = st.file_uploader(
        "Upload a fruit image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original")
            st.image(image, use_container_width=True)

        with st.spinner("Running detection..."):
            start = time.time()
            annotated_rgb, rows = run_detection(image_bgr)
            elapsed = (time.time() - start) * 1000

        with col2:
            st.subheader("🎯 Detection Result")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.markdown(
            f'<div class="metric-card"><h3>{len(rows)}</h3>Objects detected</div>',
            unsafe_allow_html=True,
        )
        m2.markdown(
            f'<div class="metric-card"><h3>{elapsed:.1f} ms</h3>Inference time</div>',
            unsafe_allow_html=True,
        )
        avg_conf = (
            np.mean([float(r["Confidence"].strip("%")) for r in rows]) if rows else 0
        )
        m3.markdown(
            f'<div class="metric-card"><h3>{avg_conf:.1f}%</h3>Avg. confidence</div>',
            unsafe_allow_html=True,
        )

        if rows:
            st.markdown("### 📋 Detected Objects")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.warning("No objects detected. Try lowering the confidence threshold in the sidebar.")
    else:
        st.info("👆 Upload an image to get started.")

# ------------------------------------------------------------------
# Video mode
# ------------------------------------------------------------------
else:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(uploaded_video)

        if st.button("▶️ Run detection on video"):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress = st.progress(0, text="Processing video...")
            frame_idx = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                annotated = results[0].plot()
                writer.write(annotated)

                frame_idx += 1
                if total_frames > 0:
                    progress.progress(min(frame_idx / total_frames, 1.0), text=f"Processing frame {frame_idx}/{total_frames}")

            cap.release()
            writer.release()
            progress.empty()

            st.success("✅ Done! Here is the annotated video:")
            st.video(out_path)
    else:
        st.info("👆 Upload a video to get started.")

# ------------------------------------------------------------------
st.markdown(
    '<p class="footer">Fruit Ripeness Detection · YOLOv8n-OBB · Built with Streamlit</p>',
    unsafe_allow_html=True,
)
