import json
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"

st.set_page_config(page_title="Computer Vision Obstacle Detector", layout="wide")
st.title("Prototype 2: Computer Vision Obstacle Detector")

st.markdown("This prototype **simulates** a real-time object detector (e.g., YOLO) by overlaying known detections on synthetic scenes. It demonstrates the **UI/UX, thresholding, and alerting** you'd expect in a deployed system. Replace the detections with a live model to go from mock → production.")

# Sidebar controls
conf_thr = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
show_labels = st.sidebar.checkbox("Show class labels", value=True)
alert_threshold = st.sidebar.slider("High-risk alert if N+ obstacles", min_value=1, max_value=10, value=5, step=1)

# Load scenes/detections
det_json = json.loads((ASSETS / "detections.json").read_text())
scene_names = [Path(s["image"]).name for s in det_json]
scene_choice = st.selectbox("Select scene", scene_names, index=0)

scene_data = next(s for s in det_json if Path(s["image"]).name == scene_choice)
img = Image.open(ASSETS / scene_data["image"]).convert("RGB")
draw = ImageDraw.Draw(img)

# Draw detections above threshold
kept = [d for d in scene_data["detections"] if d["confidence"] >= conf_thr]

# Color map
colors = {
    "vessel": (0, 200, 255),
    "buoy": (255, 120, 0),
    "iceberg": (0, 255, 180),
    "bird_flock": (255, 255, 0),
}

for det in kept:
    x1, y1, x2, y2 = det["bbox"]
    cls = det["class"]
    conf = det["confidence"]
    color = colors.get(cls, (255, 255, 255))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if show_labels:
        label = f"{cls} {conf:.2f}"
        tw, th = draw.textlength(label), 14
        draw.rectangle([x1, y1-18, x1+tw+8, y1], fill=color)
        draw.text((x1+4, y1-16), label, fill=(0,0,0))

# Risk summary
col1, col2 = st.columns([3,1])
with col1:
    st.image(img, caption=f"Detections ≥ {conf_thr:.2f}. Replace with live model outputs to deploy.")

with col2:
    st.subheader("Summary")
    st.metric("Detections (≥ thr.)", len(kept))
    by_cls = {}
    for d in kept:
        by_cls[d["class"]] = by_cls.get(d["class"], 0) + 1
    for k, v in by_cls.items():
        st.write(f"- **{k}**: {v}")

    if len(kept) >= alert_threshold:
        st.error(" High-risk: too many obstacles in scene")
    else:
        st.success("Risk acceptable")

st.caption("Note: This is a mock CV pipeline for demo. In production, we will use (e.g., YOLOv8/Detectron2) and fuse with radar/IR for robustness.")
