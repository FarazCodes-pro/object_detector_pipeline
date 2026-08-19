import os
import tempfile
import time
from collections import Counter

import cv2
import pandas as pd
import streamlit as st
import yaml

from src.detector import YOLODetector
from src.segmenter import segment_frame
from src.tracker import CentroidTracker
from src.metrics import MetricsCollector
from src.visualizer import draw_detections, draw_contours, draw_count_overlay
from src.report import generate_whitepaper

st.set_page_config(
    page_title="CV Object Detector & Segmenter",
    layout="wide"
)

@st.cache_resource
def load_detector(model_path, conf_threshold, iou_threshold):
    """Load and cache the YOLO detector."""
    return YOLODetector(
        model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold
    )

def load_config():
    """Load configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()

    # Session state to persist results across reruns
    if "processed" not in st.session_state:
        st.session_state.processed = False
        st.session_state.metrics_df = None
        st.session_state.csv_path = None
        st.session_state.pdf_path = None

    st.title("Computer Vision Object Detector & Frame-By-Frame Segmenter Pipeline")
    st.markdown(
        "Real-time YOLO detection + adaptive threshold segmentation with "
        "contour tracking and performance metrics."
    )

    # ---------- Sidebar ----------
    with st.sidebar:
        st.header("Settings")

        input_source = st.radio(
            "Input Source",
            ["Upload Video", "Webcam"],
            index=0
        )
        uploaded_file = None
        if input_source == "Upload Video":
            uploaded_file = st.file_uploader(
                "Choose a video",
                type=["mp4", "avi", "mov", "mkv"]
            )
        else:
            st.info("Webcam will start when processing begins.")

        st.subheader("Model Parameters")
        model_path = st.text_input(
            "Model Path",
            value=config["model"]["path"]
        )
        conf_threshold = st.slider(
            "Confidence Threshold",
            0.1, 0.9,
            float(config["model"]["conf_threshold"]),
            0.05
        )
        iou_threshold = st.slider(
            "IoU Threshold",
            0.1, 0.9,
            float(config["model"]["iou_threshold"]),
            0.05
        )
        enable_yolo = st.checkbox("Enable YOLO Detection", value=True)

        st.subheader("Segmentation Parameters")
        enable_segmentation = st.checkbox(
            "Enable Adaptive Threshold Segmentation",
            value=True
        )
        gaussian_kernel = st.slider(
            "Gaussian Kernel Size",
            3, 21, step=2,
            value=int(config["preprocess"]["gaussian_kernel"][0])
        )
        adaptive_block_size = st.slider(
            "Adaptive Threshold Block Size",
            3, 51, step=2,
            value=int(config["preprocess"]["adaptive_block_size"])
        )
        adaptive_C = st.slider(
            "Adaptive Threshold C",
            -10, 10,
            value=int(config["preprocess"]["adaptive_C"])
        )
        morph_kernel = st.slider(
            "Morph Kernel Size",
            1, 15, step=2,
            value=int(config["preprocess"]["morph_kernel"][0])
        )
        min_area = st.slider(
            "Min Contour Area",
            100, 5000,
            value=int(config["preprocess"]["min_contour_area"]),
            step=100
        )

        st.subheader("Tracking & Limits")
        enable_tracking = st.checkbox("Enable Centroid Tracking", value=True)
        max_frames = st.slider(
            "Max Frames to Process",
            10, 500, 100, step=10
        )
        process_button = st.button("Start Processing")

    # ---------- Main Layout ----------
    col1, col2 = st.columns([3, 1])
    with col1:
        stframe = st.empty()
    with col2:
        st.subheader("Live Counts")
        count_placeholder = st.empty()
        st.subheader("Metrics")
        metrics_placeholder = st.empty()

    # ---------- Processing ----------
    if process_button:
        video_path = None

        if input_source == "Upload Video":
            if uploaded_file is None:
                st.error("Please upload a video file.")
                return
            # FIX: Use a context manager to close the file handle after writing
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            cap = cv2.VideoCapture(video_path)
        else:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam.")
                return

        detector = None
        if enable_yolo:
            detector = load_detector(
                model_path, conf_threshold, iou_threshold
            )

        tracker = CentroidTracker(
            max_disappeared=5,
            max_distance=50
        )
        metrics_collector = MetricsCollector()

        frame_id = 0
        start_time = time.time()
        progress_bar = st.progress(0)

        while cap.isOpened():
            if frame_id >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.time()

            detections = []
            contours = []

            if enable_yolo and detector is not None:
                detections = detector.detect(frame)

            if enable_segmentation:
                _, contours = segment_frame(
                    frame,
                    gaussian_kernel=(gaussian_kernel, gaussian_kernel),
                    gaussian_sigma=0,
                    adaptive_block_size=adaptive_block_size,
                    adaptive_C=adaptive_C,
                    morph_kernel=(morph_kernel, morph_kernel),
                    min_contour_area=min_area
                )

            yolo_counts = Counter(
                [d["class_name"] for d in detections]
            )
            avg_conf = (
                sum(d["confidence"] for d in detections) / len(detections)
                if detections else 0.0
            )

            if enable_tracking and contours:
                centroids = [item["centroid"] for item in contours]
                _ = tracker.update(centroids)

            annotated = frame.copy()
            if detections:
                annotated = draw_detections(annotated, detections)
            if contours:
                annotated = draw_contours(annotated, contours)
            annotated = draw_count_overlay(
                annotated,
                yolo_counts,
                len(contours),
                fps=1.0 / (time.time() - frame_start + 1e-6)
            )

            annotated_rgb = cv2.cvtColor(
                annotated, cv2.COLOR_BGR2RGB
            )
            stframe.image(
                annotated_rgb,
                channels="RGB",
                use_container_width=True
            )

            count_text = (
                f"**YOLO Detections:** {len(detections)}  \n"
                f"**Contours:** {len(contours)}  \n"
            )
            for cls, cnt in yolo_counts.items():
                count_text += f"**{cls}:** {cnt}  \n"
            count_placeholder.markdown(count_text)

            processing_time = (time.time() - frame_start) * 1000
            fps = 1.0 / (time.time() - frame_start + 1e-6)

            metrics_collector.add(
                frame_id,
                time.time() - start_time,
                len(detections),
                len(contours),
                dict(yolo_counts),
                avg_conf,
                processing_time,
                fps
            )

            metrics_placeholder.markdown(
                f"Frame: {frame_id}  \n"
                f"Processing: {processing_time:.1f} ms  \n"
                f"FPS: {fps:.1f}"
            )

            frame_id += 1
            progress_bar.progress(min(frame_id / max_frames, 1.0))

        cap.release()
        # Now safe to delete the temporary file on Windows
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)

        metrics_df = metrics_collector.to_dataframe()
        os.makedirs("output", exist_ok=True)
        csv_path = "output/metrics.csv"
        metrics_collector.save_csv(csv_path)

        st.session_state.processed = True
        st.session_state.metrics_df = metrics_df
        st.session_state.csv_path = csv_path
        st.session_state.pdf_path = None

        st.success(
            f"Processed {frame_id} frames. Metrics saved to `{csv_path}`."
        )

    # ---------- Results ----------
    if st.session_state.processed and st.session_state.metrics_df is not None:
        metrics_df = st.session_state.metrics_df

        st.subheader("Frame-by-Frame Metrics")
        st.dataframe(metrics_df)

        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.line_chart(metrics_df[["processing_time_ms"]])
        with col_chart2:
            st.line_chart(
                metrics_df[["yolo_detections", "contour_count"]]
            )

        if (
            st.session_state.csv_path
            and os.path.exists(st.session_state.csv_path)
        ):
            with open(st.session_state.csv_path, "rb") as f:
                st.download_button(
                    "Download Metrics CSV",
                    f,
                    file_name="metrics.csv"
                )

        if st.button("Generate Whitepaper PDF"):
            pdf_path = generate_whitepaper(
                metrics_df, output_dir="output"
            )
            st.session_state.pdf_path = pdf_path
            st.success(f"Whitepaper generated at `{pdf_path}`")

        if (
            st.session_state.pdf_path
            and os.path.exists(st.session_state.pdf_path)
        ):
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button(
                    "Download Whitepaper PDF",
                    f,
                    file_name="whitepaper.pdf"
                )

if __name__ == "__main__":
    main()