# Computer Vision Object Detector & Segmenter Pipeline

A real-time computer vision pipeline that combines **YOLOv8 object detection** with **classical adaptive threshold segmentation + Gaussian filtering**. It extracts and tracks target contours in video streams, overlays dynamic count vectors, and logs frame‑by‑frame inference metrics. A Streamlit UI provides interactive control and visualisation.

## Features

- 🔍 YOLOv8 object detection (via Ultralytics)
- 🧩 Adaptive threshold segmentation with Gaussian blur and morphological closing
- 🎯 Contour extraction and centroid‑based tracking
- 📊 Live count overlays (class counts, contour count, FPS)
- 📈 Frame‑by‑frame metrics logging (CSV + charts)
- 📄 Automatic PDF whitepaper generation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/FarazCodes-pro/object-detector-pipeline.git
   cd object-detector-pipeline