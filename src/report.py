import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak
)

def generate_whitepaper(metrics_df: pd.DataFrame, output_dir: str = "output"):
    """Generate a PDF whitepaper from collected metrics."""
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: processing time
    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df["frame_id"], metrics_df["processing_time_ms"])
    plt.xlabel("Frame")
    plt.ylabel("Processing Time (ms)")
    plt.title("Frame-by-Frame Processing Time")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "processing_time.png"))
    plt.close()

    # Plot 2: counts
    plt.figure(figsize=(8, 4))
    plt.plot(metrics_df["frame_id"], metrics_df["yolo_detections"],
             label="YOLO Detections")
    plt.plot(metrics_df["frame_id"], metrics_df["contour_count"],
             label="Contour Count")
    plt.xlabel("Frame")
    plt.ylabel("Count")
    plt.title("Object Count per Frame")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "counts.png"))
    plt.close()

    pdf_path = os.path.join(output_dir, "whitepaper.pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=72, leftMargin=72,
        topMargin=72, bottomMargin=72
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    body_style = styles["BodyText"]

    story = []
    story.append(Paragraph(
        "Computer Vision Object Detector & Frame-By-Frame Segmenter Pipeline",
        title_style
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Model Performance Whitepaper", styles["Heading2"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("1. Introduction", heading_style))
    story.append(Paragraph(
        "This whitepaper presents the performance evaluation of a real-time "
        "computer vision pipeline that combines a YOLOv8 object detector with "
        "classical image segmentation using adaptive thresholding and Gaussian "
        "filtering. The system is designed to extract and track target contours "
        "in video streams, overlay dynamic count vectors, and log frame-by-frame "
        "inference metrics.",
        body_style
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("2. Methodology", heading_style))
    story.append(Paragraph(
        "The pipeline processes each video frame through two parallel branches: "
        "(1) a YOLOv8 detector that predicts bounding boxes and class labels, and "
        "(2) a segmentation branch that applies a Gaussian blur, adaptive threshold, "
        "morphological closing, and contour extraction. Contours are filtered by area. "
        "A centroid tracker maintains object identities across frames. Metrics such as "
        "processing time, detection counts, and average confidence are recorded.",
        body_style
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. Results Summary", heading_style))

    total_frames = len(metrics_df)
    avg_proc = metrics_df["processing_time_ms"].mean()
    avg_fps = metrics_df["fps"].mean()
    avg_yolo = metrics_df["yolo_detections"].mean()
    avg_contours = metrics_df["contour_count"].mean()

    data = [
        ["Metric", "Value"],
        ["Total Frames", str(total_frames)],
        ["Average Processing Time (ms)", f"{avg_proc:.2f}"],
        ["Average FPS", f"{avg_fps:.2f}"],
        ["Average YOLO Detections/Frame", f"{avg_yolo:.2f}"],
        ["Average Contours/Frame", f"{avg_contours:.2f}"],
    ]

    table = Table(data, colWidths=[3 * inch, 2 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("4. Frame-by-Frame Metrics", heading_style))
    story.append(Image(
        os.path.join(plot_dir, "processing_time.png"),
        width=6 * inch, height=3 * inch
    ))
    story.append(Spacer(1, 12))
    story.append(Image(
        os.path.join(plot_dir, "counts.png"),
        width=6 * inch, height=3 * inch
    ))
    story.append(Spacer(1, 12))

    story.append(Paragraph("5. Discussion and Conclusion", heading_style))
    story.append(Paragraph(
        "The results demonstrate the feasibility of combining deep learning detection "
        "with classical segmentation for real-time contour tracking. Processing time "
        "remains stable across frames, and the dynamic count overlay provides immediate "
        "visual feedback. Future work may integrate more advanced trackers and optimize "
        "for edge devices.",
        body_style
    ))
    story.append(PageBreak())

    doc.build(story)
    return pdf_path