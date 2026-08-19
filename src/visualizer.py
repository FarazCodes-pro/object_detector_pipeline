import cv2
from collections import Counter

def draw_detections(frame, detections):
    """Draw YOLO bounding boxes and labels."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_contours(frame, contour_items, color=(255, 0, 0)):
    """Draw extracted contours and centroids."""
    for item in contour_items:
        cv2.drawContours(frame, [item["contour"]], -1, color, 2)
        x, y, w, h = item["bbox"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
        cv2.circle(frame, item["centroid"], 4, (0, 0, 255), -1)
    return frame

def draw_count_overlay(frame, yolo_counts, contour_count, fps=None):
    """Overlay dynamic count vectors."""
    y_offset = 30
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_offset += 30

    cv2.putText(frame, f"Contours: {contour_count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y_offset += 30

    for cls, count in yolo_counts.items():
        text = f"{cls}: {count}"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30

    return frame