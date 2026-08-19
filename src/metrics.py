import pandas as pd
class MetricsCollector:
    """Collect per-frame inference metrics."""

    def __init__(self):
        self.records = []

    def add(self, frame_id, timestamp, yolo_count, contour_count,
            class_counts, avg_conf, processing_time, fps):
        self.records.append({
            "frame_id": frame_id,
            "timestamp": timestamp,
            "yolo_detections": yolo_count,
            "contour_count": contour_count,
            "class_counts": class_counts,
            "avg_confidence": avg_conf,
            "processing_time_ms": processing_time,
            "fps": fps
        })

    def to_dataframe(self):
        return pd.DataFrame(self.records)

    def save_csv(self, path):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        return df