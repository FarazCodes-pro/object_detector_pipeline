from ultralytics import YOLO

class YOLODetector:
    """Wrapper around Ultralytics YOLO for object detection."""

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.conf = conf_threshold
        self.iou = iou_threshold
        self.class_names = self.model.names

    def detect(self, frame):
        """Run inference on a BGR frame and return detections."""
        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        detections = []

        if len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = self.class_names.get(cls_id, str(cls_id))

                    detections.append({
                        "bbox": [int(xyxy[0]), int(xyxy[1]),
                                 int(xyxy[2]), int(xyxy[3])],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name
                    })

        return detections