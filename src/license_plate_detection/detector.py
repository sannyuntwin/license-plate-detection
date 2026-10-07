from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path="models/yolov8n.pt", confidence=0.4, class_id=None):
        self.model = YOLO(model_path)
        self.confidence = confidence

        names = self.model.names
        if isinstance(names, list):
            names = dict(enumerate(names))

        if class_id is not None:
            if class_id not in names:
                raise ValueError(f"Class ID {class_id} is not present in model {model_path}")
            self.class_ids = [class_id]
        else:
            self.class_ids = [
                class_index
                for class_index, name in names.items()
                if "license plate" in str(name).lower()
                or "number plate" in str(name).lower()
                or str(name).lower() in {"plate", "license_plate", "number_plate"}
            ]
            if not self.class_ids:
                raise ValueError(
                    f"Model {model_path} has no license-plate class. "
                    "Use plate-specific weights or provide --class-id."
                )

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            classes=self.class_ids,
            verbose=False,
        )
        boxes = []

        for r in results:
            if r.boxes is not None:
                for box in r.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    boxes.append((x1, y1, x2, y2))

        return boxes
